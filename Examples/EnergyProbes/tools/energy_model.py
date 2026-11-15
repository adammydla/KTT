#!/usr/bin/env python3
"""The energy model: an energy per CUPTI event, fitted over the probes, and how well it ranks real kernels.

1. Fit. Every probe gives its energy per access (the slope of its fit across duties in <power>_energy_table.json, or
   the energy of its duty-100 row if it was measured at one duty) and its CUPTI counts per access (a --profile run,
   <counters>.csv). Non-negative least squares on relative errors gives the energy of one event of each kind
   (energy_events.py): a warp instruction of each pipe, a sector served by L1, L2 or DRAM, an L2 request, a shared
   wavefront and a replay for a bank conflict, a constant-cache request and miss, a nanosecond of DRAM kept awake.
2. Score. For every configuration of the --validation runs (<validation>_validation.csv) the dynamic energy per launch
   is the sum of its counts times those energies; the total adds the sleep baseline over the kernel's time. For picking
   the configuration that uses the least energy, what matters is the scatter of the error between the configurations
   of one kernel, not its bias. Per kernel and run: the standard deviation (scatter) and mean (bias) of the relative
   error of the dynamic energy, the mean absolute error of the total energy, and the regret (how much more energy the
   configuration the model ranks best uses than the measured best), next to a ranking by kernel time alone (constant
   power), the reference to beat.

    python3 tools/energy_model.py --power Probes --counters Profile [--validation Validation[,Validation2]]

tools/build_model.py writes the same fit into a model file for the predictors.
"""

import argparse
import collections
import itertools
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_table  # noqa: E402
import energy_events  # noqa: E402
from energy_events import EVENTS, add_busy, counter, events, number, read_rows, value_of  # noqa: E402,F401

# A probe of the fit: its label ("level operation wWIDTH"), energy per access (pJ) and events per access.
Probe = collections.namedtuple("Probe", "label energy counts")

# The FFMA probe that computes x * x + c reads one register twice and takes c from the constant bank, and costs 15-20 %
# less per FFMA than the FMAs of real kernels (acc = a * b + acc), which read three different registers (ffma3). It
# stays out of the fit; the ffma-loop probe, which computes x * x + c too, is re-expressed with it (see probe_table).
TWO_REGISTER_FFMA = "compute ffma"

# An event enters the fit only if some probe executes it at least this often per access of one thread (one warp
# instruction per 10 warp accesses): events that only occur a few times per launch (the division in the probes' setup)
# would otherwise get absurd energies that explain other probes.
MIN_EVENTS_PER_ACCESS = 0.1 / 32

# The counters the events are computed from, with the events each one feeds. A counter that a run did not collect counts
# as 0, which silently removes its events, so check_same_counters() compares the runs.
COUNTER_EVENTS = {name: entry["feeds"] for name, entry in counter_table.ENTRIES.items()
                  if "feeds" in entry and not {"alias_of", "alternative"} & set(entry)}


def key_of(row):
    return row["level"], row["operation"], int(row["width_bytes"])


def label_of(key):
    return f"{key[0]} {key[1]}" + (f" w{key[2]}" if key[2] else "")


def probe_rows(prefix):
    """The probe rows of a run at duty 100 (on random data and all SMs: runs of earlier versions had other rows too)."""
    return [r for r in read_rows(prefix + ".csv") if r["duty_pct"] == "100" and r.get("payload", "random") == "random"
            and r.get("active_sms", "all") == "all"]


def probe_table(power_prefixes, counters_prefixes, busy_scale=None):
    """The probes (Probe tuples) and the energy table of the last power run, its baseline slope the mean over the runs.
    With several power runs, a probe's energy is the mean over the runs that have it; with several counter runs, its
    counts come from the first that has it."""
    tables, collected, times = [], collections.defaultdict(list), collections.defaultdict(list)
    for prefix in power_prefixes:
        with open(prefix + "_energy_table.json") as file:
            tables.append(json.load(file))
        duties = collections.Counter(key_of(r) for r in read_rows(prefix + ".csv") if r["duty_pct"] not in ("-", "")
                                     and r.get("payload", "random") == "random" and r.get("active_sms", "all") == "all")
        full_rate = {key_of(r): r for r in probe_rows(prefix)}
        run = {}
        for key, row in full_rate.items():
            if number(row["accesses_per_launch"]) > 0 and number(row["active_s"]) > 0:
                times[key].append(number(row["active_s"]) / number(row["accesses_per_launch"]) * 1e9)
            # A probe measured at one duty (the dependent loads, whose rate varies with the parallelism instead, and
            # the write-path and misaligned grids) enters with its row's energy, which keeps power that does not
            # scale with the rate.
            if duties[key] == 1 and number(row["pj_per_access"]) > 0 and row.get("valid", "1") == "1":
                run[key] = number(row["pj_per_access"])
        for fit in tables[-1]["fits"]:
            if fit.get("payload", "random") != "random" or fit["pj_per_access"] is None:
                continue
            key = (fit["level"], fit["operation"], int(fit["width_bytes"]))
            run[key] = float(fit["pj_per_access"])
            # A fit whose fixed power exceeds half the full-rate dynamic power is not a line through the rows (the
            # constant L1.5, a shared bottleneck whose request rate does not follow the duty): the row's energy.
            row = full_rate.get(key)
            if row is not None and number(row["pj_per_access"]) > 0 and fit["fixed_w"] is not None \
                    and abs(fit["fixed_w"]) > 0.5 * (number(row["power_w"]) - number(row["baseline_w"])):
                run[key] = number(row["pj_per_access"])
                print(f"{prefix}: {label_of(key)}: fit not linear (fixed {fit['fixed_w']:.1f} W), using duty 100: "
                      f"{run[key]:.2f} instead of {float(fit['pj_per_access']):.2f} pJ per access")
        # The ffma-loop probe's FFMAs as ones with three registers, so that its branches are priced on the FFMA of the fit.
        loop, two, three = (("compute", name, 0) for name in ("ffma-loop", "ffma", "ffma3"))
        if loop in run and two in run and three in run:
            run[loop] += run[three] - run[two]
        for key, energy in run.items():
            collected[key].append(energy)
    energies = {key: float(np.mean(values)) for key, values in collected.items()}
    # The fit minimises relative errors: a probe at or below 0 (a weak probe whose fit came out negative from noise)
    # would enter with a huge or negative weight, so it stays out.
    for key in sorted(key for key, energy in energies.items() if not energy > 0):
        print(f"{label_of(key)}: energy {energies.pop(key):.2f} pJ per access is not above 0, left out of the fit")
    table = dict(tables[-1])
    table["baseline"] = dict(table["baseline"], slope_w_per_c=float(np.mean([t["baseline"]["slope_w_per_c"] for t in tables])))

    probes = {}
    for row in (r for prefix in counters_prefixes for r in probe_rows(prefix)):
        key = key_of(row)
        accesses = number(row["accesses_per_launch"])
        if key in energies and key not in probes and accesses > 0 and row.get("smsp__inst_executed.sum"):
            counts = add_busy({name: value / accesses for name, value in events(row).items()},
                              float(np.mean(times[key])) if times[key] else math.nan, busy_scale)
            probes[key] = Probe(label_of(key), energies[key], counts)
    return [p for p in probes.values() if p.label != TWO_REGISTER_FFMA or ("compute", "ffma3", 0) not in probes], table


def fit(probes):
    """Energies per event (pJ) by non-negative least squares on relative residuals, the model's energy of every probe,
    and the events no probe exercises (they get 0)."""
    a = np.array([[p.counts[name] for name in EVENTS] for p in probes])
    b = np.array([p.energy for p in probes])
    used = a.max(axis=0) >= MIN_EVENTS_PER_ACCESS
    scaled = a[:, used] / b[:, None]
    norms = scaled.max(axis=0)
    solution, _ = nnls(scaled / norms, np.ones(len(probes)))
    coefficients = np.zeros(len(EVENTS))
    coefficients[used] = solution / norms
    return dict(zip(EVENTS, coefficients)), a @ coefficients, {name for name, flag in zip(EVENTS, used) if not flag}


def load_validation(prefix, table):
    """The configurations of a --validate run that have power and counters, with their measured energies per launch
    recomputed from the run's baseline moved to each row's temperature. When that run's sleeps spanned too few degrees
    for a slope of their own, the slope of the probe run's baseline (`table`) is used."""
    rows = [row for row in read_rows(prefix + "_validation.csv") if row["valid"] == "1" and "smsp__inst_executed.sum" in row
            and not math.isnan(number(row["dynamic_energy_j"])) and counter(row, "smsp__inst_executed.sum") > 0
            and row.get("repeat") in (None, "", "0")]
    if not rows:
        raise SystemExit(f"{prefix}: the validation run has no rows with power and counters")
    slope = number(rows[0].get("baseline_slope_w_per_c"))
    if not slope > 0:
        slope = float(table["baseline"]["slope_w_per_c"])
        print(f"{prefix}: the run's baseline has no temperature model; using the probe run's {slope:.2f} W/C")
    for row in rows:
        base = number(row["baseline_mean_w"]) + slope * (number(row["temperature_c"]) - number(row["baseline_temperature_c"]))
        row["baseline_w"] = base
        row["dynamic_energy_j"] = (number(row["power_w"]) - base) * number(row["wall_s"])
        row["energy_j"] = base * number(row["kernel_s"]) + row["dynamic_energy_j"]
    return rows


def counters_present(rows):
    """The counters of COUNTER_EVENTS that some row has a value of (under their name or an alias)."""
    return {name for name in COUNTER_EVENTS if any(not math.isnan(value_of(row, name)) for row in rows)}


def check_same_counters(sources):
    """Every counter an event needs must be in all runs (the probes' counter runs, the validation runs) or in none: a
    counter that one of them lacks counts as 0 there, and the model would price the others' events into what remains."""
    present = {name: counters_present(rows) for name, rows in sources.items() if rows}
    partial = sorted(set.union(*present.values()) - set.intersection(*present.values())) if present else []
    if partial:
        raise SystemExit("counters the model needs are in some runs and not in others (they would count as 0 there):\n"
                         + "\n".join(f"  {name} ({COUNTER_EVENTS[name]}): not in "
                                     + ", ".join(s for s, names in present.items() if name not in names) for name in partial)
                         + "\nprofile the runs with the same counters (tools/counters.json)")


def predict(rows, energies, busy_scale=None):
    """Per configuration: (kernel, measured dynamic, predicted dynamic, measured total, predicted total, energy at
    the mean power of all configurations over its own time) in J per launch."""
    mean_power = np.mean([number(r["energy_j"]) / number(r["kernel_s"]) for r in rows])
    results = []
    for row in rows:
        counts = add_busy(events(row), number(row["kernel_s"]) * 1e9, busy_scale)
        dynamic = sum(counts[name] * energies[name] for name in EVENTS) * 1e-12
        results.append((row["kernel"], number(row["dynamic_energy_j"]), dynamic, number(row["energy_j"]),
                        number(row["baseline_w"]) * number(row["kernel_s"]) + dynamic, mean_power * number(row["kernel_s"])))
    return results


BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, float("inf"))]


def score(validations, energies, busy_scale=None):
    """Prints how the model ranks the configurations of each kernel of each run ({run: rows}); returns the lines'
    numbers: (run, kernel, n, scatter, bias, total MAPE, regret, regret by time alone)."""
    lines = []
    pairs = {b: [0, 0, 0] for b in BINS}  # model right, time-only right, pairs
    for run, rows in validations.items():
        results = predict(rows, energies, busy_scale)
        for kernel in dict.fromkeys(r[0] for r in results):
            _, measured_dynamic, dynamic, measured, predicted, constant = map(np.array, zip(*[r for r in results if r[0] == kernel]))
            error = dynamic / measured_dynamic - 1
            lines.append((os.path.basename(run), kernel, len(error), error.std(), error.mean(),
                          np.abs(predicted / measured - 1).mean(), measured[np.argmin(predicted)] / measured.min() - 1,
                          measured[np.argmin(constant)] / measured.min() - 1))
            for i, j in itertools.combinations(range(len(measured)), 2):
                difference = max(measured[i], measured[j]) / min(measured[i], measured[j]) - 1
                b = next(b for b in BINS if b[0] <= difference < b[1])
                pairs[b][0] += (measured[i] < measured[j]) == (predicted[i] < predicted[j])
                pairs[b][1] += (measured[i] < measured[j]) == (constant[i] < constant[j])
                pairs[b][2] += 1
    print(f"\n  {'run':16} {'kernel':14} {'n':>3} {'scatter':>8} {'bias':>7} {'total MAPE':>11} {'regret':>7} "
          f"{'regret, time only':>18}")
    for run, kernel, n, sd, bias, mape, regret, regret_time in lines:
        print(f"  {run:16} {kernel:14} {n:3d} {100 * sd:7.1f}% {100 * bias:+6.1f}% {100 * mape:10.1f}% {100 * regret:6.1f}% "
              f"{100 * regret_time:17.1f}%")
    print(f"  mean over the kernel runs: scatter {100 * np.mean([l[3] for l in lines]):.1f} %, bias "
          f"{100 * np.mean([l[4] for l in lines]):+.1f} %, total MAPE {100 * np.mean([l[5] for l in lines]):.1f} %, "
          f"max regret {100 * max(l[6] for l in lines):.1f} % (time only {100 * max(l[7] for l in lines):.1f} %)")
    print("  pairs of configurations of one kernel ordered correctly, model / time only, by measured difference:")
    for (low, high), (model, time_only, n) in pairs.items():
        if n:
            label = f"{100 * low:.0f}-{100 * high:.0f} %" if high != float("inf") else f">{100 * low:.0f} %"
            print(f"    {label:9} {100 * model / n:5.1f} % / {100 * time_only / n:5.1f} %   ({n} pairs)")
    return lines


def unpriced_events(rows, unused):
    """Warning lines for the events that no probe exercises (so they cost 0) but that validation kernels count."""
    lines = []
    for name in sorted(unused):
        kernels = sorted({row["kernel"] for row in rows if events(row)[name] > 0})
        if kernels:
            lines.append(f"WARNING: no probe exercises '{name}', so it costs 0 in the model, but these kernels count it: "
                         + ", ".join(kernels))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power", required=True, help="prefix of the probe run (its .csv and _energy_table.json); "
                        "several, comma-separated, are averaged")
    parser.add_argument("--counters", required=True, help="prefix of the --profile run (its .csv); several, "
                        "comma-separated, are searched in order")
    parser.add_argument("--validation", help="prefixes of --validate runs (their _validation.csv), comma-separated")
    parser.add_argument("--busy-scale", type=float, default=energy_events.BUSY_SCALE_GBS,
                        help="DRAM rate in GB/s of the busy event's shape 1 - exp(-rate / scale) (default: %(default)g, "
                        "measured on an RTX 3090)")
    arguments = parser.parse_args()
    probes, table = probe_table(arguments.power.split(","), arguments.counters.split(","), arguments.busy_scale)
    if not probes:
        raise SystemExit("no probe has both an energy and counters; check the prefixes")
    energies, fitted, unused = fit(probes)
    print(f"Fit over {len(probes)} probes (pJ per access: measured, model, error):")
    for probe, model in zip(probes, fitted):
        print(f"  {probe.label:34} {probe.energy:9.2f} {model:9.2f} {100 * (model / probe.energy - 1):+6.1f} %")
    errors = np.abs(fitted / np.array([p.energy for p in probes]) - 1)
    print(f"  median error {100 * np.median(errors):.1f} %, {int((errors <= 0.10).sum())} of {len(probes)} within 10 %")
    print("\nEnergy per event (pJ; 'no probe': no probe exercises it, so it counts as 0; DRAM busy: pJ per ns = mW):")
    for name in EVENTS:
        print(f"  {name:32} {energies[name]:9.1f}" + ("  no probe" if name in unused else ""))
    if arguments.validation:
        validations = {run: load_validation(run, table) for run in arguments.validation.split(",")}
        check_same_counters({"the probes": [r for prefix in arguments.counters.split(",") for r in probe_rows(prefix)],
                             **validations})
        for line in unpriced_events([row for rows in validations.values() for row in rows], unused):
            print(line)
        score(validations, energies, arguments.busy_scale)


if __name__ == "__main__":
    main()
