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
import contextlib
import io
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

# The probes of the optional group "tensor" (--only tensor): IMMA, DMMA, sparse HMMA and FP8 matrix products. They are
# checks, not part of the fit: main() reports how the fitted prices (one "inst tensor" price from HMMA, DMMA with the
# fp64 pipe on GA102) predict them, which says whether a kernel using these paths is priced right. Whether one of
# them should enter the fit is a model change under the rules of NEXT_STEPS_PLAN 8.5.
# The lanes probes (FFMA, IMAD in 1 or 8 lanes of every warp, 2026-11-17) price "inst inactive lanes": a warp instruction
# with few active lanes costs about a third of a full one more than its lanes (100 pJ for one FFMA lane against 8.8 pJ per
# lane of a full warp). They enter the fit when their counter run has smsp__thread_inst_executed.sum, which tells
# inactive lanes from predicated-off ones (those cost almost nothing: the indexing probe); otherwise they are checks.
LANES_PROBES = {f"compute {op}-lanes{n}" for op in ("ffma", "imad") for n in (1, 8)}
CHECK_PROBES = {"compute dmma", "compute hmma-sparse", "compute qmma-fp8"}
# The optional opcode probes (--only opcodes): checks of single opcodes against their pipe's price (tools/opcode_energy.py),
# never in the fit.
OPCODE_PREFIX = "compute opcode-"
# The optional broadcast probes (--only broadcast, 2026-11-18): shared loads where 2, 16 or 32 lanes read one address;
# checks, never in the fit (their label has "-bcast").
BROADCAST_TAG = "-bcast"

# The IMMA probe prices "inst tensor imma" (since 2026-11-16): it enters the fit when its counter run has the IMMA count,
# otherwise (an older counter run) it would be fitted as HMMA, so it is a check then.
IMMA_PROBE = "compute imma"

# An event enters the fit only if some probe executes it at least this often per access of one thread (one warp
# instruction per 10 warp accesses): events that only occur a few times per launch (the division in the probes' setup)
# would otherwise get absurd energies that explain other probes.
MIN_EVENTS_PER_ACCESS = 0.1 / 32

# The counters the events are computed from, with the events each one feeds. A counter that a run did not collect counts
# as 0, which silently removes its events, so check_same_counters() compares the runs.
# Optional counters (only some chips or runs have them; without them their events are counted in another one) are left
# out of that check.
COUNTER_EVENTS = {name: entry["feeds"] for name, entry in counter_table.ENTRIES.items()
                  if "feeds" in entry and not {"alias_of", "alternative", "optional"} & set(entry)}


def key_of(row):
    return row["level"], row["operation"], int(row["width_bytes"])


def label_of(key):
    return f"{key[0]} {key[1]}" + (f" w{key[2]}" if key[2] else "")


def probe_rows(prefix):
    """The probe rows of a run at duty 100 (on random data and all SMs: runs of earlier versions had other rows too)."""
    return [r for r in read_rows(prefix + ".csv") if r["duty_pct"] == "100" and r.get("payload", "random") == "random"
            and r.get("active_sms", "all") == "all"]


def probe_table(power_prefixes, counters_prefixes, busy_scale=None, checks=False):
    """The probes (Probe tuples) and the energy table of the power run with the most sleep baselines, its baseline slope the
    mean over the runs with three sleeps or more.
    With several power runs, a probe's energy is the mean over the runs that have it; with several counter runs, its
    counts come from the first that has it. The CHECK_PROBES are left out, or with `checks` returned as the only probes."""
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
    # The baseline is that of the run with the most sleep baselines (a run of a few probes, such as --only compute, has
    # two, at one temperature and with no slope of its own); the slope is the mean over the runs with at least three.
    table = dict(max(tables, key=lambda t: t["baseline"].get("runs", 0)))
    sloped = [t["baseline"]["slope_w_per_c"] for t in tables if t["baseline"].get("runs", 3) >= 3] \
        or [t["baseline"]["slope_w_per_c"] for t in tables]
    table["baseline"] = dict(table["baseline"], slope_w_per_c=float(np.mean(sloped)))

    probes = {}
    for row in (r for prefix in counters_prefixes for r in probe_rows(prefix)):
        key = key_of(row)
        accesses = number(row["accesses_per_launch"])
        if key in energies and key not in probes and accesses > 0 and row.get("smsp__inst_executed.sum"):
            counts = add_busy({name: value / accesses for name, value in events(row).items()},
                              float(np.mean(times[key])) if times[key] else math.nan, busy_scale)
            probes[key] = Probe(label_of(key), energies[key], counts)
    kept = [p for p in probes.values() if p.label != TWO_REGISTER_FFMA or ("compute", "ffma3", 0) not in probes]
    is_check = lambda p: p.label in CHECK_PROBES or (p.label == IMMA_PROBE and not p.counts["inst tensor imma"] > 0) \
        or (p.label in LANES_PROBES and not p.counts["inst inactive lanes"] > 0) \
        or p.label.startswith(OPCODE_PREFIX) or BROADCAST_TAG in p.label  # noqa: E731
    return [p for p in kept if is_check(p) == checks], table


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


# The DRAM busy event's scale (energy_events.BUSY_SCALE_GBS, 95 GB/s) was fitted on an RTX 3090 (fit_busy.py of the
# earlier program, dependent-load runs Instr and LowRate, 2026-11-05). The fit prices the event on every GPU, but its
# shape is a setting: busy_scale_scan() measures which scales this GPU's probes allow.
BUSY_SCALE_SOURCE = "RTX 3090 (dependent-load runs Instr and LowRate, 2026-11-05)"
BUSY_SCAN_GBS = tuple(float(f"{10 * 2 ** (k / 3):.3g}") for k in range(25))  # 10 to 2560 GB/s, three steps per doubling


def busy_scale_scan(power_prefixes, counters_prefixes):
    """How well the probes fit with each scale of BUSY_SCAN_GBS, and without the busy event. Returns a dict: the scan
    [(scale, sum of squared relative errors)], `none` (that sum without the event), `best` (the scale with the smallest),
    `interval` (the scales whose sum is within a profile-likelihood 95 % bound of the smallest: + 3.84 x the residual
    variance; None at an end of the scan means unbounded there), and `needed` (whether leaving the event out is outside
    that bound). The event is in the NNLS fit, so only its shape is scanned; this uses the probes only, no real kernel."""
    def sse(scale):
        with contextlib.redirect_stdout(io.StringIO()):
            probes, _ = probe_table(power_prefixes, counters_prefixes, scale)
        _, fitted, unused = fit(probes)
        energies = np.array([p.energy for p in probes])
        return float(np.sum((fitted / energies - 1) ** 2)), len(probes), len(EVENTS) - len(unused)
    scan = [(scale, sse(scale)[0]) for scale in BUSY_SCAN_GBS]
    none, n, events_used = sse(1e12)
    best, smallest = min(scan, key=lambda s: s[1])
    bound = smallest + 3.84 * smallest / max(n - events_used, 1)
    inside = [scale for scale, value in scan if value <= bound]
    interval = (None if inside[0] == scan[0][0] else inside[0], None if inside[-1] == scan[-1][0] else inside[-1])
    return {"scan": scan, "none": none, "best": best, "interval": interval, "bound": bound, "needed": none > bound}


def busy_scale_choice(power_prefixes, counters_prefixes, inherited=None):
    """The scale a model built from these runs uses, and where it comes from: the inherited value (BUSY_SCALE_GBS) if it
    lies in this GPU's interval, else the scan's best. Returns (scale, source, scan result)."""
    inherited = inherited or energy_events.BUSY_SCALE_GBS
    result = busy_scale_scan(power_prefixes, counters_prefixes)
    low, high = result["interval"]
    inside = (low is None or inherited >= low) and (high is None or inherited <= high)
    if inside:
        return inherited, f"the value fitted on the {BUSY_SCALE_SOURCE}, inside this GPU's interval", result
    return result["best"], "fitted on these probes (the inherited value is outside this GPU's interval)", result


def describe_busy_scan(result):
    low, high = result["interval"]
    text = (f"DRAM busy scale: best fit of the probes at {result['best']:g} GB/s; 95 % interval "
            f"{'<' + format(BUSY_SCAN_GBS[0], 'g') if low is None else format(low, 'g')} to "
            f"{'>' + format(BUSY_SCAN_GBS[-1], 'g') if high is None else format(high, 'g')} GB/s")
    text += ("; without the event the fit is worse (sum of squared errors " if result["needed"] else
             "; the probes do not need the event (sum of squared errors ")
    return text + f"{result['none']:.3f} against {min(v for _, v in result['scan']):.3f})"


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
        # Between launches the GPU draws gap_below_baseline_w less than the baseline (Validation.cpp; runs before
        # 2026-11-17 have no such column, and their kernel_s is KTT's time, which includes that gap).
        gap = number(row.get("gap_below_baseline_w"))
        gap = gap if not math.isnan(gap) else 0.0
        row["dynamic_energy_j"] = (number(row["power_w"]) - base) * number(row["wall_s"]) \
            + gap * (number(row["wall_s"]) - number(row["kernel_s"]))
        row["energy_j"] = base * number(row["kernel_s"]) + row["dynamic_energy_j"]
    return rows


# Configurations whose SM clock differs from the model's by more than this (throttled, or at the power cap) are not what a
# fixed-clock model describes: build_model.py leaves them out of the uncertainty, and score() out of every score, so that
# scatter, bias, MAPE, regret and pairs describe one operating point (as ClockTolerance in Common.h).
CLOCK_TOLERANCE = 0.02


def at_model_clock(rows, clock_mhz):
    """The rows within CLOCK_TOLERANCE of `clock_mhz` (rows without a clock are kept), and the others."""
    if math.isnan(number(clock_mhz)):
        return list(rows), []
    kept, off = [], []
    for row in rows:
        clock = number(row.get("sm_clock_mhz"))
        (kept if math.isnan(clock) or abs(clock / number(clock_mhz) - 1) <= CLOCK_TOLERANCE else off).append(row)
    return kept, off


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


def score(validations, energies, busy_scale=None, clock_mhz=None):
    """Prints how the model ranks the configurations of each kernel of each run ({run: rows}); returns the lines'
    numbers: (run, kernel, n, scatter, bias, total MAPE, regret, regret by time alone). With `clock_mhz` (the model's
    clock), every number, the time-only regret and the pairs included, uses only the rows at that clock (at_model_clock);
    without it, all rows (off-clock and power-capped launches mixed in)."""
    lines = []
    pairs = {b: [0, 0, 0] for b in BINS}  # model right, time-only right, pairs
    if clock_mhz is not None:
        filtered, left_out = {}, []
        for run, rows in validations.items():
            filtered[run], off = at_model_clock(rows, clock_mhz)
            left_out += [(os.path.basename(run), r["kernel"]) for r in off]
        total = sum(len(rows) for rows in validations.values())
        print(f"\nScope: fixed clock, {number(clock_mhz):.0f} MHz +- {100 * CLOCK_TOLERANCE:.0f} %: {total - len(left_out)} of "
              f"{total} configurations" + (f"; left out off the clock: " + ", ".join(
                  f"{run} {kernel} {n}" for (run, kernel), n in collections.Counter(left_out).items()) if left_out else ""))
        validations = {run: rows for run, rows in filtered.items() if rows}
    else:
        print("\nScope: all clocks (off-clock and power-capped launches included; not the fixed-clock model's scope)")
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


def identifiability(probes, top=6):
    """How well the probes separate the events: the condition number of the fit's scaled matrix (columns the fit uses) and
    the pairs of events whose per-probe counts are most correlated (their prices trade off: compare their sum)."""
    a = np.array([[p.counts[name] for name in EVENTS] for p in probes]) / np.array([p.energy for p in probes])[:, None]
    used = [i for i, name in enumerate(EVENTS) if a[:, i].max() * np.array([p.energy for p in probes]).max() > 0
            and np.array([p.counts[name] for p in probes]).max() >= MIN_EVENTS_PER_ACCESS]
    scaled = a[:, used] / a[:, used].max(axis=0)
    singular = np.linalg.svd(scaled, compute_uv=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(scaled.T)
    pairs = sorted(((abs(correlation[i, j]), EVENTS[used[i]], EVENTS[used[j]]) for i in range(len(used))
                    for j in range(i + 1, len(used)) if np.isfinite(correlation[i, j])), reverse=True)[:top]
    return float(singular[0] / singular[-1]), pairs


def bootstrap(probes, validations, samples, seed=1, busy_scale=None):
    """The fit repeated on `samples` resamplings of the probes (with replacement), each keeping at least one probe of every
    event the full fit prices (several events have one probe each; without it they would cost 0, which says nothing about
    the probe set's noise). Returns, per event, the 5th, 50th and
    95th percentile of its price, and over the validation configurations ({run: rows}, already filtered): the median
    relative sd of the predicted dynamic energy, the share of same-kernel pairs whose predicted order flips in more than
    10 % of the resamplings, and the 95th percentile of the largest regret. The prices are proxies that depend on the
    probe set; this says how much the predictions and the ranking depend on it."""
    rng = np.random.default_rng(seed)
    configurations = [(run, r) for run, rows in validations.items() for r in rows]
    counts = np.array([[add_busy(events(r), number(r["kernel_s"]) * 1e9, busy_scale)[name] for name in EVENTS]
                       for _, r in configurations]) * 1e-12
    baseline = np.array([number(r["baseline_w"]) * number(r["kernel_s"]) for _, r in configurations])
    measured = np.array([number(r["energy_j"]) for _, r in configurations])
    groups = collections.defaultdict(list)
    for index, (run, r) in enumerate(configurations):
        groups[(run, r["kernel"])].append(index)
    prices, predictions, regrets = [], [], []
    _, _, unused_full = fit(probes)
    while len(prices) < samples:
        chosen = [probes[i] for i in rng.integers(0, len(probes), len(probes))]
        energies, _, unused = fit(chosen)
        if unused != unused_full:
            continue
        coefficients = np.array([energies[name] for name in EVENTS])
        prices.append(coefficients)
        dynamic = counts @ coefficients
        predictions.append(dynamic)
        total = baseline + dynamic
        regrets.append(max(measured[g][np.argmin(total[g])] / measured[g].min() - 1 for g in groups.values()))
    prices, predictions = np.array(prices), np.array(predictions)
    totals = predictions + baseline
    flips = pairs = 0
    for indices in groups.values():
        for i, j in itertools.combinations(indices, 2):
            order = totals[:, i] < totals[:, j]
            flips += min(order.mean(), 1 - order.mean()) > 0.10
            pairs += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.nanmedian(predictions.std(axis=0) / predictions.mean(axis=0))
    percentiles = {name: tuple(np.percentile(prices[:, k], [5, 50, 95])) for k, name in enumerate(EVENTS)}
    return percentiles, float(spread), flips / pairs if pairs else math.nan, float(np.percentile(regrets, 95))


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
    parser.add_argument("--bootstrap", type=int, default=0, metavar="N", help="also refit on N resamplings of the "
                        "probes and report how much the prices, the predictions and the ranking move (e.g. 200)")
    parser.add_argument("--leave-out", default="", help="kernels to leave out of the scores, comma-separated (such as "
                        "mixed, the control kernel of the earlier validation runs, which the program no longer has)")
    parser.add_argument("--all-clocks", action="store_true", help="score every configuration, also those more than "
                        f"{100 * CLOCK_TOLERANCE:.0f} %% off the probe run's clock (default: only those at it, as build_model.py)")
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
    checks, _ = probe_table(arguments.power.split(","), arguments.counters.split(","), arguments.busy_scale, checks=True)
    if checks:
        print("Checks, not in the fit (the optional tensor, lanes, opcode and broadcast probes; pJ per access: measured, "
              "model, error):")
        for probe in checks:
            model = sum(probe.counts[name] * energies[name] for name in EVENTS)
            print(f"  {probe.label:34} {probe.energy:9.2f} {model:9.2f} {100 * (model / probe.energy - 1):+6.1f} %")
    print("\nEnergy per event (pJ; 'no probe': no probe exercises it, so it counts as 0; DRAM busy: pJ per ns = mW):")
    for name in EVENTS:
        print(f"  {name:32} {energies[name]:9.1f}" + ("  no probe" if name in unused else ""))
    if arguments.validation:
        left_out = set(filter(None, arguments.leave_out.split(",")))
        validations = {run: [r for r in load_validation(run, table) if r["kernel"] not in left_out]
                       for run in arguments.validation.split(",")}
        check_same_counters({"the probes": [r for prefix in arguments.counters.split(",") for r in probe_rows(prefix)],
                             **validations})
        for line in unpriced_events([row for rows in validations.values() for row in rows], unused):
            print(line)
        score(validations, energies, arguments.busy_scale,
              None if arguments.all_clocks else table.get("sm_clock_mhz"))
    condition, pairs = identifiability(probes)
    print(f"\nIdentifiability: condition number of the fit's scaled matrix {condition:.0f}; most correlated events over the "
          "probes (their prices trade off, compare sums): " + "; ".join(f"{a} / {b} {r:.2f}" for r, a, b in pairs))
    if arguments.bootstrap:
        scored = {}
        if arguments.validation:
            for run, rows in validations.items():
                scored[run] = rows if arguments.all_clocks else at_model_clock(rows, table.get("sm_clock_mhz"))[0]
        percentiles, spread, flips, regret = bootstrap(probes, scored, arguments.bootstrap, busy_scale=arguments.busy_scale)
        print(f"Bootstrap over the probes ({arguments.bootstrap} resamplings): prices as statistical proxies, 5th / 50th / "
              "95th percentile (pJ):")
        for name in EVENTS:
            low, middle, high = percentiles[name]
            if high > 0:
                print(f"  {name:32} {low:9.1f} {middle:9.1f} {high:9.1f}")
        if scored:
            print(f"  predicted dynamic energy of the validation configurations: median relative sd {100 * spread:.1f} %; "
                  f"pairs of one kernel whose order flips in more than 10 % of the resamplings: {100 * flips:.1f} %; "
                  f"largest regret, 95th percentile: {100 * regret:.1f} %")


if __name__ == "__main__":
    main()
