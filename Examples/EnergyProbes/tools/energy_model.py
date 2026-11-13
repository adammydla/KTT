#!/usr/bin/env python3
"""Energy model from the probes, checked on KTT example kernels.

1. Fit: every probe gives its energy per access (the fit across duties, <power>_energy_table.json) and its CUPTI counts per
   access (a run with --profile, <counters>.csv). Non-negative least squares on relative errors gives the energy of one
   event of each kind: a warp instruction of each pipe (the lsu pipe's split into shared-memory, global-path and other
   instructions; with --fma-halves the fma pipe's into its heavy and lite halves), a sector served by L1 (loads, texture fetches), L2 or DRAM, an atomic sector in L2, a shared-memory
   wavefront and a wavefront replayed for a bank conflict, a constant-cache request and miss (as Wattchmen and AccelWattch
   do from their microbenchmarks), and a nanosecond of DRAM kept awake by steady traffic (add_busy(); --no-busy: off).
   With --weighted, every family of probes (a study, see family()) weighs about the same
   in the fit, so that the 30 bank-conflict probes do not outweigh the 3 plain loads of a level. That helps counter runs
   without the lsu instructions by memory space, where the shared probes otherwise price all lsu instructions too low;
   with the split it changes nothing.
2. Predict: for every configuration of a --validate run (<validation>_validation.csv), the dynamic energy per launch is
   the sum of its counts times those energies; the total adds the sleep baseline over the kernel's time. Both are compared
   with the measurement, per kernel: mean absolute percentage error, correlation, and how much more energy the
   configuration the model ranks best uses than the measured best (regret).

    python3 tools/energy_model.py --power Model[,Study,...] --counters Profile9[,ModelProfile] \
        --validation Validation3 [--csv out.csv] [--cross-validate] [--mixed Mixed]
    # the RTX 3090 model of 2026-11-05:
    python3 tools/energy_model.py --power Full,Ffma3,Writes,Instr --counters ProfileFull4 --validation Validation11

With --mixed, the mixed-intensity probes of a run (loads with 0 to 1024 FFMAs each) are predicted the same way. A
validation run with the control kernel (--validate all,mixed: the same probe, measured like the example kernels) is
predicted apart from the example kernels, and with --mixed its measured energies are compared with the probe run's: a
difference there is the measurement's, not the model's.

Two reference models show what the counters add: a constant power over the kernel's time (the mean of all
configurations), and the probe fit with only instructions and DRAM sectors.
"""

import argparse
import collections
import csv
import json
import math
import os
import re
import sys

import numpy as np
from scipy.optimize import nnls
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_table  # noqa: E402


# The events and how they are counted are in energy_events.py (shared with energy_predictor.py). Its settings FMA_HALVES,
# BUSY and BUSY_SCALE_GBS are set through configure().
import energy_events  # noqa: E402
from energy_events import (BUSY_EVENT, EVENTS, GLOBAL_INSTRUCTIONS, SHARED_INSTRUCTIONS, add_busy, counter, events,  # noqa: E402,F401
                           number, value_of)

ALIASES = counter_table.ALIASES
read_rows = counter_table.read_rows  # the rows of a run's CSV (repairs rows with a comma in the note)
BUSY_SCALE_GBS = energy_events.BUSY_SCALE_GBS  # the default


def configure(fma_halves=False, busy=True, busy_scale=energy_events.BUSY_SCALE_GBS):
    """The model's options (--fma-halves, --no-busy, --busy-scale), for this module and the tools that import it."""
    energy_events.FMA_HALVES = fma_halves
    energy_events.BUSY = busy
    energy_events.BUSY_SCALE_GBS = busy_scale


SIMPLE = ["inst alu", "inst fma (fp32, imad)", "inst fma lite (fp32)", "inst fp16", "inst fp64", "inst tensor",
          "inst xu (mufu)", "inst lsu shared", "inst lsu global", "inst lsu other", "inst other",
          "DRAM read sector", "DRAM write sector"]


# The FFMA probes that compute x * x + c: they read one register twice and take c from the constant bank, and cost 15-20 %
# less per FFMA than the FMAs of real kernels (acc = a * b + acc), which read three different registers. When a run has the
# probe with three register operands (ffma3), they stay out of the fit.
TWO_REGISTER_FFMA = ("compute ffma", "compute ffma-static")

# Probes that the events cannot describe and that therefore stay out of the fit; they are predicted as a check. FMUL and
# FADD count as fma-pipe instructions like an FFMA (the opcode counters need SASS patching), and MUFU.SIN comes with
# __sinf's FMUL.RZ.
CHECK_ONLY = ("compute fmul-fadd", "compute mufu-sin")


# The optional counter groups of the model, each with one of its counters: runs profiled before a group was collected do not
# have it, and then its events are counted together with the rest of their pipe.
COUNTER_GROUPS = counter_table.GROUPS


def counter_groups(row):
    """The optional counter groups that a row has."""
    return frozenset(name for name, column in COUNTER_GROUPS.items() if row.get(column, "") not in ("", None))


def family(level, operation):
    """The study a probe belongs to. A family of n probes weighs sqrt(n) in the fit (each probe 1 / sqrt(n)): more probes of
    a kind still count more, but not n times more."""
    if operation.startswith(("red", "atom")):
        return "atomics"
    if operation in ("copy", "update", "local"):
        return "writes"
    if "way" in operation:
        return "bank conflicts"
    if "-run" in operation or "-stream" in operation:
        return "locality " + level
    if "-sub-" in operation:
        return "sub-sector " + level
    if operation.startswith("chase"):
        return "latency"
    if level == "compute":
        if operation in ("ffma-static", "ffma-indexed"):
            return "indexing"
        return "instructions" if operation in ("shfl", "bar", "ffma-loop", "fmul-fadd", "mufu-sin") else "arithmetic"
    if level.startswith("const"):
        return "constant"
    if operation == "tex":
        return "texture"
    if level in ("shared", "none"):
        return level
    return "loads and stores " + level


# A probe of the fit: its label, energy per access (pJ), events per access, family, and the optional counter groups its
# counters have.
Probe = collections.namedtuple("Probe", "label energy counts family groups")


def probe_table(power_prefixes, counters_prefixes):
    """The probes with random data on all SMs (Probe tuples). With several power runs, a probe's energy is the mean over
    the runs that have it; the baseline slope is the mean of their slopes. With several counter runs, a probe's counts
    come from the first run that has it."""
    tables = []
    collected = {}
    times = {}  # nanoseconds per access at duty 100, per probe
    for prefix in power_prefixes:
        with open(prefix + "_energy_table.json") as file:
            table = json.load(file)
        tables.append(table)
        full_rate = {(r["level"], r["operation"], int(r["width_bytes"])): r for r in read_rows(prefix + ".csv")
                     if r["payload"] == "random" and r["active_sms"] == "all" and r["duty_pct"] == "100"}
        for key, row in full_rate.items():
            if number(row["accesses_per_launch"]) > 0 and number(row["active_s"]) > 0:
                times.setdefault(key, []).append(number(row["active_s"]) / number(row["accesses_per_launch"]) * 1e9)
        run = {}
        for fit in table["fits"]:
            if fit["payload"] == "random" and fit["pj_per_access"] is not None:
                key = (fit["level"], fit["operation"], int(fit["width_bytes"]))
                energy = float(fit["pj_per_access"])
                # As in EnergyProbes' summary: a fit whose fixed power exceeds half the full-rate dynamic power is not
                # a line through the rows (the constant L1.5), so the full-rate row's energy is used.
                row = full_rate.get(key)
                if row is not None and number(row["pj_per_access"]) > 0 and fit["fixed_w"] is not None \
                        and abs(fit["fixed_w"]) > 0.5 * (number(row["power_w"]) - number(row["baseline_w"])):
                    energy = number(row["pj_per_access"])
                    print(f"{prefix}: {key[0]} {key[1]}: fit not linear (fixed {fit['fixed_w']:.1f} W), using duty 100: "
                          f"{energy:.2f} instead of {float(fit['pj_per_access']):.2f} pJ per access")
                run[key] = energy
        # The latency probes run at duty 100 only (their access rate varies with the parallelism instead): they enter with
        # their full-rate energy, which keeps what waiting costs.
        for key, row in full_rate.items():
            if key[1].startswith("chase") and key not in run and number(row["pj_per_access"]) > 0:
                run[key] = number(row["pj_per_access"])
        # The ffma-loop probe computes x * x + c, as the two-register FFMA probe; when the run also has ffma3, its FFMA is
        # re-expressed as one with three registers (+ ffma3 - ffma of the same run), so that its branches are priced on
        # the FFMA the fit uses.
        loop, two, three = (("compute", name, 0) for name in ("ffma-loop", "ffma", "ffma3"))
        if loop in run and two in run and three in run:
            run[loop] += run[three] - run[two]
        for key, energy in run.items():
            collected.setdefault(key, []).append(energy)
    energies = {key: float(np.mean(values)) for key, values in collected.items()}
    # The fit minimises relative errors (it divides by the energy): a probe at or below 0 (a weak probe whose fit across
    # duties came out negative from noise) would enter with a huge or negative weight, so it stays out.
    for key in sorted(key for key, energy in energies.items() if not energy > 0):
        print(f"{key[0]} {key[1]}" + (f" w{key[2]}" if key[2] else "") + f": energy {energies[key]:.2f} pJ per access is "
              "not above 0, left out of the fit")
        del energies[key]
    table = dict(tables[-1])
    table["baseline"] = dict(table["baseline"])
    table["baseline"]["slope_w_per_c"] = float(np.mean([t["baseline"]["slope_w_per_c"] for t in tables]))

    rows = []
    for prefix in counters_prefixes:
        rows += read_rows(prefix + ".csv")
    probes = []
    seen = set()
    for row in rows:
        if row["active_sms"] != "all" or row["payload"] != "random" or row["duty_pct"] != "100":
            continue
        key = (row["level"], row["operation"], int(row["width_bytes"]))
        accesses = number(row["accesses_per_launch"])
        if key not in energies or key in seen or not accesses > 0 or "smsp__inst_executed.sum" not in row:
            continue
        seen.add(key)
        per_access = add_busy({name: value / accesses for name, value in events(row).items()},
                              float(np.mean(times[key])) if key in times else math.nan)
        label = f"{key[0]} {key[1]}" + (f" w{key[2]}" if key[2] else "")
        probes.append(Probe(label, energies[key], per_access, family(key[0], key[1]), counter_groups(row)))
    return probes, table


# An event enters the fit only if some probe executes it at least this often per access of one thread (one warp
# instruction per 10 warp accesses): events that only occur a few times per launch (e.g. the division in the probes'
# setup, 1e-9 per access) would otherwise get absurd energies that explain other probes.
MIN_EVENTS_PER_ACCESS = 0.1 / 32


def fit(probes, names, weighted=False):
    """Energies per event (pJ) by non-negative least squares on relative residuals; events no probe exercises get 0. With
    `weighted`, every probe's residual is weighed 1 / sqrt(probes in its family)."""
    a = np.array([[p.counts[name] for name in names] for p in probes])
    b = np.array([p.energy for p in probes])
    sizes = collections.Counter(p.family for p in probes)
    weights = np.array([1 / math.sqrt(sizes[p.family]) if weighted else 1.0 for p in probes])
    used = a.max(axis=0) >= MIN_EVENTS_PER_ACCESS
    scaled = a[:, used] / b[:, None] * weights[:, None]
    norms = scaled.max(axis=0)
    solution, _ = nnls(scaled / norms, weights)
    coefficients = np.zeros(len(names))
    coefficients[used] = solution / norms
    return dict(zip(names, coefficients)), a @ coefficients, {name for name, flag in zip(names, used) if not flag}


def summary(measured, predicted):
    measured = np.asarray(measured)
    predicted = np.asarray(predicted)
    error = (predicted - measured) / measured
    result = {"n": len(measured), "mape": 100 * np.mean(np.abs(error)), "bias": 100 * np.mean(error)}
    if len(measured) > 2:
        result["pearson"] = pearsonr(measured, predicted)[0]
        result["spearman"] = spearmanr(measured, predicted)[0]
        result["regret"] = 100 * (measured[np.argmin(predicted)] / measured.min() - 1)
    return result


def describe(name, s):
    text = f"  {name:30} n={s['n']:3d}  MAPE {s['mape']:6.1f} %  bias {s['bias']:+6.1f} %"
    if "pearson" in s:
        text += f"  r {s['pearson']:5.3f}  rho {s['spearman']:5.3f}  regret {s['regret']:5.1f} %"
    return text


def print_fit(probes, fitted, energies, unused):
    print(f"Fit over {len(probes)} probes (pJ per access: measured, model, error):")
    for probe, model in zip(probes, fitted):
        print(f"  {probe.label:22} {probe.energy:9.2f} {model:9.2f} {100 * (model / probe.energy - 1):+6.1f} %")
    print("\nEnergy per event (pJ; 0 = not needed by the fit, 'no probe' = no probe exercises it, so it counts as 0):")
    for name in EVENTS:
        if ((name == "inst fma lite (fp32)" and not energy_events.FMA_HALVES)
                or (name == BUSY_EVENT and not energy_events.BUSY)):
            continue
        print(f"  {name:24} {energies[name]:9.1f}" + ("  no probe" if name in unused else "")
              + ("  (pJ per ns = mW while awake)" if name == BUSY_EVENT else ""))


# Combinations of events that the probes determine well, for comparing GPUs: events that always occur together trade off
# in the fit (on an RTX 2080 Ti "L2 read miss request" came out 14265 and "DRAM read sector" 394 pJ, on the RTX 3090 6724 and
# 2419 pJ), but their sum for a line read was 15.8 against 16.4 nJ.
COMBINED = [
    ("128-byte line read from DRAM (4 DRAM read sectors + L2 read miss request)", "nJ",
     {"DRAM read sector": 4, "L2 read miss request": 1}),
    ("128-byte line written back to DRAM (4 DRAM write sectors)", "nJ", {"DRAM write sector": 4}),
    ("conflict-free shared load (lsu shared instruction + shared ld wavefront)", "pJ",
     {"inst lsu shared": 1, "shared ld wavefront": 1}),
    ("FFMA (warp instruction of the fma pipe)", "pJ", {"inst fma (fp32, imad)": 1}),
]


def combined_costs(energies):
    """(label, value, unit) of COMBINED from the energies per event."""
    return [(label, sum(n * energies[e] for e, n in parts.items()) / (1000 if unit == "nJ" else 1), unit)
            for label, unit, parts in COMBINED]


def print_combined(energies):
    print("\nCombined costs (well determined by the probes; compare GPUs on these, not on single events):")
    for label, value, unit in combined_costs(energies):
        print(f"  {label:76} {value:8.2f} {unit}")


def unpriced_events(validation, unused):
    """Lines for the events that no probe exercises (so the fit prices them at 0) but that validation kernels count: the
    model predicts those as free. Instruction events also give their share of the kernel's warp instructions."""
    lines = []
    for name in sorted(unused):
        kernels = collections.defaultdict(list)
        for row in validation:
            count = events(row)[name]
            if count > 0:
                total = counter(row, "smsp__inst_executed.sum")
                kernels[row["kernel"]].append((count, count / total if name.startswith("inst") and total > 0 else None))
        if kernels:
            parts = []
            for kernel, values in sorted(kernels.items()):
                share = [s for _, s in values if s is not None]
                parts.append(f"{kernel} {np.mean([c for c, _ in values]):.3g} per launch"
                             + ((f" ({100 * np.mean(share):.1f} % of its warp instructions)" if np.mean(share) >= 0.001
                                 else " (under 0.1 % of its warp instructions)") if share else ""))
            lines.append(f"  {name}: " + ", ".join(parts))
    if lines:
        lines.insert(0, "WARNING: no probe exercises these events, so they cost 0 in the model, but validation kernels "
                        "count them (they are predicted as free):")
    return lines


def cross_validate(probes, weighted):
    """Leave-one-out: every probe predicted from the event energies fitted over all the other probes. This tests whether
    the energies per event are a consistent decomposition: a probe's energy should follow from what the others say its
    events cost. A probe with an event that no other probe exercises cannot be predicted this way; those are listed
    apart."""
    errors, alone = [], []
    for i, (label, energy, counts, _, _) in enumerate(probes):
        energies, _, unused = fit(probes[:i] + probes[i + 1:], EVENTS, weighted)
        own = [name for name in EVENTS if counts[name] >= MIN_EVENTS_PER_ACCESS and name in unused]
        if own:
            alone.append(f"{label} ({', '.join(own)})")
            continue
        model = sum(counts[name] * energies[name] for name in EVENTS)
        errors.append((label, energy, model, model / energy - 1))

    relative = np.abs([e[3] for e in errors])
    print(f"\nLeave-one-out over {len(probes)} probes (each predicted from a fit without it): {len(errors)} predicted, "
          f"MAPE {100 * relative.mean():.1f} %, median {100 * np.median(relative):.1f} %, within 10 %: "
          f"{int((relative <= 0.10).sum())}, within 20 %: {int((relative <= 0.20).sum())}")
    print("  largest errors (measured, predicted pJ per access):")
    for label, energy, model, error in sorted(errors, key=lambda e: -abs(e[3]))[:10]:
        print(f"    {label:28} {energy:9.2f} {model:9.2f} {100 * error:+6.1f} %")
    if alone:
        print("  not predictable (an event of their own): " + "; ".join(alone))


def load_validation(prefix, table):
    """The configurations of a --validate run that have power and counters, with their measured energies per launch
    recomputed from the run's baseline moved to each row's temperature. When that run's sleeps spanned too few degrees
    for a slope of their own, the slope of the probe run's baseline (`table`) is used."""
    rows = [row for row in read_rows(prefix + "_validation.csv") if row["valid"] == "1" and "smsp__inst_executed.sum" in row
            and not math.isnan(number(row["dynamic_energy_j"])) and counter(row, "smsp__inst_executed.sum") > 0]
    if not rows:
        raise SystemExit("the validation run has no rows with power and counters")

    slope = number(rows[0].get("baseline_slope_w_per_c"))
    if not slope > 0:
        slope = float(table["baseline"]["slope_w_per_c"])
        print(f"\nThe validation run's baseline has no temperature model; using the probe run's {slope:.2f} W/C.")
    for row in rows:
        if "baseline_mean_w" in row:
            base = number(row["baseline_mean_w"]) + slope * (number(row["temperature_c"])
                                                             - number(row["baseline_temperature_c"]))
            row["baseline_w"] = base
            row["dynamic_energy_j"] = (number(row["power_w"]) - base) * number(row["wall_s"])
            row["energy_j"] = base * number(row["kernel_s"]) + row["dynamic_energy_j"]
    return rows


def predict(rows, energies, simple):
    """Measured and predicted energies per launch of every configuration: the probe model, the instructions-and-DRAM
    model, and the constant-power reference (the mean power of all configurations over each one's kernel time)."""
    mean_power = np.mean([number(r["energy_j"]) / number(r["kernel_s"]) for r in rows])
    results = []
    for row in rows:
        counts = add_busy(events(row), number(row["kernel_s"]) * 1e9)
        dynamic = sum(counts[name] * energies[name] for name in EVENTS) * 1e-12
        dynamic_simple = sum(counts[name] * simple[name] for name in SIMPLE) * 1e-12
        baseline = number(row["baseline_w"]) * number(row["kernel_s"])
        results.append({
            "kernel": row["kernel"], "configuration": row["configuration"], "busy": number(row["busy"]),
            "kernel_s": number(row["kernel_s"]), "measured_dynamic_j": number(row["dynamic_energy_j"]),
            "predicted_dynamic_j": dynamic, "measured_j": number(row["energy_j"]), "predicted_j": baseline + dynamic,
            "simple_j": baseline + dynamic_simple, "constant_power_j": mean_power * number(row["kernel_s"]),
            "shares": {name: counts[name] * energies[name] * 1e-12 / dynamic if dynamic > 0 else 0.0 for name in EVENTS},
        })
    return results


# The control kernel of a validation run (--validate mixed): the mixed-intensity probe, measured like the example kernels.
CONTROL = "mixed"


def report(results):
    print(f"\nValidation ({len(results)} configurations with power and counters):")
    for kernel in sorted(set(r["kernel"] for r in results), key=[r["kernel"] for r in results].index):
        group = [r for r in results if r["kernel"] == kernel]
        busy = np.mean([r["busy"] for r in group])
        print(f"\n{kernel}{' (control kernel, not in the totals)' if kernel == CONTROL else ''}: {len(group)} configurations, "
              f"{1e3 * min(r['kernel_s'] for r in group):.2f}-"
              f"{1e3 * max(r['kernel_s'] for r in group):.2f} ms per launch, GPU busy {100 * busy:.0f} % of the host time")
        measured = [r["measured_j"] for r in group]
        print(describe("total energy, probe model", summary(measured, [r["predicted_j"] for r in group])))
        print(describe("dynamic energy, probe model", summary([r["measured_dynamic_j"] for r in group],
                                                                [r["predicted_dynamic_j"] for r in group])))
        print(describe("total, instructions + DRAM", summary(measured, [r["simple_j"] for r in group])))
        print(describe("total, constant power", summary(measured, [r["constant_power_j"] for r in group])))
        shares = {name: np.mean([r["shares"][name] for r in group]) for name in EVENTS}
        top = sorted(shares.items(), key=lambda item: -item[1])[:5]
        print("  predicted dynamic energy by event: " + ", ".join(f"{name} {100 * share:.0f} %" for name, share in top))

    examples = [r for r in results if r["kernel"] != CONTROL]
    if not examples:
        return
    print("\nAll example kernels:")
    print(describe("total energy, probe model", summary([r["measured_j"] for r in examples],
                                                        [r["predicted_j"] for r in examples])))
    print(describe("dynamic energy, probe model", summary([r["measured_dynamic_j"] for r in examples],
                                                          [r["predicted_dynamic_j"] for r in examples])))
    print("  (regret and correlations over all kernels mix different problems; the per-kernel lines are the ones that "
          "matter for tuning)")


MIXED = re.compile(r"ld-ffma(\d+)$")


def load_mixed(power_prefixes, counters_prefixes):
    """The mixed-intensity probes (16-byte loads with K FFMAs each, --mixed): measured energy per load at duty 100 from the
    first power run that has them, events per load from the first counters run that has them. They run at one duty only,
    so they have no fit across duties and are not in the probe fit: like the validation kernels, they check the model, here
    with loads and FFMAs in known proportions from memory-bound to compute-bound."""
    def full_rate(prefixes):
        found = {}
        for prefix in prefixes:
            for row in read_rows(prefix + ".csv"):
                match = MIXED.match(row["operation"])
                if match and row["payload"] == "random" and row["active_sms"] == "all" and row["duty_pct"] == "100":
                    found.setdefault((row["level"], int(match.group(1))), row)
        return found

    measured = full_rate(power_prefixes)
    counted = full_rate(counters_prefixes)
    probes = []
    for key in sorted(measured, key=lambda k: (["L1", "L2", "DRAM"].index(k[0]) if k[0] in ("L1", "L2", "DRAM") else 3,
                                               k[1])):
        energy = number(measured[key]["pj_per_access"])
        row = counted.get(key)
        accesses = number(row["accesses_per_launch"]) if row else math.nan
        if not energy > 0 or not accesses > 0 or "smsp__inst_executed.sum" not in row:
            continue
        power = measured[key]
        ns = number(power["active_s"]) / number(power["accesses_per_launch"]) * 1e9
        probes.append((key[0], key[1], energy, add_busy({name: value / accesses for name, value in events(row).items()}, ns),
                       counter_groups(row)))
    return probes


def report_mixed(probes, energies):
    print("\nMixed intensity (16-byte loads with K FFMAs each, not in the fit; pJ per load: measured, predicted, error, and "
          "the predicted share of the fma pipe):")
    print(f"  {'level':6} {'FFMA/ld':>8} {'measured':>10} {'predicted':>10} {'error':>8} {'FFMA share':>11}")
    errors = {}
    for level, fmas, energy, counts, _ in probes:
        predicted = sum(counts[name] * energies[name] for name in EVENTS)
        fma = sum(counts[name] * energies[name] for name in ("inst fma (fp32, imad)", "inst fma lite (fp32)"))
        share = fma / predicted if predicted > 0 else 0.0
        error = predicted / energy - 1
        errors.setdefault(level, []).append(error)
        print(f"  {level:6} {fmas:8d} {energy:10.1f} {predicted:10.1f} {100 * error:+7.1f} % {100 * share:9.0f} %")
    print("  per level: " + "; ".join(f"{level} MAPE {100 * np.mean(np.abs(e)):.1f} %, bias {100 * np.mean(e):+.1f} %"
                                      for level, e in errors.items()))


def report_control(rows, power_prefixes):
    """The control kernel's configurations measured twice: in the validation run (back-to-back launches of 5 or 50 ms
    through KTT, host time, that run's baseline at the row's temperature) and in a probe run with --mixed (200 ms launches
    in KTT's precise measurement, GPU time, that run's baseline). Same kernel, same configuration, so the ratio of the two
    energies per load is what the validation measurement adds or loses; the model does not enter."""
    probe = {}
    for prefix in power_prefixes:
        for row in read_rows(prefix + ".csv"):
            match = MIXED.match(row["operation"])
            if match and row["payload"] == "random" and row["active_sms"] == "all" and row["duty_pct"] == "100":
                probe.setdefault((int(row["footprint_bytes"]), int(match.group(1))), row)
    found = []
    for row in rows:
        accesses = number(row.get("accesses_per_launch"))
        if row["kernel"] != CONTROL or not accesses > 0:
            continue
        parameters = dict(item.split("=", 1) for item in row["configuration"].split())
        reference = probe.get((int(parameters["FOOTPRINT"]), int(parameters["MIX_FMAS"])))
        if reference is None:
            continue
        validation = number(row["dynamic_energy_j"]) / accesses * 1e12
        found.append((reference["level"], int(parameters["MIX_FMAS"]), int(parameters.get("LAUNCH_MS", 0)),
                      number(row["temperature_c"]), number(reference["temperature_c"]),
                      number(row["power_w"]) - number(row["baseline_w"]),
                      number(reference["power_w"]) - number(reference["baseline_w"]), validation,
                      number(reference["pj_per_access"])))
    if not found:
        print("\nThe validation run has no control kernel with a configuration of the --mixed run.")
        return
    print("\nControl kernel measured both ways (validation run vs --mixed probe run; pJ per load):")
    print(f"  {'level':6} {'FFMA/ld':>8} {'launch':>7} {'T val':>6} {'T probe':>8} {'P-base val':>11} {'P-base probe':>13} "
          f"{'pJ val':>9} {'pJ probe':>9} {'ratio':>7}")
    order = ["L1", "L2", "DRAM"]
    for level, fmas, launch, t_val, t_probe, p_val, p_probe, validation, reference in sorted(
            found, key=lambda f: (order.index(f[0]) if f[0] in order else 3, f[1], f[2])):
        ratio = f"{100 * (validation / reference - 1):+6.1f} %" if reference > 0 else "  (the probe row has no energy)"
        print(f"  {level:6} {fmas:8d} {launch:5d} ms {t_val:6.1f} {t_probe:8.1f} {p_val:9.1f} W {p_probe:11.1f} W "
              f"{validation:9.1f} {reference:9.1f} {ratio}")
    for launch in sorted(set(f[2] for f in found)):
        ratios = np.array([f[7] / f[8] - 1 for f in found if f[2] == launch and f[8] > 0])
        print(f"  {launch} ms launches: validation run {100 * ratios.mean():+.1f} % (sd {100 * ratios.std():.1f}) against "
              f"the probe run, {len(ratios)} configurations")


def write_outputs(path, results, energies, probes, table):
    """The predictions per configuration (CSV, with each event's share of the predicted dynamic energy) and the energies
    per event (<path>_energies.json)."""
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["kernel", "configuration", "kernel_s", "busy", "measured_j", "predicted_j", "simple_j",
                         "constant_power_j", "measured_dynamic_j", "predicted_dynamic_j"] + EVENTS)
        for r in results:
            writer.writerow([r["kernel"], r["configuration"], r["kernel_s"], r["busy"], r["measured_j"],
                             r["predicted_j"], r["simple_j"], r["constant_power_j"], r["measured_dynamic_j"],
                             r["predicted_dynamic_j"]] + [r["shares"][name] for name in EVENTS])
    with open(path.rsplit(".", 1)[0] + "_energies.json", "w") as file:
        json.dump({"pj_per_event": energies, "probes": len(probes), "device": table.get("device")}, file, indent=2)


# The counters the events are computed from, with the events each one feeds (tools/check_counters.py prints the same). A
# counter that a run did not collect counts as 0, which silently removes its events, so check_required_counters() compares
# the runs. Counters of the optional groups (COUNTER_GROUPS) and the fp16 counters (one of two names) are left to their own
# handling.
COUNTER_EVENTS = {name: entry["feeds"] for name, entry in counter_table.ENTRIES.items()
                  if "feeds" in entry and not {"alias_of", "alternative", "only_with"} & set(entry)}
# Counters that belong to the optional groups and are checked there (or that a GPU may lack by design: no LDGSTS before
# Ampere).
OPTIONAL_COUNTERS = {name for name, entry in counter_table.ENTRIES.items() if "group" in entry}


def counters_present(rows):
    """The counters of COUNTER_EVENTS that some row has a value of (under their name or an alias)."""
    return {name for name in COUNTER_EVENTS if any(not math.isnan(value_of(row, name)) for row in rows)}


def check_required_counters(sources):
    """Every counter an event needs must be in all counter runs (probes, mixed probes, validation) or in none: a counter
    that one of them lacks counts as 0 there, and the model would price the others' events into what remains. A counter
    that none has is reported, with the events that then read 0."""
    present = {name: counters_present(rows) for name, rows in sources.items() if rows}
    everywhere = set.intersection(*present.values()) if present else set()
    anywhere = set.union(*present.values()) if present else set()
    partial = sorted(name for name in anywhere - everywhere if name not in OPTIONAL_COUNTERS)
    if partial:
        lines = [f"  {name} ({COUNTER_EVENTS[name]}): in " + ", ".join(s for s, names in present.items() if name in names)
                 + "; not in " + ", ".join(s for s, names in present.items() if name not in names) for name in partial]
        raise SystemExit("counters the model needs are in some runs and not in others (they would count as 0 there):\n"
                         + "\n".join(lines) + "\nprofile the runs with the same counters, or map a renamed counter in "
                         "ALIASES")
    missing = sorted(name for name in COUNTER_EVENTS if name not in anywhere and name not in OPTIONAL_COUNTERS)
    if missing and present:
        print("No run has these counters, so their events read 0 everywhere:\n" + "\n".join(
            f"  {name} ({COUNTER_EVENTS[name]})" for name in missing) + "\n")


def counter_rows(prefixes):
    """The probe rows of profiled runs (random data, all SMs, duty 100) that have counters."""
    rows = []
    for prefix in prefixes:
        rows += [row for row in read_rows(prefix + ".csv") if row["active_sms"] == "all" and row["payload"] == "random"
                 and row["duty_pct"] == "100" and row.get("smsp__inst_executed.sum")]
    return rows


def check_counter_groups(sources):
    """The probes and the kernels they predict must count the same events: every optional counter group that the model
    uses must be in all counter runs or in none (then its pipe is one event, as before the group was collected)."""
    found = set().union(*sources.values())
    for group in (g for g in COUNTER_GROUPS if energy_events.FMA_HALVES or g != "fma pipe halves"):
        flags = {group in groups for groups in found}
        if len(flags) > 1:
            kinds = ", ".join(f"{name} " + ("with" if all(group in g for g in seen) else "without"
                                            if not any(group in g for g in seen) else "partly with")
                              for name, seen in sources.items() if seen)
            raise SystemExit(f"the {group} are in some counter runs and not in others ({kinds}); use runs profiled with "
                             "the same counters")
    if found and not any("lsu instructions by memory space" in groups for groups in found):
        print("The counter runs have no lsu instructions by memory space: all of them are one event, 'inst lsu other'.\n")
    if energy_events.FMA_HALVES and found and not any("fma pipe halves" in groups for groups in found):
        print("The counter runs have no fma pipe halves: the fma pipe stays one event, 'inst fma (fp32, imad)'.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power", required=True,
                        help="prefix of the probe power run (its _energy_table.json); several, comma-separated, are averaged")
    parser.add_argument("--counters", required=True,
                        help="prefix of the profiled probe run (its .csv); several, comma-separated, are searched in order")
    parser.add_argument("--validation", help="prefix of the --validate run (its _validation.csv)")
    parser.add_argument("--mixed",
                        help="prefix of a run with --mixed (its .csv); several, comma-separated, are searched in order. "
                             "Its mixed-intensity probes are predicted like the validation kernels, with their counts "
                             "from the --counters runs")
    parser.add_argument("--csv", help="write the predictions per configuration to this file")
    parser.add_argument("--cross-validate", action="store_true",
                        help="also predict every probe from a fit without it (leave-one-out)")
    parser.add_argument("--weighted", action="store_true",
                        help="weigh every family of probes about the same in the fit (default: every probe the same)")
    parser.add_argument("--fma-halves", action="store_true",
                        help="price the fma pipe's heavy (FP32, IMAD) and lite (FP32) halves apart (counter runs since "
                             "ProfileFull3; default: one fma event)")
    parser.add_argument("--no-busy", action="store_true",
                        help="without the DRAM busy-time event (the fixed power of steady DRAM traffic)")
    parser.add_argument("--busy-scale", type=float, default=BUSY_SCALE_GBS,
                        help=f"DRAM rate in GB/s of the busy event's shape 1 - exp(-rate / scale), as tools/fit_busy.py "
                             f"measures it (default {BUSY_SCALE_GBS:g}, from an RTX 3090)")
    arguments = parser.parse_args()
    configure(arguments.fma_halves, not arguments.no_busy, arguments.busy_scale)
    if not arguments.validation and not arguments.mixed:
        parser.error("give --validation, --mixed or both")
    if arguments.csv and not arguments.validation:
        parser.error("--csv writes the predictions of the --validation run")

    probes, table = probe_table(arguments.power.split(","), arguments.counters.split(","))
    if not probes:
        raise SystemExit("no probe has both an energy and counters; check the prefixes")
    mixed = load_mixed(arguments.mixed.split(","), arguments.counters.split(",")) if arguments.mixed else []
    if arguments.mixed and not mixed:
        raise SystemExit("no mixed-intensity probe has both an energy and counters; check --mixed and --counters")
    validation = load_validation(arguments.validation, table) if arguments.validation else []
    check_counter_groups({"probes (--counters)": {p.groups for p in probes}, "mixed probes": {m[4] for m in mixed},
                          "validation": {counter_groups(row) for row in validation}})
    check_required_counters({"probes (--counters)": counter_rows(arguments.counters.split(",")),
                             "validation": validation})

    if any(p.label == "compute ffma3" for p in probes):
        print("FFMA with three register operands (ffma3) in the probes: " + ", ".join(TWO_REGISTER_FFMA)
              + " (x * x + c) stay out of the fit.\n")
        probes = [p for p in probes if p.label not in TWO_REGISTER_FFMA]

    checks = [p for p in probes if p.label in CHECK_ONLY]
    probes = [p for p in probes if p.label not in CHECK_ONLY]
    weighted = arguments.weighted
    energies, fitted, unused = fit(probes, EVENTS, weighted)
    simple, _, _ = fit(probes, SIMPLE, weighted)
    print_fit(probes, fitted, energies, unused)
    print_combined(energies)
    for line in unpriced_events(validation, unused):
        print(line)
    if checks:
        print("\nProbes outside the fit (pJ per operation: measured, predicted, error):")
        for probe in checks:
            predicted = sum(probe.counts[name] * energies[name] for name in EVENTS)
            print(f"  {probe.label:22} {probe.energy:9.2f} {predicted:9.2f} {100 * (predicted / probe.energy - 1):+6.1f} %")
    if arguments.cross_validate:
        cross_validate(probes, weighted)
    if mixed:
        report_mixed(mixed, energies)
    if validation:
        results = predict(validation, energies, simple)
        report(results)
        if arguments.mixed and any(row["kernel"] == CONTROL for row in validation):
            report_control(validation, arguments.mixed.split(","))
        if arguments.csv:
            write_outputs(arguments.csv, results, energies, probes, table)


if __name__ == "__main__":
    main()
