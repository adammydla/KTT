#!/usr/bin/env python3
"""Does a model file fit this GPU? A check of a few minutes, without profiling.

A model is calibrated on one GPU at one SM clock. Before a tuner trusts it on another card of the same type, after a
driver update, or at another clock, run a short probe run at full rate there and compare it with what the model's own
calibration run measured for the same probes:

    ./EnergyProbesCuda $GPU --mt4g FILE --compute --duty 100 --warmup-s 30 --idle-s 0 \
        --min-time-ms 5000 --max-time-ms 10000 --output Quick                         # about 5 min
    python3 tools/check_model.py --model rtx3090_1200mhz.model.json --run Quick

It compares, per probe: the time per access (the clock and the memory behave as in the calibration), and the energy per
access at full rate. It also compares the baseline power at the run's temperature (40-60 % of a kernel's energy is the
baseline). With --counters, the event counts per access of a short profiled run (--quick --profile) are compared with the
calibration's, which shows a driver or GPU that counts differently. The verdict:

  PASS      times within 3 %, energies: median within 5 %, 80 % of the probes within 10 %, the baseline within 3 %, and
            (with --counters) the counts as in the calibration. Use the model as it is.
  BASELINE  as PASS, but the baseline differs by more than 3 % (on the second RTX 3090 its slope over temperature was 0.73
            instead of 0.28 W/C): --write FILE writes a model with this GPU's baseline.
  SCALED    times agree, the energies are off by a common factor (their spread around the median is small). The model's
            ranking holds, its absolute energies need the factor; --write FILE writes a model with the factor and this
            GPU's baseline applied.
  FAIL      times differ (another clock or memory configuration), the energies differ probe by probe, or the counts
            differ. Calibrate this GPU (docs/CALIBRATION.md: Porting to another GPU).

Only probes that the run measured at a clock matching its own baseline count; with rows withheld for that reason the
GPU's clock is not constant (not locked, or throttled), and a check at full rate cannot tell how the model does there.
"""

import argparse
import copy
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_table  # noqa: E402
import energy_events  # noqa: E402
import energy_predictor  # noqa: E402

number = counter_table.number

TIME_TOLERANCE = 0.03
MEDIAN_TOLERANCE = 0.05
PROBE_TOLERANCE = 0.10
PROBE_SHARE = 0.80
# Spread of the energy ratios around their median below which a common factor describes them (median absolute deviation).
SCALE_SPREAD = 0.05
# Probes this close to the baseline are too noisy for a 5 s window (W above the baseline in the calibration).
MIN_DYNAMIC_W = 10.0
MIN_PROBES = 8
# The baseline at the run's temperature against the model's (Quick0 on the second RTX 3090: +0.3 %).
BASELINE_TOLERANCE = 0.03
# Counts: a probe differs when its events, priced with the model, move its energy by more than this; the check fails when
# a probe moves by more than COUNT_FAIL or more than COUNT_SHARE of the probes differ. Profiled counts of the same probe
# agree within 0.5 % between runs on one GPU.
COUNT_TOLERANCE = 0.05
COUNT_FAIL = 0.10
COUNT_SHARE = 0.10


def compare(model, rows, use_withheld=False):
    """(label, time ratio, energy ratio, dynamic W) per probe the run and the model both have; and the labels left out
    because their clock did not match the run's baseline."""
    reference = {p["label"]: p for p in model.data["probes"] if p.get("full_rate_pj_per_access")}
    found, withheld = [], []
    for row in rows:
        if row["payload"] != "random" or row["active_sms"] != "all" or row["duty_pct"] != "100":
            continue
        label = f"{row['level']} {row['operation']}" + (f" w{int(row['width_bytes'])}" if int(row["width_bytes"]) else "")
        probe = reference.get(label)
        accesses = number(row["accesses_per_launch"])
        if probe is None or not accesses > 0 or probe["dynamic_w"] < MIN_DYNAMIC_W:
            continue
        clock = number(row.get("clock_difference_pct"))
        energy = number(row["pj_per_access"])
        if not math.isnan(clock) and abs(clock) > 2 and not use_withheld:
            withheld.append(label)
            continue
        if math.isnan(energy):
            dynamic = number(row["power_w"]) - number(row["baseline_w"])
            energy = dynamic * number(row["active_s"]) / accesses * 1e12 if use_withheld else math.nan
        if not energy > 0:
            withheld.append(label)
            continue
        ns = number(row["active_s"]) / accesses * 1e9
        found.append((label, ns / probe["ns_per_access"], energy / probe["full_rate_pj_per_access"], probe["dynamic_w"],
                      number(row["sm_clock_mhz"])))
    return found, withheld


def compare_counts(model, rows):
    """(label, relative change of the probe's model energy, the event that moves it most) per probe that the profiled run
    and the model file both have. Model files from before 2026-11-13 have no counts: None."""
    reference = {p["label"]: p["counts"] for p in model.data["probes"] if p.get("counts")}
    if not reference:
        return None
    found = []
    for row in rows:
        if row["payload"] != "random" or row["active_sms"] != "all" or row["duty_pct"] != "100" \
                or not row.get("smsp__inst_executed.sum"):
            continue
        label = f"{row['level']} {row['operation']}" + (f" w{int(row['width_bytes'])}" if int(row["width_bytes"]) else "")
        accesses = number(row["accesses_per_launch"])
        if label not in reference or not accesses > 0:
            continue
        counts = {name: value / accesses for name, value in energy_events.events(row).items()}
        expected = reference[label]
        energy = sum(expected.get(name, 0.0) * model.pj.get(name, 0.0) for name in expected)
        if not energy > 0:
            continue
        moves = {name: (counts.get(name, 0.0) - expected.get(name, 0.0)) * model.pj.get(name, 0.0) / energy
                 for name in set(counts) | set(expected) if name != "DRAM busy ns"}
        worst = max(moves, key=lambda name: abs(moves[name]))
        found.append((label, sum(moves.values()), worst, sum(abs(v) for v in moves.values())))
    return found


def run_baseline(rows):
    """Mean power and temperature of the run's sleep rows."""
    sleeps = [(number(r["power_w"]), number(r["temperature_c"])) for r in rows
              if r["level"] == "baseline" and r["active_sms"] == "all" and r["probe"].startswith("sleep-")]
    if not sleeps:
        return math.nan, math.nan
    return statistics.mean(p for p, _ in sleeps), statistics.mean(t for _, t in sleeps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model file (tools/build_model.py)")
    parser.add_argument("--run", required=True, help="prefix of the short probe run on this GPU (its .csv)")
    parser.add_argument("--write", help="with the verdict SCALED (or PASS): write a model with the energies times the "
                        "common factor and this run's baseline")
    parser.add_argument("--counters", help="prefix of a short profiled run on this GPU (--quick --profile): compare its "
                        "event counts per access with the calibration's")
    parser.add_argument("--use-withheld", action="store_true",
                        help="also compare probes whose clock differed from the run's baseline (to look at an unlocked "
                             "GPU; their energies are biased by the baseline's clock, so no verdict follows)")
    parser.add_argument("--verbose", action="store_true", help="print every probe")
    arguments = parser.parse_args()

    model = energy_predictor.Model.load(arguments.model)
    rows = counter_table.read_rows(arguments.run + ".csv")
    found, withheld = compare(model, rows, arguments.use_withheld)
    print(f"Model {os.path.basename(arguments.model)}: {model.device.get('name')} at {model.device.get('sm_clock_mhz')} MHz")
    clocks = [f[4] for f in found if not math.isnan(f[4])]
    if clocks:
        print(f"Run {arguments.run}: {len(found)} probes compared, SM clock {min(clocks):.0f}-{max(clocks):.0f} MHz"
              + (f", {len(withheld)} left out (clock differs from the run's baseline, or no energy)" if withheld else ""))
    if len(found) < MIN_PROBES:
        print(f"FAIL: only {len(found)} probes can be compared (need {MIN_PROBES})"
              + (": the clock is not constant on this GPU, lock it or calibrate at its own clock" if withheld else ""))
        return 1

    times = [f[1] for f in found]
    energies = [f[2] for f in found]
    median = statistics.median(energies)
    spread = statistics.median(abs(e / median - 1) for e in energies)
    within = sum(abs(e - 1) <= PROBE_TOLERANCE for e in energies) / len(energies)
    slow = [f for f in found if abs(f[1] - 1) > TIME_TOLERANCE]
    base_w, base_t = run_baseline(rows)
    expected = model.baseline_w(base_t)

    if arguments.verbose:
        print(f"  {'probe':28} {'time':>8} {'energy':>8} {'W above baseline (model)':>25}")
        for label, time, energy, dynamic, _ in sorted(found, key=lambda f: -abs(f[2] - 1)):
            print(f"  {label:28} {100 * (time - 1):+7.1f}% {100 * (energy - 1):+7.1f}% {dynamic:25.1f}")
    print(f"  time per access:   median {100 * (statistics.median(times) - 1):+.1f} %, {len(slow)} of {len(found)} probes "
          f"off by more than {100 * TIME_TOLERANCE:.0f} %"
          + (" (" + ", ".join(f"{f[0]} {100 * (f[1] - 1):+.0f} %" for f in slow[:4]) + (", ..." if len(slow) > 4 else "") + ")"
             if slow else ""))
    print(f"  energy per access: median {100 * (median - 1):+.1f} %, spread around it {100 * spread:.1f} %, "
          f"{100 * within:.0f} % of the probes within {100 * PROBE_TOLERANCE:.0f} %")
    if not math.isnan(base_w):
        print(f"  baseline: {base_w:.1f} W at {base_t:.1f} C, the model expects {expected:.1f} W "
              f"({100 * (base_w / expected - 1):+.1f} %)")

    counts_ok = True
    if arguments.counters:
        compared = compare_counts(model, counter_table.read_rows(arguments.counters + ".csv"))
        if compared is None:
            print("  counts: the model file has no counts per probe (rebuild it with tools/build_model.py)")
        elif not compared:
            print(f"  counts: no probe of {arguments.counters} is in the model file")
            counts_ok = False
        else:
            off = [c for c in compared if c[3] > COUNT_TOLERANCE]
            bad = [c for c in compared if c[3] > COUNT_FAIL]
            counts_ok = not bad and len(off) <= COUNT_SHARE * len(compared)
            print(f"  counts: {len(compared)} probes compared, {len(off)} move their model energy by more than "
                  f"{100 * COUNT_TOLERANCE:.0f} %" + (" (" + ", ".join(f"{c[0]} {100 * c[3]:.0f} % ({c[2]})"
                                                                         for c in sorted(off, key=lambda c: -c[3])[:4])
                                                       + ")" if off else ""))
    baseline_ok = math.isnan(base_w) or abs(base_w / expected - 1) <= BASELINE_TOLERANCE

    if arguments.use_withheld:
        print("No verdict with --use-withheld: the energies above include rows at another clock than their baseline.")
        return 0
    if len(slow) > 0.2 * len(found):
        verdict = "FAIL"
        print("FAIL: the probes run at another speed than in the calibration (another clock or memory configuration).")
    elif not counts_ok:
        verdict = "FAIL"
        print("FAIL: the event counts differ from the calibration's (another GPU or driver counts differently); "
              "calibrate this GPU.")
    elif abs(median - 1) <= MEDIAN_TOLERANCE and within >= PROBE_SHARE and baseline_ok:
        verdict = "PASS"
        print("PASS: the model fits this GPU.")
    elif abs(median - 1) <= MEDIAN_TOLERANCE and within >= PROBE_SHARE:
        verdict = "BASELINE"
        print(f"BASELINE: the energies fit, the baseline differs by {100 * (base_w / expected - 1):+.1f} % "
              f"(more than {100 * BASELINE_TOLERANCE:.0f} %); --write takes this GPU's baseline.")
    elif spread <= SCALE_SPREAD:
        verdict = "SCALED"
        print(f"SCALED: the energies differ by a common factor of {median:.3f}; the model ranks configurations as "
              "before, absolute energies need the factor (--write)."
              + ("" if baseline_ok else f" The baseline differs by {100 * (base_w / expected - 1):+.1f} % too."))
    else:
        verdict = "FAIL"
        print("FAIL: the energies differ probe by probe; calibrate this GPU.")

    if arguments.write and verdict != "FAIL":
        data = copy.deepcopy(model.data)
        factor = median if verdict == "SCALED" else 1.0
        for entry in data["events"].values():
            entry["pj"] *= factor
        if not math.isnan(base_w):
            data["baseline"]["mean_w"] = base_w
            data["baseline"]["mean_temperature_c"] = base_t
        data["adjusted"] = {"from": os.path.basename(arguments.model), "run": os.path.basename(arguments.run),
                            "energy_factor": factor, "probes": len(found), "verdict": verdict}
        with open(arguments.write, "w") as file:
            json.dump(data, file, indent=1)
            file.write("\n")
        print(f"Wrote {arguments.write} (energies x {factor:.3f}, baseline {data['baseline']['mean_w']:.1f} W at "
              f"{data['baseline']['mean_temperature_c']:.1f} C)")
    return 0 if verdict != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
