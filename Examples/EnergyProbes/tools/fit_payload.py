#!/usr/bin/env python3
"""Energy over the data: fits of a --payload-bits run.

    python3 tools/fit_payload.py PayloadBits1[,PayloadBits2,...]   # their _fits.csv and _payloads.json

Several runs (each with zero and random, which every --payload-bits run measures) are combined on energies relative to
the same run's random payload, so that differences between runs cancel.

For every probe of the payload study (16-byte ld at L1, L2 and DRAM, the no-load control, and FFMA, DFMA and HMMA on
operands from the payload), the energy per access (the fit across duties) of every payload is set against the payload's
bit statistics (<prefix>_payloads.json: the fraction of one bits and of bits that toggle between consecutive words, or
between the two alternating operand sets for the arithmetic). Fitted on the training payloads (zero, random, ones:Q,
same:Q, flip:T):
- two factors, E = E0 + a x ones + b x toggles (as VAMPIRE for DRAM, Ghose et al., arXiv 1807.05102), with R2;
- one factor each, toggles only and ones only, for comparison;
and the held-out payloads (unique:K, lsb0:N, sorted: Gregersen et al., arXiv 2409.18324) predicted by the two-factor fit.
The no-load control must come out flat: if it does not, the differences are not the data's.
"""

import argparse
import csv
import json
import math
import sys

import numpy as np

TRAINING = ("zero", "random", "ones:", "same:", "flip:")
PROBES = [("L1", "ld"), ("L2", "ld"), ("DRAM", "ld"), ("none", "no-load"),
          ("compute", "ffma-bits"), ("compute", "dfma-bits"), ("compute", "hmma-bits")]


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def training(payload):
    return payload in ("zero", "random") or payload.startswith(TRAINING[2:])


def least_squares(x, y):
    a = np.c_[np.ones(len(y)), x] if x.ndim == 2 else np.c_[np.ones(len(y)), x[:, None]]
    coefficients, *_ = np.linalg.lstsq(a, y, rcond=None)
    fitted = a @ coefficients
    total = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum((y - fitted) ** 2) / total if total > 0 else math.nan
    return coefficients, fitted, r2


CANDIDATES = {"words": ["toggles", "toggles_lag8", "toggles_lag32"], "operands": ["toggles", "lanes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", help="--output prefixes of runs with --payload-bits, comma-separated")
    parser.add_argument("--stats", help="bit statistics to use instead of the runs' _payloads.json (for runs whose file "
                        "lacks the lagged and lane toggles)")
    parser.add_argument("--skip", default="", help="payloads to leave out, comma-separated; RUN:PAYLOAD leaves it out of "
                        "that run only (runs before 2026-11-06 21:00 read random data for 'zero')")
    arguments = parser.parse_args()
    statistics, fits = {}, []
    runs = arguments.runs.split(",")
    for run in runs:
        with open(run + "_payloads.json") as file:
            statistics.update(json.load(file))
        with open(run + "_fits.csv") as file:
            fits.append(list(csv.DictReader(file)))
    if arguments.stats:
        with open(arguments.stats) as file:
            statistics.update(json.load(file))
    skip = set(filter(None, arguments.skip.split(",")))
    skipped = lambda run, payload: payload in skip or f"{run}:{payload}" in skip

    for level, operation in PROBES:
        kind = operation if operation.endswith("-bits") else "words"
        # Energies relative to the run's own random payload, averaged over the runs that have a payload, then scaled by
        # the mean random energy of the runs.
        relative, randoms = {}, []
        for run, run_fits in zip(runs, fits):
            run_points = {f["payload"]: number(f["pj_per_access"]) for f in run_fits
                          if f["level"] == level and f["operation"] == operation and f["payload"] in statistics
                          and not skipped(run, f["payload"])
                          and (operation not in ("ld", "no-load") or f["width_bytes"] == "16")}
            if "random" not in run_points or not run_points["random"] > 0:
                continue
            randoms.append(run_points["random"])
            for payload, energy in run_points.items():
                relative.setdefault(payload, []).append(energy / run_points["random"])
        if len(relative) < 3:
            continue
        random_energy = float(np.mean(randoms))
        points = {payload: float(np.mean(values)) * random_energy for payload, values in relative.items()}
        candidates = [c for c in CANDIDATES["words" if kind == "words" else "operands"] if c in statistics["random"][kind]]
        print(f"\n{level} {operation}: pJ per access (per operation of one thread for the arithmetic), {kind} statistics; "
              f"{len(randoms)} runs, random {random_energy:.2f} pJ"
              + (f" (runs {', '.join(f'{r:.2f}' for r in randoms)})" if len(randoms) > 1 else ""))
        print(f"  {'payload':12} {'ones':>6} " + " ".join(f"{c:>13}" for c in candidates) + f" {'pJ':>10} {'/ random':>9}")
        for payload, energy in sorted(points.items(), key=lambda p: (not training(p[0]), p[0])):
            s = statistics[payload][kind]
            print(f"  {payload:12} {s['ones']:6.3f} " + " ".join(f"{s.get(c, math.nan):13.3f}" for c in candidates)
                  + f" {energy:10.2f} {energy / random_energy:9.3f}" + ("" if training(payload) else "   (held out)"))
        train = [p for p in points if training(p) and not math.isnan(points[p])]
        test = [p for p in points if not training(p) and not math.isnan(points[p])]
        if len(train) < 4:
            print("  fewer than 4 training payloads: no fit")
            continue
        energy = np.array([points[p] for p in train])
        ones = np.array([statistics[p][kind]["ones"] for p in train])
        _, _, r2_ones = least_squares(ones, energy)
        print(f"  ones only: R2 {r2_ones:.3f}")
        for candidate in candidates:
            toggles = np.array([statistics[p][kind][candidate] for p in train])
            both, fitted, r2 = least_squares(np.c_[ones, toggles], energy)
            _, _, r2_toggles = least_squares(toggles, energy)
            rms = math.sqrt(np.mean((energy - fitted) ** 2))
            held = []
            for payload in test:
                s = statistics[payload][kind]
                predicted = both[0] + both[1] * s["ones"] + both[2] * s[candidate]
                held.append(f"{payload} {100 * (predicted / points[payload] - 1):+.1f} %")
            print(f"  {candidate:13} E = {both[0]:.2f} {both[1]:+.2f} x ones {both[2]:+.2f} x toggles: R2 {r2:.3f} "
                  f"(toggles only {r2_toggles:.3f}), rms {100 * rms / random_energy:.1f} % of random"
                  + ("; held out: " + ", ".join(held) if held else ""))


if __name__ == "__main__":
    sys.exit(main())
