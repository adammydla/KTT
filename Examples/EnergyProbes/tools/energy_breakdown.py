#!/usr/bin/env python3
"""How well a model file predicts measured kernels, and where their energy goes (2026-11-16).

For every configuration of --validate runs (power, time and counters measured), the model file predicts the energy of
a launch the way a tuner would (energy_predictor.Model, at the row's temperature) and this compares it with the
measurement. Only configurations at the model's clock count (energy_model.CLOCK_TOLERANCE). Per kernel it prints:

  1. accuracy: the measured power range, the error of the predicted average power and total energy, the scatter and
     bias of the dynamic energy, the rank correlation of the energies, the regret of the model's choice (next to the
     fastest configuration's), and the error by speed (fastest, middle and slowest third: the bad configurations);
  2. where the energy goes: the baseline (the GPU awake for the kernel's time) and the dynamic energy of each part
     (compute, instruction issue, L1 and shared memory, constant cache, L2, DRAM) in the measured-best configuration
     and in the median one;
  3. what lowers it: from the median configuration to the best one, how much each part saves (predicted) next to the
     measured total saving, and the part that saves most.

The parts are the model's events (energy_events.py) grouped; their prices are statistical proxies fitted over the
probes (docs/PROBES.md, "What the prices are"), so a part's energy is the model's attribution, not a measurement of
that unit. Only the total is measured.

    python3 tools/energy_breakdown.py --model models/rtx3090_1200mhz.model.json --validation GemmMany1 [--csv out.csv]
"""

import argparse
import collections
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import energy_model as m  # noqa: E402
import energy_predictor  # noqa: E402

PARTS = collections.OrderedDict([
    ("compute", ("inst alu", "inst fma (fp32, imad)", "inst fp16", "inst fp64", "inst tensor", "inst tensor imma",
                 "inst xu (mufu)")),
    ("issue", ("inst lsu shared", "inst lsu global", "inst lsu other", "inst other", "inst tex", "inst inactive lanes")),
    ("L1/shared", ("L1 sector", "L1 tex sector", "shared ld wavefront", "shared st wavefront", "shared ld conflict",
                   "shared st conflict", "shared atom wavefront")),
    ("constant", ("const request", "const L1 miss")),
    ("L2", ("L2 read sector", "L2 write sector", "L2 read request", "L2 write request", "L2 atom sector",
            "L2 red sector")),
    ("DRAM", ("L2 read miss request", "L2 partial write miss request", "DRAM read sector", "DRAM write sector",
              "DRAM busy ns")),
])
assert sorted(e for events in PARTS.values() for e in events) == sorted(m.EVENTS), "every event in one part"

# What a part's energy means for the programmer, for the advice line.
ADVICE = {
    "baseline": "run faster (the GPU draws its baseline for the whole kernel time): more parallelism, fewer stalls",
    "compute": "fewer or cheaper arithmetic instructions (no redundant work, lower precision where allowed)",
    "issue": "fewer load/store and control instructions (wider vector accesses, less index arithmetic and branching) "
             "and fewer partly idle warps (divergent branches leave lanes inactive that still cost a third of a warp)",
    "L1/shared": "fewer L1 and shared-memory accesses (reuse values in registers, avoid bank conflicts)",
    "constant": "fewer constant-cache misses",
    "L2": "fewer L2 transactions (better reuse in L1/shared, full 32-byte sectors per request)",
    "DRAM": "less DRAM traffic (tiling for reuse, coalesced full-line accesses, no repeated reads)",
}


def parts_of(prediction, model):
    """Joules per launch of the baseline and of every part, from one prediction. The parts add up to the prediction's
    dynamic energy: at the power limit (the prediction is the limit times the time) and with a kernel scale, every part
    is scaled by the same factor, as Antepara et al. (SC'25) rescale the energies per operation of a capped run."""
    joules = {"baseline": prediction.baseline_j}
    for part, events in PARTS.items():
        joules[part] = sum(prediction.events.get(e, 0.0) * model.pj.get(e, 0.0) for e in events) * 1e-12
    total = sum(v for k, v in joules.items() if k != "baseline")
    if total > 0:
        factor = prediction.dynamic_j / total
        joules.update({k: v * factor for k, v in joules.items() if k != "baseline"})
    return joules


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else math.nan


def analyse(model, prefixes):
    table = {"baseline": {"slope_w_per_c": model.baseline["slope_w_per_c"]}}
    kernels = collections.OrderedDict()
    left_out = 0
    for prefix in prefixes:
        rows = m.load_validation(prefix, table)
        kept, off = m.at_model_clock(rows, model.device.get("sm_clock_mhz"))
        left_out += len(off)
        for row in kept:
            p = model.predict(row, m.number(row["kernel_s"]), m.number(row["temperature_c"]),
                              m.number(row["sm_clock_mhz"]), row["configuration"], strict=False)
            kernels.setdefault(row["kernel"], []).append((row, p))
    return kernels, left_out


def report(model, kernels, csv_path=None):
    records = []
    print(f"Model: {model.device.get('name')} at {model.device.get('sm_clock_mhz')} MHz; parts are the model's attribution "
          "(statistical proxy prices), only totals are measured\n")
    print("Power and energy per launch (power err: mean |predicted / measured - 1| of the average power, the same for the"
          " total energy; rank r: Spearman correlation of predicted and measured; scatter, bias: of the dynamic energy)")
    print(f"{'kernel':14} {'n':>4} {'power W':>12} {'power err':>9} {'power r':>7} {'scatter':>7} {'bias':>6} "
          f"{'energy r':>8} {'regret':>6} {'time only':>9}   dynamic error fast / mid / slow third (mean)")
    summary = []
    for kernel, entries in kernels.items():
        rows = [r for r, _ in entries]
        preds = [p for _, p in entries]
        seconds = np.array([m.number(r["kernel_s"]) for r in rows])
        measured = np.array([r["energy_j"] for r in rows])
        measured_dynamic = np.array([r["dynamic_energy_j"] for r in rows])
        predicted = np.array([p.energy_j for p in preds])
        dynamic = np.array([p.dynamic_j for p in preds])
        power = measured / seconds
        predicted_power = predicted / seconds
        error = dynamic / measured_dynamic - 1
        regret = measured[np.argmin(predicted)] / measured.min() - 1
        time_regret = measured[np.argmin(seconds)] / measured.min() - 1
        order = np.argsort(seconds)
        thirds = [error[order[i * len(order) // 3:(i + 1) * len(order) // 3]].mean() for i in range(3)]
        power_error = np.abs(predicted_power / power - 1).mean()
        summary.append((kernel, len(rows), power_error, spearman(predicted_power, power), error.std(),
                        error.mean(), spearman(predicted, measured), regret, time_regret))
        print(f"{kernel:14} {len(rows):4d} {power.min():5.0f}-{power.max():<5.0f} {100 * power_error:8.1f}% "
              f"{spearman(predicted_power, power):7.2f} {100 * error.std():6.1f}% {100 * error.mean():+5.1f}% "
              f"{spearman(predicted, measured):8.2f} {100 * regret:5.1f}% {100 * time_regret:8.1f}%   "
              + " / ".join(f"{100 * t:+.1f}%" for t in thirds))
        for r, p, e in zip(rows, preds, error):
            records.append({"kernel": kernel, "configuration": r["configuration"], "kernel_s": r["kernel_s"],
                            "measured_j": r["energy_j"], "predicted_j": p.energy_j,
                            "measured_power_w": r["energy_j"] / m.number(r["kernel_s"]),
                            "predicted_power_w": p.energy_j / m.number(r["kernel_s"]), "dynamic_error": e,
                            **{f"{part}_j": value for part, value in parts_of(p, model).items()}})
    if summary:
        s = np.array([x[2:] for x in summary], dtype=float)
        print(f"{'mean':14} {sum(x[1] for x in summary):4d} {'':12} {100 * s[:, 0].mean():8.1f}% {s[:, 1].mean():7.2f} "
              f"{100 * s[:, 2].mean():6.1f}% {100 * s[:, 3].mean():+5.1f}% {s[:, 4].mean():8.2f} {100 * s[:, 5].max():5.1f}% "
              f"{100 * s[:, 6].max():8.1f}%   (regrets: max)")

    print("\nLowest power is not lowest energy: the configuration with the lowest average power, measured, against the one "
          "with the lowest energy, and whether the model finds each")
    print(f"  {'kernel':14} {'min-power config':>32} {'min-energy config':>34}")
    for kernel, entries in kernels.items():
        rows = [r for r, _ in entries]
        preds = [p for _, p in entries]
        seconds = np.array([m.number(r["kernel_s"]) for r in rows])
        measured = np.array([r["energy_j"] for r in rows])
        power = measured / seconds
        predicted_power = np.array([p.energy_j for p in preds]) / seconds
        low_power, low_energy = int(np.argmin(power)), int(np.argmin(measured))
        found = int(np.argmin(predicted_power))
        print(f"  {kernel:14} {power[low_power]:6.0f} W, {100 * (measured[low_power] / measured[low_energy] - 1):+5.0f} % energy, "
              f"{seconds[low_power] / seconds[low_energy]:4.1f}x time   {power[low_energy]:6.0f} W   model's min-power pick: "
              f"{power[found]:.0f} W ({100 * (power[found] / power[low_power] - 1):+.1f} %)")

    print("\nWhere the energy goes (share of the predicted total per launch) and what lowers it (median -> best):")
    names = ["baseline"] + list(PARTS)
    print(f"  {'kernel':14} {'config':7} " + " ".join(f"{n:>9}" for n in names) + "   saving: predicted / measured, largest parts")
    for kernel, entries in kernels.items():
        rows = [r for r, _ in entries]
        preds = [p for _, p in entries]
        measured = np.array([r["energy_j"] for r in rows])
        best = int(np.argmin(measured))
        median = int(np.argsort(measured)[len(measured) // 2])
        shares = {}
        for label, index in (("best", best), ("median", median)):
            parts = parts_of(preds[index], model)
            total = sum(parts.values())
            shares[label] = parts
            line = f"  {kernel if label == 'best' else '':14} {label:7} " + " ".join(f"{100 * parts[n] / total:8.0f}%"
                                                                                      for n in names)
            if label == "median":
                saving = {n: shares["median"][n] - shares["best"][n] for n in names}
                predicted_saving = sum(saving.values())
                measured_saving = measured[median] - measured[best]
                top = sorted(saving.items(), key=lambda item: -item[1])[:2]
                line += (f"   {100 * predicted_saving / sum(shares['median'].values()):.0f} % / "
                         f"{100 * measured_saving / measured[median]:.0f} %, " + ", ".join(
                             f"{n} {100 * v / predicted_saving:.0f} %" for n, v in top if predicted_saving > 0))
            print(line)
    print("\nAdvice per kernel (the part that saves most from the median to the best configuration):")
    for kernel, entries in kernels.items():
        rows = [r for r, _ in entries]
        preds = [p for _, p in entries]
        measured = np.array([r["energy_j"] for r in rows])
        best, median = int(np.argmin(measured)), int(np.argsort(measured)[len(measured) // 2])
        saving = {n: parts_of(preds[median], model)[n] - parts_of(preds[best], model)[n] for n in names}
        part = max(saving, key=saving.get)
        print(f"  {kernel:14} {part}: {ADVICE[part]}")

    if csv_path:
        with open(csv_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nWrote {csv_path} ({len(records)} configurations)")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model file (tools/build_model.py)")
    parser.add_argument("--validation", required=True, help="--validate run prefixes, comma-separated")
    parser.add_argument("--leave-out", default="", help="kernels to leave out, comma-separated")
    parser.add_argument("--csv", help="write one row per configuration with the parts to this file")
    parser.add_argument("--power-limit", type=float, help="the GPU's power limit in W: configurations predicted above it "
                        "get the limit times their time, their parts scaled to match (default: the model file's, 0 = none)")
    arguments = parser.parse_args()
    model = energy_predictor.Model.load(arguments.model)
    model.power_limit_w = energy_predictor.file_power_limit(model) if arguments.power_limit is None \
        else (arguments.power_limit or None)
    kernels, left_out = analyse(model, arguments.validation.split(","))
    for name in filter(None, arguments.leave_out.split(",")):
        kernels.pop(name, None)
    capped = sum(p.clamped for rows in kernels.values() for _, p in rows)
    if capped:
        print(f"{capped} configurations predicted above the power limit of {model.power_limit_w:.0f} W: the limit times "
              "their time, their parts scaled to match")
    if left_out:
        print(f"{left_out} configurations more than {100 * m.CLOCK_TOLERANCE:.0f} % off the model's clock left out")
    report(model, kernels, arguments.csv)


if __name__ == "__main__":
    main()
