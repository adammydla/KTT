#!/usr/bin/env python3
"""Screening of candidate events (NEXT_STEPS_PLAN M3 step 4): every screening counter of the counter table (entries with
"study": true, collected with --study-counters) is added in turn to the model's events, as one more event priced by the
fit, and judged by

  (a) the rms error of the fit on the probes that the present events describe worst: the misaligned and write-path
      probes (-off, -sub-), the L1 sweep (-s<n>-l<n>) and the loads with integer work (ld-int<n>), when present;
  (b) the scatter of the design kernels (transpose, gemm, nbody, coulomb) on the validation runs;
  (c) the median fit error over all probes.

A candidate is kept only if (a) and (b) both improve; with about 30 candidates some will look good by chance, so a kept
one must be confirmed on the first held-out set (reduction, convolution, pnpoly) before it enters, and the screening be
named in the write-up. The counters must be in the --counters runs (probes) and in the validation runs.

    python3 tools/screen_counters.py --power Preset3 --counters ProfileStudy1 --full-rate MisalignedWrite2,L1Sweep1 \\
        --validation ValidationStudy1 [--counter NAME ...]
"""
import argparse
import contextlib
import io
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_table  # noqa: E402
import energy_model as m  # noqa: E402

DESIGN = ("transpose", "gemm", "nbody", "coulomb")
SASS_CLASSES = ["fadd", "fmul", "ffma", "fp64", "fp16", "integer", "bit", "conversion", "control", "memory", "misc",
                "uniform", "inter_thread_communication"]


def value(row, name):
    """A counter of a row. SASS counters (collected by patching the kernel) are scaled to the hardware count: a probe runs
    for a fixed time, and in the instrumented pass it gets through far fewer iterations (about 250x on the RTX 3090), so
    only their shares are right; on kernels of fixed work the scale is about 1."""
    raw = m.number(row.get(name))
    if "__sass_" not in name or math.isnan(raw):
        return raw
    total = sum(max(m.number(row.get(f"smsp__sass_thread_inst_executed_op_{c}_pred_on.sum")), 0) for c in SASS_CLASSES)
    hardware = m.number(row.get("smsp__thread_inst_executed_pred_on.sum"))
    return raw * hardware / total if total > 0 and hardware > 0 else math.nan
HARD = re.compile(r"-off\d+|-sub-|-s\d+-l\d+|ld-int\d+")


def probe_rows(counters):
    """The counter rows of the probes, by the probe label energy_model.probe_table gives them."""
    rows = {}
    for prefix in counters:
        for row in m.read_rows(prefix + ".csv"):
            if row["active_sms"] != "all" or row["payload"] != "random" or row["duty_pct"] != "100":
                continue
            key = (row["level"], row["operation"], int(row["width_bytes"]))
            label = f"{key[0]} {key[1]}" + (f" w{key[2]}" if key[2] else "")
            rows.setdefault(label, row)
    return rows


def scores(probes, kernels, names):
    """(rms on the hard probes, design scatter, median fit error, energies)."""
    energies, fitted, _ = m.fit(probes, names)
    errors = np.array([f / p.energy - 1 for f, p in zip(fitted, probes)])
    hard = np.array([e for e, p in zip(errors, probes) if HARD.search(p.label)])
    variances = []
    for key, group in kernels.items():
        if key[1] not in DESIGN or len(group) < 2:
            continue
        e = np.array([sum(c[n] * energies[n] for n in names) * 1e-12 / measured - 1 for c, measured in group])
        variances.append(e.var(ddof=1))
    return (float(np.sqrt(np.mean(hard ** 2))) if len(hard) else math.nan, math.sqrt(np.mean(variances)),
            float(np.median(np.abs(errors))), energies)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power", required=True)
    parser.add_argument("--counters", required=True, help="profiled probe runs with --study-counters, comma-separated")
    parser.add_argument("--full-rate", help="runs at one duty, as in energy_model.py")
    parser.add_argument("--validation", required=True, help="validation runs with --study-counters, comma-separated")
    parser.add_argument("--counter", action="append", help="screen only these counters (default: every study counter)")
    args = parser.parse_args()
    split = lambda text: text.split(",") if text else []  # noqa: E731
    counters = split(args.counters)
    with contextlib.redirect_stdout(io.StringIO()):
        probes, table = m.probe_table(split(args.power), counters, split(args.full_rate))
        validation = []
        for run in split(args.validation):
            validation += [(run, r) for r in m.load_validation(run, table) if r["kernel"] != m.CONTROL]
    if any(p.label == "compute ffma3" for p in probes):
        probes = [p for p in probes if p.label not in m.TWO_REGISTER_FFMA]
    probes = [p for p in probes if p.label not in m.CHECK_ONLY]
    rows = probe_rows(counters)
    candidates = args.counter or [e["name"] for e in counter_table.ENTRIES.values() if e.get("study")]
    # Only the probes whose counter rows have the screening counters (reference and candidates on the same probes).
    marker = "smsp__inst_issued.sum"
    with_study = [p for p in probes if not math.isnan(m.number(rows.get(p.label, {}).get(marker)))]
    if len(with_study) < len(probes):
        print(f"{len(probes) - len(with_study)} probes have no screening counters (their families were not in the counter "
              f"run with --study-counters) and stay out of the screening")
        probes = with_study

    def kernels_with(candidate):
        result = {}
        for run, r in validation:
            counts = m.add_busy(m.events(r), m.number(r["kernel_s"]) * 1e9)
            if candidate:
                counts[candidate] = max(value(r, candidate), 0.0)
            result.setdefault((run, r["kernel"]), []).append((counts, float(r["dynamic_energy_j"])))
        return result

    base = scores(probes, kernels_with(None), list(m.EVENTS))
    print(f"{len(probes)} probes ({sum(1 for p in probes if HARD.search(p.label))} of them hard), "
          f"{len(validation)} validation configurations")
    print(f"reference: hard probes rms {100 * base[0]:.1f} %, design scatter {100 * base[1]:.2f} %, "
          f"probe fit median {100 * base[2]:.2f} %")
    print(f"\n  {'candidate':62} {'hard rms':>8} {'scatter':>8} {'median':>7} {'price [pJ]':>11}  verdict")
    for candidate in candidates:
        missing = [p.label for p in probes if math.isnan(m.number(rows.get(p.label, {}).get(candidate)))]
        in_validation = all(not math.isnan(m.number(r.get(candidate))) for _, r in validation)
        if len(missing) > len(probes) // 2 or not in_validation:
            print(f"  {candidate:62} not collected (probes without it: {len(missing)}, validation: "
                  f"{'yes' if in_validation else 'no'})")
            continue
        augmented = []
        for p in probes:
            row = rows.get(p.label, {})
            accesses = m.number(row.get("accesses_per_launch"))
            found = value(row, candidate)
            counts = dict(p.counts)
            counts[candidate] = max(found / accesses, 0.0) if accesses > 0 and not math.isnan(found) else 0.0
            augmented.append(m.Probe(p.label, p.energy, counts, p.family, p.groups))
        hard, scatter, median, energies = scores(augmented, kernels_with(candidate), list(m.EVENTS) + [candidate])
        keep = (math.isnan(base[0]) or hard < base[0]) and scatter < base[1]
        print(f"  {candidate:62} {100 * hard:7.1f} % {100 * scatter:7.2f} % {100 * median:6.2f} % {energies[candidate]:11.2f}"
              f"  {'keep (confirm on held-out)' if keep else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
