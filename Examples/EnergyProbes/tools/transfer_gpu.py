#!/usr/bin/env python3
"""Transfer the probe table of a reference GPU to a new GPU from a subset of probes measured there.

A full probe run takes hours; the counter run (--model --profile, about 12 min on an RTX 3090) does not depend on power.
Following Wattchmen (Tran et al., arXiv 2603.26435: per-instruction energies of a new system from a random 10 % of them
and an affine fit to the old system, 13 % MAPE), this measures the energies of a few probes on the new GPU, maps the
reference GPU's table onto it, and predicts the rest:
- global: E_new = a + b x E_ref over all probes of the subset (least squares on relative errors);
- per kind: E_new = k x E_ref with one factor per kind of probe (compute, shared and L1, L2, DRAM), from the subset's
  probes of that kind, and the global map for a kind the subset has none of.

    # leave-out test: how well do random subsets predict the rest, by subset size (no validation runs needed)
    python3 tools/transfer_gpu.py --ref-power R/Full,R/Ffma3,R/Writes,R/Instr --ref-counters R/ProfileFull4 \
        --power Full --counters ProfileFull --leave-out
    # the same, and the energy model of each transferred table scored on the new GPU's validation runs
    ... --leave-out --validation Validation1,Validation2
    # the predicted table for a chosen subset (labels as energy_model.py prints them), as CSV
    ... --subset "L2 ld w16,DRAM ld w16,compute ffma3,shared ld.shared w16" --csv transferred.csv

The new GPU's --power run must contain the subset (for the leave-out test, all probes, of which the rest serve as the
truth); its --counters run gives every probe's counts.
"""

import argparse
import contextlib
import csv
import io
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import energy_model as m  # noqa: E402

FRACTIONS = (0.05, 0.1, 0.2, 0.3, 0.5)


def kind(label):
    """The kind of a probe for the per-kind map: compute, shared and L1, L2, DRAM (by the level in its label)."""
    level = label.split()[0]
    if level == "compute" or level == "none":
        return "compute"
    if level in ("shared", "L1") or level.startswith("const"):
        return "shared and L1"
    return "DRAM" if level == "DRAM" else "L2"


def probes_of(power, counters):
    with contextlib.redirect_stdout(io.StringIO()):
        probes, table = m.probe_table(power.split(","), counters.split(","))
    return {p.label: p for p in probes}, table


def affine(x, y):
    """a, b of y = a + b x by least squares on relative errors (weights 1 / y)."""
    w = 1 / np.asarray(y)
    a = np.c_[np.ones_like(x), x] * w[:, None]
    (intercept, slope), *_ = np.linalg.lstsq(a, np.asarray(y) * w, rcond=None)
    return intercept, slope


def transfer(reference, measured, subset):
    """Predicted energies of every probe of `reference` from the measured energies of `subset`: (global, per kind)."""
    x = np.array([reference[k] for k in subset])
    y = np.array([measured[k] for k in subset])
    a, b = affine(x, y)
    global_map = {k: max(a + b * e, 1e-6) for k, e in reference.items()}
    factors = {}
    for group in {kind(k) for k in subset}:
        keys = [k for k in subset if kind(k) == group]
        factors[group] = float(np.exp(np.mean([np.log(measured[k] / reference[k]) for k in keys])))
    per_kind = {k: e * factors[kind(k)] if kind(k) in factors else global_map[k] for k, e in reference.items()}
    for k in subset:  # measured probes keep their own energies
        global_map[k] = per_kind[k] = measured[k]
    return global_map, per_kind


def score(probes, energies_by_label, validations):
    """Mean within-kernel scatter, bias and total MAPE and the max regret (as score_ranking.py) of the model fitted to
    `probes` with the given energies."""
    fitted = [m.Probe(p.label, energies_by_label[p.label], p.counts, p.family, p.groups) for p in probes]
    if any(p.label == "compute ffma3" for p in fitted):
        fitted = [p for p in fitted if p.label not in m.TWO_REGISTER_FFMA]
    fitted = [p for p in fitted if p.label not in m.CHECK_ONLY]
    energies, _, _ = m.fit(fitted, m.EVENTS, False)
    simple, _, _ = m.fit(fitted, m.SIMPLE, False)
    scatter, bias, mape, regret = [], [], [], []
    for rows in validations.values():
        results = m.predict(rows, energies, simple)
        for kernel in dict.fromkeys(r["kernel"] for r in results):
            if kernel == m.CONTROL:
                continue
            group = [r for r in results if r["kernel"] == kernel]
            measured = np.array([r["measured_j"] for r in group])
            predicted = np.array([r["predicted_j"] for r in group])
            error = np.array([r["predicted_dynamic_j"] / r["measured_dynamic_j"] - 1 for r in group])
            scatter.append(error.std()); bias.append(error.mean())
            mape.append(np.abs(predicted / measured - 1).mean())
            regret.append(measured[np.argmin(predicted)] / measured.min() - 1)
    return 100 * np.mean(scatter), 100 * np.mean(bias), 100 * np.mean(mape), 100 * max(regret)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-power", required=True)
    parser.add_argument("--ref-counters", required=True)
    parser.add_argument("--power", required=True, help="the new GPU's power runs (the subset, or all probes)")
    parser.add_argument("--counters", required=True, help="the new GPU's counter runs")
    parser.add_argument("--leave-out", action="store_true", help="random subsets of each size predict the rest")
    parser.add_argument("--draws", type=int, default=200, help="random subsets per size (default 200)")
    parser.add_argument("--validation", help="the new GPU's validation runs, to score the transferred models")
    parser.add_argument("--subset", help="labels of the measured probes, comma-separated (without --leave-out)")
    parser.add_argument("--csv", help="write the predicted table (with --subset)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-busy", action="store_true", help="without the DRAM busy event (energy_model.py --no-busy); "
                        "use it when the reference GPU's probe times would be paired with the new GPU's counts")
    arguments = parser.parse_args()
    m.configure(busy=not arguments.no_busy)

    reference_probes, _ = probes_of(arguments.ref_power, arguments.ref_counters)
    new_probes, table = probes_of(arguments.power, arguments.counters)
    common = sorted(set(reference_probes) & set(new_probes))
    reference = {k: reference_probes[k].energy for k in common}
    measured = {k: new_probes[k].energy for k in common}
    print(f"{len(common)} probes in both tables ({len(reference_probes)} reference, {len(new_probes)} new)")
    ratios = np.array([measured[k] / reference[k] for k in common])
    print(f"new / reference: median {np.median(ratios):.3f}, quartiles {np.percentile(ratios, 25):.3f}-"
          f"{np.percentile(ratios, 75):.3f}; by kind: " + ", ".join(
              f"{g} {np.median([measured[k] / reference[k] for k in common if kind(k) == g]):.2f}"
              for g in sorted({kind(k) for k in common})))

    validations = {}
    if arguments.validation:
        for v in arguments.validation.split(","):
            with contextlib.redirect_stdout(io.StringIO()):
                validations[v] = m.load_validation(v, table)

    if arguments.subset:
        subset = [s.strip() for s in arguments.subset.split(",")]
        missing = [s for s in subset if s not in common]
        if missing:
            raise SystemExit(f"not in both tables: {missing}")
        global_map, per_kind = transfer(reference, measured, subset)
        rest = [k for k in common if k not in subset]
        for name, predicted in (("global", global_map), ("per kind", per_kind)):
            errors = np.array([predicted[k] / measured[k] - 1 for k in rest])
            print(f"{name}: the {len(rest)} other probes within a median {100 * np.median(np.abs(errors)):.1f} %, "
                  f"90 % within {100 * np.percentile(np.abs(errors), 90):.1f} %")
        if arguments.csv:
            with open(arguments.csv, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["probe", "kind", "reference_pj", "measured_pj", "global_pj", "per_kind_pj", "in_subset"])
                for k in sorted(reference_probes):
                    writer.writerow([k, kind(k), reference_probes[k].energy, measured.get(k, ""), global_map.get(k, ""),
                                     per_kind.get(k, ""), int(k in subset)])
            print(f"wrote {arguments.csv}")
        return

    if not arguments.leave_out:
        raise SystemExit("give --leave-out or --subset")
    rng = random.Random(arguments.seed)
    full = None
    if validations:
        full = score([new_probes[k] for k in common], measured, validations)
        same = score([new_probes[k] for k in common], reference, validations)
        print(f"\nModel scores on the new GPU's validation runs (scatter, bias, total MAPE, max regret):\n"
              f"  all probes measured on the new GPU:   {full[0]:5.1f} % {full[1]:+6.1f} % {full[2]:5.1f} % {full[3]:5.1f} %\n"
              f"  the reference table unchanged:        {same[0]:5.1f} % {same[1]:+6.1f} % {same[2]:5.1f} % {same[3]:5.1f} %")
    print(f"\nLeave-out: {arguments.draws} random subsets per size predict the other probes (median |error| over all "
          f"predicted probes, then the median and 90th percentile over the draws)")
    print(f"  {'subset':>12}  {'no transfer':>11}  {'global affine':>14}  {'per kind':>12}" +
          ("   model with the per-kind table (mean of 20 draws)" if validations else ""))
    for fraction in FRACTIONS:
        size = max(2, round(fraction * len(common)))
        summary = {"none": [], "global": [], "per kind": []}
        scores = []
        for draw in range(arguments.draws):
            subset = rng.sample(common, size)
            rest = [k for k in common if k not in subset]
            global_map, per_kind = transfer(reference, measured, subset)
            for name, predicted in (("none", reference), ("global", global_map), ("per kind", per_kind)):
                summary[name].append(np.median([abs(predicted[k] / measured[k] - 1) for k in rest]))
            if validations and draw < 20:
                scores.append(score([new_probes[k] for k in common], per_kind, validations))
        cells = [f"{100 * np.median(v):5.1f} / {100 * np.percentile(v, 90):4.1f} %" for v in summary.values()]
        line = f"  {size:4d} ({100 * fraction:2.0f} %)  " + "  ".join(f"{c:>12}" for c in cells)
        if scores:
            s = np.mean(scores, axis=0)
            line += f"   {s[0]:5.1f} % {s[1]:+6.1f} % {s[2]:5.1f} % {s[3]:5.1f} %"
        print(line)


if __name__ == "__main__":
    main()
