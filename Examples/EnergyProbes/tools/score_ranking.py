#!/usr/bin/env python3
"""How well the energy model ranks the configurations of a kernel, over one or more validation runs.

For picking the configuration that uses the least energy, what matters is the scatter of the error between the
configurations of one kernel, not its bias. Per kernel and run this prints the standard deviation (scatter) and mean
(bias) of the relative error of the dynamic energy, the mean absolute error of the total energy, and the regret (how much
more energy the configuration the model ranks best uses than the measured best). Over all kernels it counts the pairs of
configurations of one kernel that the model orders like the measurement, by bins of their measured difference, next to
a ranking by kernel time alone (constant power), the reference to beat. The control kernel (--validate mixed) is shown
on its own and stays out of the totals.

    python3 tools/score_ranking.py --power Full --counters ProfileFull --validation Validation1,Validation2
    python3 tools/score_ranking.py ... --fma-halves      # the same with the options of energy_model.py
"""

import argparse
import contextlib
import io
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import energy_model as m  # noqa: E402

BINS = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, float("inf"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power", required=True, help="probe power runs, comma-separated (as for energy_model.py)")
    parser.add_argument("--counters", required=True, help="profiled probe runs, comma-separated")
    parser.add_argument("--validation", required=True, help="validation runs, comma-separated")
    parser.add_argument("--weighted", action="store_true", help="as in energy_model.py")
    parser.add_argument("--fma-halves", action="store_true", help="as in energy_model.py")
    parser.add_argument("--no-busy", action="store_true", help="as in energy_model.py")
    parser.add_argument("--busy-scale", type=float, default=m.BUSY_SCALE_GBS, help="as in energy_model.py")
    parser.add_argument("--kernels", help="only these kernels, comma-separated (e.g. the held-out ones)")
    arguments = parser.parse_args()
    m.configure(arguments.fma_halves, not arguments.no_busy, arguments.busy_scale)
    selected = set(arguments.kernels.split(",")) if arguments.kernels else None

    with contextlib.redirect_stdout(io.StringIO()):
        probes, table = m.probe_table(arguments.power.split(","), arguments.counters.split(","))
    if any(p.label == "compute ffma3" for p in probes):
        probes = [p for p in probes if p.label not in m.TWO_REGISTER_FFMA]
    probes = [p for p in probes if p.label not in m.CHECK_ONLY]
    energies, _, unused = m.fit(probes, m.EVENTS, arguments.weighted)
    simple, _, _ = m.fit(probes, m.SIMPLE, arguments.weighted)

    validations = {}
    for validation in arguments.validation.split(","):
        with contextlib.redirect_stdout(io.StringIO()):
            validations[validation] = m.load_validation(validation, table)
    # The same guards as energy_model.py: the probes and the validation runs must count the same events.
    m.check_counter_groups({"probes (--counters)": {p.groups for p in probes},
                            **{v: {m.counter_groups(row) for row in rows} for v, rows in validations.items()}})
    m.check_required_counters({"probes (--counters)": m.counter_rows(arguments.counters.split(",")), **validations})
    for line in m.unpriced_events([row for rows in validations.values() for row in rows], unused):
        print(line)

    lines, control = [], []
    pairs = {b: [0, 0, 0] for b in BINS}  # model right, time-only right, pairs
    for validation, rows in validations.items():
        results = m.predict(rows, energies, simple)
        for kernel in dict.fromkeys(r["kernel"] for r in results):
            if selected and kernel not in selected and kernel != m.CONTROL:
                continue
            group = [r for r in results if r["kernel"] == kernel]
            measured = np.array([r["measured_j"] for r in group])
            predicted = np.array([r["predicted_j"] for r in group])
            constant = np.array([r["constant_power_j"] for r in group])
            error = np.array([r["predicted_dynamic_j"] / r["measured_dynamic_j"] - 1 for r in group])
            line = (validation, kernel, len(group), error.std(), error.mean(), np.abs(predicted / measured - 1).mean(),
                    measured[np.argmin(predicted)] / measured.min() - 1, measured[np.argmin(constant)] / measured.min() - 1)
            if kernel == m.CONTROL:
                control.append(line)
                continue
            lines.append(line)
            for i, j in itertools.combinations(range(len(group)), 2):
                difference = max(measured[i], measured[j]) / min(measured[i], measured[j]) - 1
                b = next(b for b in BINS if b[0] <= difference < b[1])
                pairs[b][0] += (measured[i] < measured[j]) == (predicted[i] < predicted[j])
                pairs[b][1] += (measured[i] < measured[j]) == (constant[i] < constant[j])
                pairs[b][2] += 1

    print(f"Model: power {arguments.power}, counters {arguments.counters}, {len(probes)} probes"
          + (", --weighted" if arguments.weighted else "") + (", --fma-halves" if arguments.fma_halves else "")
          + (", --no-busy" if arguments.no_busy else "")
          + (f", --busy-scale {arguments.busy_scale:g}" if arguments.busy_scale != parser.get_default("busy_scale")
             else ""))
    print(f"  {'run':14} {'kernel':10} {'n':>3} {'scatter':>8} {'bias':>7} {'total MAPE':>11} {'regret':>7} "
          f"{'regret, time only':>18}")
    for run, kernel, n, sd, bias, mape, regret, regret_time in lines + control:
        print(f"  {run:14} {kernel:10} {n:3d} {100 * sd:7.1f}% {100 * bias:+6.1f}% {100 * mape:10.1f}% "
              f"{100 * regret:6.1f}% {100 * regret_time:17.1f}%" + ("  (control)" if kernel == m.CONTROL else ""))
    if not lines:
        return
    print(f"  mean over the example kernels: scatter {100 * np.mean([l[3] for l in lines]):.1f} %, bias "
          f"{100 * np.mean([l[4] for l in lines]):+.1f} %, total MAPE {100 * np.mean([l[5] for l in lines]):.1f} %, "
          f"max regret {100 * max(l[6] for l in lines):.1f} % (time only {100 * max(l[7] for l in lines):.1f} %)")
    print("  pairs of configurations ordered correctly, model / time only, by measured difference:")
    for (low, high), (model, time_only, n) in pairs.items():
        label = f"{100 * low:.0f}-{100 * high:.0f} %" if high != float("inf") else f">{100 * low:.0f} %"
        if n:
            print(f"    {label:9} {100 * model / n:5.1f} % / {100 * time_only / n:5.1f} %   ({n} pairs)")


if __name__ == "__main__":
    main()
