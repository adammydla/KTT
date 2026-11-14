#!/usr/bin/env python3
"""How much the kernels' predictions depend on which probes are in the fit (NEXT_STEPS_PLAN P4).

The event energies come from a fit over the probes. Some events occur in fixed proportions in every probe, so the fit can
trade one for the other, and the kernels have them in other proportions: a prediction then depends on the probe set,
not only on the physics. This prints, for a set of power, counter and validation runs:

  (a) every price's coefficient of variation over resamples of the probes, drawn within families (each family keeps
      its size), so that no event loses all its probes;
  (b) per validation kernel, the sd over those fits of the kernel's mean prediction (a bias that moves with the probe
      set) and of the differences between its configurations (what the ranking sees);
  (c) the same scores with each probe family left out.

A model change is judged by these too (the rule of NEXT_STEPS_PLAN 8.5 and 11.5): the kernels' predictions should
depend less on the probe set, not only fit better. Reference (Preset3 + ProfileModel1, Validation11 + HeldOut1, plain
resampling, 2026-11-14): kernel means 1.9-6.2 %, within-kernel 2.2-5.3 %. The shipped model since 2026-11-14 adds
--counters ProfileNew2 --full-rate MisalignedWrite2 (five calibration runs: means 3.7 -> 3.6 %, within 2.9 -> 2.6 %).

    python3 tools/fit_stability.py --power Preset3 --counters ProfileModel1 \\
        --validation Validation11,Validation12,Validation13,HeldOut1,HeldOut2 [--resamples 300] [--weighted]
"""
import argparse
import collections
import contextlib
import io
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import energy_model as m  # noqa: E402


def fit_probes(power, counters, full_rate=(), sweep=()):
    """The probes the model's fit uses (as energy_model.py main selects them)."""
    with contextlib.redirect_stdout(io.StringIO()):
        probes, table = m.probe_table(power, counters, full_rate, sweep)
    if any(p.label == "compute ffma3" for p in probes):
        probes = [p for p in probes if p.label not in m.TWO_REGISTER_FFMA]
    return [p for p in probes if p.label not in m.CHECK_ONLY], table


class Kernels:
    """The validation configurations as event counts, measured dynamic energies and kernel names."""

    def __init__(self, runs, table):
        rows = []
        self.run = []
        for run in runs:
            with contextlib.redirect_stdout(io.StringIO()):
                found = [r for r in m.load_validation(run, table) if r["kernel"] != m.CONTROL]
            rows += found
            self.run += [os.path.basename(run)] * len(found)
        counts = [m.add_busy(m.events(r), m.number(r["kernel_s"]) * 1e9) for r in rows]
        self.events = np.array([[c[name] for name in m.EVENTS] for c in counts])
        self.measured = np.array([float(r["dynamic_energy_j"]) for r in rows])
        self.kernel = np.array([r["kernel"] for r in rows])
        self.configuration = [r["configuration"] for r in rows]
        self.run = np.array(self.run)
        self.names = list(dict.fromkeys(self.kernel))

    def predict(self, energies):
        return self.events @ np.array([energies[name] for name in m.EVENTS]) * 1e-12

    def score(self, energies):
        """Scatter (rms of the within-kernel sd of the error over kernel runs), the sd of the kernels' mean bias, and
        each kernel's mean bias."""
        errors = self.predict(energies) / self.measured - 1
        variances, bias = [], collections.defaultdict(list)
        for run in dict.fromkeys(self.run):
            for kernel in self.names:
                selected = (self.run == run) & (self.kernel == kernel)
                if selected.sum() >= 2:
                    variances.append(errors[selected].var(ddof=1))
                    bias[kernel].append(errors[selected].mean())
        means = {k: float(np.mean(v)) for k, v in bias.items()}
        return math.sqrt(np.mean(variances)), float(np.std(list(means.values()), ddof=1)), means


def resample(probes, rng):
    """A resample of the probes drawn within families (each family keeps its size)."""
    families = collections.defaultdict(list)
    for p in probes:
        families[p.family].append(p)
    drawn = []
    for members in families.values():
        drawn += [members[i] for i in rng.integers(0, len(members), len(members))]
    return drawn


def stability(probes, kernels, resamples=300, weighted=False, seed=0):
    """(prices' CV per event, per-kernel (bias sd, within-kernel sd), sd between kernels of the bias shift)."""
    reference, _, _ = m.fit(probes, m.EVENTS, weighted)
    base = kernels.predict(reference)
    rng = np.random.default_rng(seed)
    prices, ratios = [], []
    for _ in range(resamples):
        energies, _, _ = m.fit(resample(probes, rng), m.EVENTS, weighted)
        prices.append([energies[name] for name in m.EVENTS])
        ratios.append(kernels.predict(energies) / base)
    prices, ratios = np.array(prices), np.array(ratios)
    cv = {name: float(prices[:, i].std() / prices[:, i].mean()) for i, name in enumerate(m.EVENTS)
          if reference[name] > 0 and prices[:, i].mean() > 0}
    per_kernel = {}
    shifts = []
    for kernel in kernels.names:
        selected = kernels.kernel == kernel
        shift = ratios[:, selected].mean(axis=1)
        within = (ratios[:, selected] / shift[:, None]).std(axis=0).mean()
        per_kernel[kernel] = (float(shift.std()), float(within))
        shifts.append(shift)
    between = float(np.array(shifts).std(axis=0, ddof=1).mean())
    return reference, cv, per_kernel, between


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power", required=True, help="probe power runs, comma-separated (as for energy_model.py)")
    parser.add_argument("--counters", required=True, help="profiled probe runs, comma-separated")
    parser.add_argument("--full-rate", help="runs at one duty, as in energy_model.py")
    parser.add_argument("--sm-sweep", help="runs whose dependent-load rows on fewer SMs enter as probes, as in "
                        "energy_model.py")
    parser.add_argument("--validation", required=True, help="validation runs, comma-separated")
    parser.add_argument("--resamples", type=int, default=300)
    parser.add_argument("--weighted", action="store_true", help="as in energy_model.py")
    parser.add_argument("--no-families", action="store_true", help="skip (c), the families left out")
    parser.add_argument("--no-busy", action="store_true", help="as in energy_model.py")
    args = parser.parse_args()
    m.configure(busy=not args.no_busy)
    split = lambda text: text.split(",") if text else []  # noqa: E731
    probes, table = fit_probes(split(args.power), split(args.counters), split(args.full_rate), split(args.sm_sweep))
    kernels = Kernels(split(args.validation), table)
    reference, cv, per_kernel, between = stability(probes, kernels, args.resamples, args.weighted)
    scatter, bias_sd, bias = kernels.score(reference)
    predicted = kernels.predict(reference)
    share = (kernels.events * np.array([reference[n] for n in m.EVENTS]) * 1e-12).sum(axis=0) / predicted.sum()
    print(f"{len(probes)} probes in {len(set(p.family for p in probes))} families, {len(kernels.measured)} validation "
          f"configurations of {len(kernels.names)} kernels; scatter {100 * scatter:.2f} %, bias sd between kernels "
          f"{100 * bias_sd:.2f} %{', weighted' if args.weighted else ''}")
    print(f"\n(a) prices over {args.resamples} resamples within families: price [pJ], CV, share of the kernels' "
          "predicted dynamic energy")
    for i, name in enumerate(m.EVENTS):
        if name in cv:
            print(f"  {name:34} {reference[name]:9.2f}  CV {100 * cv[name]:5.1f} %  share {100 * share[i]:5.1f} %")
    print("\n(b) per kernel: sd of its mean prediction (bias), mean sd of its configurations' differences")
    for kernel, (shift, within) in per_kernel.items():
        print(f"  {kernel:14} bias {100 * shift:4.1f} %   within {100 * within:4.1f} %")
    shifts = [v[0] for v in per_kernel.values()]
    withins = [v[1] for v in per_kernel.values()]
    print(f"  stability: kernel means {100 * min(shifts):.1f}-{100 * max(shifts):.1f} % (mean {100 * np.mean(shifts):.1f}), "
          f"within-kernel {100 * min(withins):.1f}-{100 * max(withins):.1f} % (mean {100 * np.mean(withins):.1f}), "
          f"between kernels {100 * between:.1f} %")
    if args.no_families:
        return 0
    print("\n(c) one family left out: scatter, bias sd, the kernel whose bias moves most, events that lose their price")
    families = collections.Counter(p.family for p in probes)
    for family, count in sorted(families.items(), key=lambda item: -item[1]):
        energies, _, _ = m.fit([p for p in probes if p.family != family], m.EVENTS, args.weighted)
        s, b, means = kernels.score(energies)
        worst = max(means, key=lambda k: abs(means[k] - bias[k]))
        lost = [n for n in m.EVENTS if reference[n] > 0 and energies[n] == 0]
        print(f"  {family:28} ({count:3d}): {100 * s:5.2f} %  {100 * b:5.2f} %  {worst} {100 * (means[worst] - bias[worst]):+.1f}"
              f"  {', '.join(lost)[:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
