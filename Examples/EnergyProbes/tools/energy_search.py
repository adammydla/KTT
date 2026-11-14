#!/usr/bin/env python3
"""Which configurations to profile next: the energy model as the objective of a counter-guided search.

The energy model (energy_predictor.py) prices a configuration from its CUPTI counters and its time, so every
configuration it ranks must be profiled once. Filipovic et al. (arXiv 2102.05299) steer KTT's search with counters
instead: from the configurations profiled so far, a model of the tuning parameters predicts the rest. This tool does
that with the energy model as the objective. From the profiled configurations it fits a surrogate on the tuning
parameters (extremely randomised trees, scikit-learn), predicts the energy of every configuration not yet profiled, and
proposes the cheapest ones to profile next; the final choice is still made by the model on measured counters
(energy_predictor.py), only on fewer configurations.

Two surrogates:
  energy    the model's energy of the profiled configurations, regressed on the parameters (log scale) [default]
  counters  every counter and the time regressed on the parameters (log scale), then priced by the model; slower, and
            on the RTX 3090 runs no better (docs/FINDINGS.md, "Configurations that were not profiled")

Parameters are features as log2(value + 1); values that are not numbers are coded by their order of appearance.

    # next 4 configurations to profile, given those profiled so far (KTT JSON with counters, as --tune writes) and the
    # whole space (one "NAME=VALUE NAME=VALUE ..." per line):
    python3 tools/energy_search.py --model models/rtx3090_1200mhz.model.json --ktt RUN_counters.json \\
        --space space.txt --count 4

    # offline check on --validate runs: profile B configurations (half at random, then the predicted cheapest two at a
    # time), choose by the model, and compare the measured energy of the choice with the measured best; against B
    # configurations at random
    python3 tools/energy_search.py --model models/rtx3090_1200mhz.model.json \\
        --evaluate Validation11,Validation12,Validation13,HeldOut1,HeldOut2 [--budgets 12,20,36] [--trials 20]
"""
import argparse
import collections
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_table  # noqa: E402
import energy_predictor  # noqa: E402

KINDS = ("energy", "counters")


def parse(label):
    """"NAME=VALUE NAME=VALUE ..." -> dict."""
    return dict(item.split("=", 1) for item in label.split())


class Features:
    """Tuning parameters as numbers: log2(value + 1), or the order of appearance for values that are not numbers.
    Parameters with one value in the space are left out."""

    def __init__(self, labels):
        configurations = [parse(label) for label in labels]
        names = sorted(set().union(*configurations))
        self.names = [n for n in names if len({c.get(n) for c in configurations}) > 1]
        self.codes = {n: {} for n in self.names}
        for configuration in configurations:
            for name in self.names:
                self._value(name, configuration.get(name, ""))

    def _value(self, name, text):
        try:
            return math.log2(float(text) + 1.0)
        except ValueError:
            return float(self.codes[name].setdefault(text, len(self.codes[name])))

    def __call__(self, labels):
        rows = []
        for label in labels:
            configuration = parse(label)
            rows.append([self._value(n, configuration.get(n, "")) for n in self.names])
        return np.array(rows, dtype=float).reshape(len(labels), len(self.names))


class Surrogate:
    """Predicts the model's energy of configurations from their tuning parameters, fitted on profiled ones."""

    def __init__(self, model, features, kind="energy", trees=100, seed=0):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        self.model, self.features, self.kind, self.trees, self.seed = model, features, kind, trees, seed

    def _forest(self):
        from sklearn.ensemble import ExtraTreesRegressor
        return ExtraTreesRegressor(self.trees, random_state=self.seed)

    def fit(self, labels, counters, kernel_s, energies=None):
        """labels: configurations; counters: their counters (dicts); kernel_s: their times; energies: the model's
        energies if already computed."""
        x = self.features(labels)
        if self.kind == "energy":
            if energies is None:
                energies = [self.model.predict(c, s, strict=False).energy_j for c, s in zip(counters, kernel_s)]
            self.forest = self._forest().fit(x, np.log(np.asarray(energies, dtype=float)))
        else:
            self.names = [n for n in self.model.counter_names() if any(n in c for c in counters)]
            y = np.array([[max(counter_table.number(c.get(n)), 0.0) for n in self.names] for c in counters])
            y = np.column_stack([np.log1p(np.nan_to_num(y)), np.log(np.asarray(kernel_s, dtype=float))])
            self.forest = self._forest().fit(x, y)
        return self

    def predict(self, labels):
        """The predicted energy (J per launch) of each configuration."""
        if not labels:
            return np.array([])
        predicted = self.forest.predict(self.features(labels))
        if self.kind == "energy":
            return np.exp(predicted)
        return np.array([self.model.predict({n: max(math.expm1(v), 0.0) for n, v in zip(self.names, row[:-1])},
                                            math.exp(row[-1]), strict=False).energy_j for row in predicted])


def propose(model, space, profiled, counters, kernel_s, count, kind="energy", seed=0):
    """The `count` configurations of `space` not in `profiled` with the lowest predicted energy, best first, with their
    predictions."""
    done = set(profiled)
    candidates = [label for label in space if label not in done]
    surrogate = Surrogate(model, Features(list(space) + list(profiled)), kind, seed=seed).fit(profiled, counters, kernel_s)
    predicted = surrogate.predict(candidates)
    order = np.argsort(predicted)[:count]
    return [(candidates[i], float(predicted[i])) for i in order]


def guided(surrogate_kind, model, features, labels, counters, seconds, energies, start, budget, step=2, seed=0):
    """Indices profiled by the guided search: `start`, then the predicted cheapest `step` at a time up to `budget`."""
    chosen = list(start)
    while len(chosen) < budget:
        rest = [i for i in range(len(labels)) if i not in chosen]
        surrogate = Surrogate(model, features, surrogate_kind, seed=seed).fit(
            [labels[i] for i in chosen], [counters[i] for i in chosen], [seconds[i] for i in chosen],
            [energies[i] for i in chosen])
        predicted = surrogate.predict([labels[i] for i in rest])
        chosen += [rest[i] for i in np.argsort(predicted)[:min(step, budget - len(chosen))]]
    return chosen


def evaluate(model, runs, budgets, trials, kinds, seed=1):
    """Offline check on --validate runs (configurations pooled over the runs, the first measurement of each kept)."""
    data = collections.defaultdict(dict)
    for run in runs:
        for kernel, rows in energy_predictor.from_validation_csv(run + "_validation.csv").items():
            for label, row, seconds, _, clock in rows:
                if clock and abs(clock / float(model.device.get("sm_clock_mhz") or clock) - 1) > 0.02:
                    continue
                data[kernel].setdefault(label, (row, seconds))
    rng = np.random.default_rng(seed)
    means = collections.defaultdict(list)
    print("Mean (worst) regret of the configuration chosen by the model, measured energy against the measured best:")
    for kernel, entries in data.items():
        labels = list(entries)
        if kernel == "mixed" or len(labels) <= min(budgets):
            continue
        counters = [entries[l][0] for l in labels]
        seconds = [entries[l][1] for l in labels]
        energies = [model.predict(c, s, strict=False).energy_j for c, s in zip(counters, seconds)]
        measured = np.array([counter_table.number(c.get("energy_j")) for c in counters])
        features = Features(labels)

        def regret(chosen):
            return measured[min(chosen, key=lambda i: energies[i])] / np.nanmin(measured) - 1

        cells = []
        for budget in budgets:
            if budget >= len(labels):
                continue
            results = collections.defaultdict(list)
            for _ in range(trials):
                order = rng.permutation(len(labels))
                results["random"].append(regret(order[:budget]))
                for kind in kinds:
                    results[kind].append(regret(guided(kind, model, features, labels, counters, seconds, energies,
                                                       order[:budget // 2], budget)))
            for name, values in results.items():
                means[(budget, name)].append(np.mean(values))
            cells.append(f"B={budget}: " + ", ".join(f"{n} {100 * np.mean(v):.1f} ({100 * np.max(v):.1f})"
                                                     for n, v in results.items()))
        print(f"  {kernel:14} {len(labels):3} configurations | " + " | ".join(cells), flush=True)
    print("Mean over kernels:")
    for budget in budgets:
        print(f"  B={budget}: " + ", ".join(f"{name} {100 * np.mean(means[(budget, name)]):.2f} %"
                                           for name in ("random",) + tuple(kinds) if means[(budget, name)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model file (tools/build_model.py)")
    parser.add_argument("--kind", choices=KINDS, default="energy", help="surrogate (default energy)")
    parser.add_argument("--ktt", help="KTT JSON of the profiled configurations (with counters)")
    parser.add_argument("--csv", help="or a --validate run's CSV")
    parser.add_argument("--kernel", help="the kernel of the file to use (default: the only one)")
    parser.add_argument("--space", help="the whole space: one configuration per line, NAME=VALUE NAME=VALUE ...")
    parser.add_argument("--count", type=int, default=4, help="configurations to propose (default 4)")
    parser.add_argument("--evaluate", help="--validate run prefixes, comma-separated: the offline check")
    parser.add_argument("--budgets", default="12,20,36")
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    model = energy_predictor.Model.load(args.model)

    if args.evaluate:
        kinds = (args.kind,) if args.kind != "energy" else ("energy", "counters")
        evaluate(model, args.evaluate.split(","), [int(b) for b in args.budgets.split(",")], args.trials, kinds)
        return 0

    if not (args.ktt or args.csv) or not args.space:
        parser.error("--propose needs --ktt or --csv and --space (or use --evaluate)")
    kernels = energy_predictor.from_ktt_json(args.ktt) if args.ktt else energy_predictor.from_validation_csv(args.csv)
    if args.kernel:
        rows = kernels[args.kernel]
    elif len(kernels) == 1:
        rows = next(iter(kernels.values()))
    else:
        parser.error(f"the file has kernels {', '.join(kernels)}: choose one with --kernel")
    with open(args.space) as file:
        space = [line.strip() for line in file if line.strip() and not line.startswith("#")]
    # Labels as the space writes them (KTT orders parameters as declared; compare as sets of pairs).
    canonical = {frozenset(parse(label).items()): label for label in space}
    profiled, counters, seconds = [], [], []
    for label, row, kernel_s, _, _ in rows:
        key = frozenset(parse(label).items())
        profiled.append(canonical.get(key, label))
        counters.append(row)
        seconds.append(kernel_s)
    if len(profiled) < 4:
        parser.error("profile at least 4 configurations first")
    for label, energy in propose(model, space, profiled, counters, seconds, args.count, args.kind):
        print(f"{energy:.4g} J  {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
