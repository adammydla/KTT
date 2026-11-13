#!/usr/bin/env python3
"""Energy of a kernel from its CUPTI counters and its time, with a model file. No power measurement, no SciPy.

This is the part a tuner uses. The probes and energy_model.py are the calibration: tools/build_model.py turns their runs
into one model file (GPU, clock, energy per event, baseline, uncertainty, provenance). With that file:

    import energy_predictor
    model = energy_predictor.Model.load("rtx3090.model.json")
    p = model.predict(counters, kernel_s=0.0123)      # counters: {CUPTI name: value per launch}
    p.energy_j, p.sd_j, p.dynamic_j, p.baseline_j, p.warnings
    best = energy_predictor.recommend(predictions, objective="energy")
    best.index, best.contenders, best.confident

or from the command line, on a file that KTT wrote (the JSON output of a tuning run with profiling) or on a validation
run's CSV:

    python3 tools/energy_predictor.py --model rtx3090.model.json --ktt TuningResults.json
    python3 tools/energy_predictor.py --model rtx3090.model.json --csv Validation13_validation.csv --objective edp

What a prediction is: the baseline power (the GPU awake at the model's clock, at the given temperature) over the
kernel's time, plus the dynamic energy, the sum of the event counts times the model's energies per event.

What the uncertainty is: the model's dynamic energy is off by a factor that differs between the configurations of one
kernel with a standard deviation of about 6 % (the model file has the measured value and how it was measured). sd_j is
that fraction of the dynamic energy, plus the baseline's uncertainty over the kernel's time. It describes how well
configurations of one kernel can be told apart, which is what tuning needs; the absolute error has a bias on top (also in
the file). Two configurations whose predicted energies differ by less than about two standard deviations of their
difference are not told apart reliably: recommend() returns them as contenders. The model's order among contenders is
still better than chance and better than ordering by time (see the README), so the first choice stands; measure the
contenders (NVML, a few seconds each) when the difference matters.

A model is valid for the GPU model and the SM clock it was calibrated at. predict() warns when the clock of the
measurement differs; tools/check_model.py tests a model on a GPU in a few minutes.
"""

import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_table  # noqa: E402
import energy_events  # noqa: E402

FORMAT = "energyprobes-model"
VERSION = 1

# A clock this far from the model's (relative) gets a warning: energies per event change with clock and voltage.
CLOCK_TOLERANCE = 0.05
# An event without a price that makes up at least this share of a kernel's warp instructions gets a warning.
UNPRICED_SHARE = 0.01

OBJECTIVES = ("energy", "edp", "ed2p", "time")


class ModelError(Exception):
    """A model file that cannot be used, or counters that the model cannot price."""


Prediction = collections.namedtuple(
    "Prediction", "energy_j sd_j dynamic_j baseline_j kernel_s power_w events shares warnings label")
Prediction.__doc__ = """One configuration: total energy per launch (J) and its standard deviation, its dynamic and
baseline parts, the kernel's time (s), the mean power (W), the event counts, each event's share of the dynamic energy,
warnings (text), and the caller's label."""

Recommendation = collections.namedtuple("Recommendation", "index contenders confident objective values sds front")
Recommendation.__doc__ = """index: the configuration with the lowest predicted objective among the admissible ones.
contenders: the others that cannot be told apart from it (indices, best first). confident: no contenders. values, sds:
the objective and its standard deviation per configuration (NaN where not admissible). front: the indices on the
energy-time Pareto front, by time."""


class Model:
    def __init__(self, data, path=None):
        if data.get("format") != FORMAT:
            raise ModelError(f"{path or 'model'}: not an EnergyProbes model file (format {data.get('format')!r})")
        if int(data.get("version", 0)) > VERSION:
            raise ModelError(f"{path or 'model'}: file version {data['version']} is newer than this tool ({VERSION})")
        self.data = data
        self.path = path
        self.device = data["device"]
        self.options = data["options"]
        self.pj = {name: float(entry["pj"]) for name, entry in data["events"].items()}
        unknown = sorted(set(self.pj) - set(energy_events.EVENTS))
        if unknown:
            raise ModelError(f"{path or 'model'}: events this tool does not know: {', '.join(unknown)}")
        self.unpriced = {name for name, entry in data["events"].items() if entry.get("status") == "no probe"}
        self.baseline = data["baseline"]
        self.uncertainty = data["uncertainty"]
        self.required = list(data["counters"]["required"])
        self.groups = list(data["counters"].get("groups", []))
        # The GPU's power limit in W, if known (NVML reports it; --power-limit). A configuration predicted above it will
        # not run at the model's clock: the GPU lowers its clock to hold the limit, and the energy is then about the limit
        # times the time. predict() takes that value and adds a warning.
        self.power_limit_w = None

    @classmethod
    def load(cls, path):
        with open(path) as file:
            return cls(json.load(file), path)

    def baseline_w(self, temperature_c=None):
        """The power of the GPU awake at the model's clock without work, at a GPU temperature (default: the mean
        temperature of the calibration)."""
        base = float(self.baseline["mean_w"])
        if temperature_c is not None and not math.isnan(temperature_c):
            base += float(self.baseline["slope_w_per_c"]) * (temperature_c - float(self.baseline["mean_temperature_c"]))
        return base

    def missing_counters(self, counters):
        """The counters the model needs that `counters` has no value of."""
        return [name for name in self.required if math.isnan(counter_table.value_of(counters, name))]

    def events(self, counters, kernel_s):
        counts = energy_events.events(counters, fma_halves=bool(self.options.get("fma_halves")))
        return energy_events.add_busy(counts, kernel_s * 1e9, busy=bool(self.options.get("busy", True)),
                                      scale_gbs=float(self.options.get("busy_scale_gbs", energy_events.BUSY_SCALE_GBS)))

    def predict(self, counters, kernel_s, temperature_c=None, sm_clock_mhz=None, label=None, strict=True):
        """Energy of one launch. `counters`: CUPTI counter values summed over the launch, by name (a dict; values may be
        text). `kernel_s`: the launch's time on the GPU. `temperature_c`: GPU temperature if known. `sm_clock_mhz`: the
        SM clock during the launch if known, to check it against the model's. With `strict`, counters the model needs
        and does not find raise ModelError (they would count as 0 and the kernel would look cheaper than it is);
        without, they become a warning."""
        if not kernel_s > 0:
            raise ModelError("kernel_s must be positive")
        warnings = []
        missing = self.missing_counters(counters)
        if missing:
            text = "counters the model needs are missing: " + ", ".join(missing)
            if strict:
                raise ModelError(text + " (collect the counters of tools/counters.json, or predict with strict=False)")
            warnings.append(text + "; their events count as 0")
        for group in self.groups:
            column = counter_table.GROUPS.get(group)
            if column and math.isnan(counter_table.number(counters.get(column))):
                text = f"the model was fitted with the {group}, which these counters do not have"
                if strict:
                    raise ModelError(text)
                warnings.append(text)

        counts = self.events(counters, kernel_s)
        parts = {name: counts[name] * self.pj.get(name, 0.0) * 1e-12 for name in energy_events.EVENTS}
        dynamic = sum(parts.values())
        baseline = self.baseline_w(temperature_c) * kernel_s

        instructions = energy_events.counter(counters, "smsp__inst_executed.sum")
        for name in sorted(self.unpriced):
            if counts.get(name, 0.0) > 0 and name.startswith("inst") and instructions > 0 \
                    and counts[name] / instructions >= UNPRICED_SHARE:
                warnings.append(f"{name}: {100 * counts[name] / instructions:.1f} % of the warp instructions, but no probe "
                                "priced it (counts as free)")
            elif counts.get(name, 0.0) > 0 and not name.startswith("inst"):
                warnings.append(f"{name}: {counts[name]:.3g} per launch, but no probe priced it (counts as free)")
        clock = float(self.device.get("sm_clock_mhz") or math.nan)
        if sm_clock_mhz and not math.isnan(clock) and abs(sm_clock_mhz / clock - 1) > CLOCK_TOLERANCE:
            warnings.append(f"SM clock {sm_clock_mhz:.0f} MHz, model calibrated at {clock:.0f} MHz: energies per event "
                            "change with the clock, the prediction is only good for ranking")

        relative = float(self.uncertainty["dynamic_relative_sd"])
        baseline_sd = float(self.uncertainty.get("baseline_sd_w", 0.0)) * kernel_s
        if temperature_c is None or (isinstance(temperature_c, float) and math.isnan(temperature_c)):
            baseline_sd = math.hypot(baseline_sd, float(self.uncertainty.get("baseline_unknown_temperature_sd_w", 0.0))
                                     * kernel_s)
        shares = {name: part / dynamic if dynamic > 0 else 0.0 for name, part in parts.items()}
        sd = math.hypot(relative * dynamic, baseline_sd)
        if self.power_limit_w and baseline + dynamic > self.power_limit_w * kernel_s:
            warnings.append(f"predicted power {(baseline + dynamic) / kernel_s:.0f} W is above the GPU's limit of "
                            f"{self.power_limit_w:.0f} W: the GPU will lower its clock; energy taken as the limit times "
                            "the time")
            dynamic = max(self.power_limit_w * kernel_s - baseline, 0.0)
        return Prediction(energy_j=baseline + dynamic, sd_j=sd, dynamic_j=dynamic,
                          baseline_j=baseline, kernel_s=kernel_s, power_w=(baseline + dynamic) / kernel_s, events=counts,
                          shares=shares, warnings=warnings, label=label)

    def pair_sd(self, a, b):
        """Standard deviation of energy(a) - energy(b) in J. The dynamic errors of two configurations are taken as
        independent (that is how the model's relative sd was measured: within a kernel); a baseline error is common to
        both and only acts on the difference of their times."""
        relative = float(self.uncertainty["dynamic_relative_sd"])
        baseline = math.hypot(float(self.uncertainty.get("baseline_sd_w", 0.0)),
                              float(self.uncertainty.get("baseline_unknown_temperature_sd_w", 0.0)))
        return math.sqrt((relative * a.dynamic_j) ** 2 + (relative * b.dynamic_j) ** 2
                         + (baseline * (a.kernel_s - b.kernel_s)) ** 2)


def objective_value(prediction, objective):
    """The value to minimise: energy (J per launch), edp (J s), ed2p (J s^2) or time (s)."""
    if objective == "energy":
        return prediction.energy_j
    if objective == "edp":
        return prediction.energy_j * prediction.kernel_s
    if objective == "ed2p":
        return prediction.energy_j * prediction.kernel_s ** 2
    if objective == "time":
        return prediction.kernel_s
    raise ValueError(f"objective must be one of {', '.join(OBJECTIVES)}")


def pareto_front(predictions):
    """Indices of the configurations that no other one beats in both time and predicted energy, by time."""
    order = sorted(range(len(predictions)), key=lambda i: (predictions[i].kernel_s, predictions[i].energy_j))
    front, best = [], math.inf
    for i in order:
        if predictions[i].energy_j < best:
            front.append(i)
            best = predictions[i].energy_j
    return front


def recommend(model, predictions, objective="energy", z=2.0, max_slowdown=None, max_energy_increase=None):
    """The configuration to use, and how sure that is.

    objective: what to minimise (OBJECTIVES). max_slowdown: only configurations at most this fraction slower than the
    fastest are admissible (0.05 = within 5 % of the best time; "the least energy at nearly the best speed").
    max_energy_increase: the same against the lowest predicted energy ("the fastest at nearly the least energy", with
    objective="time"). z: configurations whose objective is within z standard deviations (of the difference) of the
    best one are contenders; 2 leaves about 1 wrong order in 5000 decided pairs on the RTX 3090 validation runs.
    """
    if not predictions:
        raise ValueError("no predictions")
    fastest = min(p.kernel_s for p in predictions)
    cheapest = min(p.energy_j for p in predictions)
    admissible = [i for i, p in enumerate(predictions)
                  if (max_slowdown is None or p.kernel_s <= fastest * (1 + max_slowdown))
                  and (max_energy_increase is None or p.energy_j <= cheapest * (1 + max_energy_increase))]
    values = [math.nan] * len(predictions)
    sds = [math.nan] * len(predictions)
    for i in admissible:
        p = predictions[i]
        values[i] = objective_value(p, objective)
        # Time is measured, so the objective's sd is the energy's times the same power of the time.
        sds[i] = 0.0 if objective == "time" else p.sd_j * values[i] / p.energy_j
    best = min(admissible, key=lambda i: values[i])
    contenders = []
    if objective != "time":
        for i in admissible:
            if i == best:
                continue
            scale_best = values[best] / predictions[best].energy_j
            scale_i = values[i] / predictions[i].energy_j
            relative = float(model.uncertainty["dynamic_relative_sd"])
            baseline = math.hypot(float(model.uncertainty.get("baseline_sd_w", 0.0)),
                                  float(model.uncertainty.get("baseline_unknown_temperature_sd_w", 0.0)))
            a, b = predictions[i], predictions[best]
            sd = math.sqrt((relative * a.dynamic_j * scale_i) ** 2 + (relative * b.dynamic_j * scale_best) ** 2
                           + (baseline * (a.kernel_s * scale_i - b.kernel_s * scale_best)) ** 2)
            if values[i] - values[best] <= z * sd:
                contenders.append(i)
        contenders.sort(key=lambda i: values[i])
    return Recommendation(index=best, contenders=contenders, confident=not contenders, objective=objective,
                          values=values, sds=sds, front=pareto_front(predictions))


# ---- Inputs -----------------------------------------------------------------------------------------------------------

def from_ktt_json(path):
    """Configurations of a KTT output file (JSON, a tuning run with CUPTI profiling): per kernel name, a list of
    (label, counters, kernel_s, temperature_c, sm_clock_mhz). Counters and durations of a result's computations are
    summed. Results that failed or have no counters are left out."""
    with open(path) as file:
        data = json.load(file)
    unit = {"Nanoseconds": 1e-9, "Microseconds": 1e-6, "Milliseconds": 1e-3, "Seconds": 1.0}[
        data.get("Metadata", {}).get("TimeUnit", "Microseconds")]
    kernels = collections.OrderedDict()
    for result in data.get("Results", []):
        if result.get("Status") != "Ok":
            continue
        counters, seconds, temperatures, clocks = {}, 0.0, [], []
        for computation in result.get("ComputationResults", []):
            seconds += float(computation.get("Duration", 0.0)) * unit
            for entry in (computation.get("ProfilingData") or {}).get("Counters", []):
                counters[entry["Name"]] = counters.get(entry["Name"], 0.0) + float(entry["Value"])
            if computation.get("Temperature"):
                temperatures.append(float(computation["Temperature"]))
            if computation.get("SMFrequency"):
                clocks.append(float(computation["SMFrequency"]))
        if not counters or not seconds > 0:
            continue
        label = " ".join(f"{p['Name']}={p['Value']}" for p in result.get("Configuration", []))
        kernels.setdefault(result.get("KernelName", "kernel"), []).append(
            (label, counters, seconds, sum(temperatures) / len(temperatures) if temperatures else None,
             sum(clocks) / len(clocks) if clocks else None))
    return kernels


def from_validation_csv(path):
    """The same from a --validate run's CSV (PREFIX_validation.csv); the row itself serves as the counters."""
    kernels = collections.OrderedDict()
    for row in counter_table.read_rows(path):
        seconds = counter_table.number(row.get("kernel_s"))
        if row.get("valid") != "1" or not seconds > 0 or not row.get("smsp__inst_executed.sum"):
            continue
        temperature = counter_table.number(row.get("temperature_c"))
        clock = counter_table.number(row.get("sm_clock_mhz"))
        kernels.setdefault(row["kernel"], []).append(
            (row["configuration"], row, seconds, None if math.isnan(temperature) else temperature,
             None if math.isnan(clock) else clock))
    return kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model file (tools/build_model.py)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--ktt", help="KTT output file (JSON) of a tuning run with CUPTI profiling")
    source.add_argument("--csv", help="PREFIX_validation.csv of a --validate run")
    parser.add_argument("--objective", choices=OBJECTIVES, default="energy", help="what to minimise (default energy)")
    parser.add_argument("--max-slowdown", type=float, help="only configurations at most this many %% slower than the "
                        "fastest")
    parser.add_argument("--max-energy-increase", type=float, help="only configurations at most this many %% above the "
                        "lowest predicted energy")
    parser.add_argument("--z", type=float, default=2.0, help="standard deviations within which configurations are "
                        "contenders (default 2)")
    parser.add_argument("--power-limit", type=float, help="the GPU's power limit in W (nvidia-smi shows it): configurations "
                        "predicted above it get the limit times their time")
    parser.add_argument("--lenient", action="store_true", help="missing counters become warnings instead of an error")
    parser.add_argument("--all", action="store_true", help="print every configuration, not only the choice")
    arguments = parser.parse_args()

    try:
        model = Model.load(arguments.model)
        model.power_limit_w = arguments.power_limit
        kernels = from_ktt_json(arguments.ktt) if arguments.ktt else from_validation_csv(arguments.csv)
        if not kernels:
            raise ModelError("no configuration with counters and a duration in the input")
        print(f"Model: {model.device.get('name')} at {model.device.get('sm_clock_mhz')} MHz, "
              f"{sum(1 for v in model.pj.values() if v > 0)} priced events, relative sd of the dynamic energy "
              f"{100 * float(model.uncertainty['dynamic_relative_sd']):.1f} %")
        for kernel, configurations in kernels.items():
            predictions = [model.predict(counters, seconds, temperature, clock, label, strict=not arguments.lenient)
                           for label, counters, seconds, temperature, clock in configurations]
            choice = recommend(model, predictions, arguments.objective, arguments.z,
                               None if arguments.max_slowdown is None else arguments.max_slowdown / 100,
                               None if arguments.max_energy_increase is None else arguments.max_energy_increase / 100)
            best = predictions[choice.index]
            print(f"\n{kernel}: {len(predictions)} configurations, objective {arguments.objective}")
            print(f"  use: {best.label}")
            print(f"       {best.energy_j:.4g} J +- {best.sd_j:.2g} per launch ({best.dynamic_j:.4g} dynamic), "
                  f"{1e3 * best.kernel_s:.4g} ms, {best.power_w:.0f} W")
            top = sorted(best.shares.items(), key=lambda item: -item[1])[:4]
            print("       dynamic energy by event: " + ", ".join(f"{name} {100 * share:.0f} %" for name, share in top))
            if choice.confident:
                print("  confident: every other admissible configuration is predicted worse by more than "
                      f"{arguments.z:g} sd")
            else:
                print(f"  {len(choice.contenders)} contender(s) within {arguments.z:g} sd (measure them if the difference "
                      "matters):")
                for i in choice.contenders[:8]:
                    p = predictions[i]
                    print(f"    {100 * (choice.values[i] / choice.values[choice.index] - 1):+5.1f} %  {p.label}  "
                          f"({p.energy_j:.4g} J, {1e3 * p.kernel_s:.4g} ms)")
                if len(choice.contenders) > 8:
                    print(f"    ... and {len(choice.contenders) - 8} more")
            fastest = min(predictions, key=lambda p: p.kernel_s)
            if fastest is not best:
                print(f"  against the fastest configuration: {100 * (best.energy_j / fastest.energy_j - 1):+.1f} % energy, "
                      f"{100 * (best.kernel_s / fastest.kernel_s - 1):+.1f} % time")
            print(f"  energy-time Pareto front: {len(choice.front)} configuration(s)")
            off_clock = sum(any("SM clock" in w for w in p.warnings) for p in predictions)
            if 2 * off_clock > len(predictions):
                print(f"  NOTICE: {off_clock} of {len(predictions)} configurations ran at another SM clock than the model's. "
                      "On a GPU that is not\n  clock-locked and runs at its power limit, energy is close to the limit times "
                      "the time: rank by time there\n  (--objective time). On an unlocked RTX 3090 the model ordered 94 % of "
                      "the pairs right, time alone 97 %.")
            for text in sorted(set(w for p in predictions for w in p.warnings if "SM clock" not in w or off_clock < 4)):
                print(f"  WARNING: {text}")
            if arguments.all:
                for i in sorted(range(len(predictions)), key=lambda i: predictions[i].energy_j):
                    p = predictions[i]
                    mark = "*" if i == choice.index else ("~" if i in choice.contenders else " ")
                    print(f"   {mark} {p.energy_j:10.4g} J +- {p.sd_j:8.2g}  {1e3 * p.kernel_s:9.4g} ms  {p.label}")
    except (ModelError, OSError, KeyError) as error:
        raise SystemExit(f"error: {error}")


if __name__ == "__main__":
    main()
