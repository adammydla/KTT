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

Comparing different kernels (two implementations of one task) is less certain: the model's bias differs from kernel to
kernel (sd between kernels in the file). Give each prediction a group (the kernel, predict(..., group=...)); recommend()
then widens the sd of pairs from different groups by that spread.

Which baseline belongs in the energy depends on how the GPU is used (accounting):
  awake       the GPU awake at the model's clock over the kernel's time (default; kernels run back to back);
  above-idle  only what the kernel costs above an idle GPU that keeps its CUDA context (the work comes once per period
              and the GPU idles in between): the awake baseline minus the idle power;
  dynamic     no baseline at all (the event energies only).

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
ACCOUNTINGS = ("awake", "above-idle", "dynamic")

# At its power limit the GPU lowers its clock and holds the limit; the energy is then the limit times the time. The time
# is measured (CV 0.06 % between runs on the RTX 3090), so what is left is how closely the GPU holds its limit: on the
# unlocked RTX 3090 (ValFree1) the 98 configurations limited by the power cap alone read 308-350 W, mean 341 W, sd 2.1 %.
LIMIT_RELATIVE_SD = 0.021
# The repeatability of a measured energy between runs (2.4 % on the RTX 3090): part of a kernel scale's error
# (EnergyModel::ScaleRunSd).
SCALE_RUN_SD = 0.024

# The keys a model file must have (both predictors check the same list; EnergyModel.cpp RequiredKeys).
REQUIRED_KEYS = (("device",), ("options",), ("events",), ("baseline", "mean_w"), ("baseline", "mean_temperature_c"),
                 ("baseline", "slope_w_per_c"), ("uncertainty", "dynamic_relative_sd"), ("counters", "required"),
                 ("counters", "lsu_shared"), ("counters", "lsu_global"), ("counters", "aliases"))

# Optional counters the events use when a GPU has them (lane counts, the fp16 pipe of Turing and GA100, the fma halves).
# Model files from before 2026-11-13 do not list them; the same names as EnergyModel.cpp then.
OPTIONAL_COUNTERS = ("sm__thread_inst_executed_pipe_alu_pred_on.sum", "sm__thread_inst_executed_pipe_fma_pred_on.sum",
                     "sm__thread_inst_executed_pipe_fp16_pred_on.sum",
                     "sm__thread_inst_executed_pipe_fma_type_fp16_pred_on.sum", "smsp__inst_executed_pipe_fp16.sum",
                     "smsp__inst_executed_pipe_fma_type_fp16.sum", "sm__thread_inst_executed_pipe_fmalite_pred_on.sum")


class ModelError(Exception):
    """A model file that cannot be used, counters that the model cannot price, or a request it cannot answer."""


Prediction = collections.namedtuple(
    "Prediction", "energy_j sd_j dynamic_j baseline_j kernel_s power_w events shares warnings label "
                  "dynamic_sd_j baseline_weight clamped group unscaled_dynamic_j",
    defaults=(None, 1.0, False, None, None))
Prediction.__doc__ = """One configuration: total energy per launch (J) and its standard deviation, its dynamic and
baseline parts (the baseline under the model's accounting), the kernel's time (s), the mean power of the GPU (W), the
event counts, each event's share of the dynamic energy, warnings (text), and the caller's label.
dynamic_sd_j: the part of sd_j that is independent between configurations (None: the model's relative sd times
dynamic_j). baseline_weight: how the baseline's error enters energy_j (1, 0, or -1 when the power limit applies under
the dynamic accounting). clamped: the power limit applied. group: the kernel, for comparing different kernels.
unscaled_dynamic_j: dynamic_j before the kernel's scale and the power limit (kernel_scale() reads it)."""

Recommendation = collections.namedtuple("Recommendation", "index contenders confident objective values sds front")
Recommendation.__doc__ = """index: the configuration with the lowest predicted objective among the admissible ones.
contenders: the others that cannot be told apart from it (indices, best first). confident: no contenders. values, sds:
the objective and its standard deviation per configuration (NaN where not admissible). front: the indices on the
energy-time Pareto front, by time."""


class Model:
    def __init__(self, data, path=None):
        name = path or "model"
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            raise ModelError(f"{name}: not an EnergyProbes model file (format "
                             f"{data.get('format') if isinstance(data, dict) else None!r})")
        if int(data.get("version", 0)) > VERSION:
            raise ModelError(f"{name}: file version {data['version']} is newer than this tool ({VERSION})")
        for key in REQUIRED_KEYS:
            entry = data
            for part in key:
                if not isinstance(entry, dict) or part not in entry:
                    raise ModelError(f"{name}: missing {'/'.join(key)}")
                entry = entry[part]
        self.data = data
        self.path = path
        self.device = data["device"]
        self.options = data["options"]
        self.pj = {event: float(entry["pj"]) for event, entry in data["events"].items()}
        unknown = sorted(set(self.pj) - set(energy_events.EVENTS))
        if unknown:
            raise ModelError(f"{name}: events this tool does not know: {', '.join(unknown)}")
        self.unpriced = {event for event, entry in data["events"].items() if entry.get("status") == "no probe"}
        self.baseline = data["baseline"]
        self.uncertainty = data["uncertainty"]
        counters = data["counters"]
        self.required = list(counters["required"])
        self.groups = list(counters.get("groups", []))
        # Aliases and group columns from the file, as EnergyModel.cpp takes them, so that both predictors read the same
        # counters whatever tools/counters.json says today. Files without group columns: the counter table's.
        self.aliases = dict(counters["aliases"])
        self.group_columns = dict(counters.get("group_columns") or {g: counter_table.GROUPS.get(g) for g in self.groups})
        self.optional = list(counters.get("optional") or OPTIONAL_COUNTERS)
        self.lsu = list(counters["lsu_shared"]) + list(counters["lsu_global"])
        # The GPU's power limit in W, if known (NVML reports it; --power-limit). A configuration predicted above it will
        # not run at the model's clock: the GPU lowers its clock to hold the limit, and the energy is then about the limit
        # times the time. predict() takes that value and adds a warning.
        self.power_limit_w = None
        # Which baseline goes into the energy (ACCOUNTINGS), and the idle power for "above-idle" (default: the GPU idle
        # with a CUDA context, from the calibration's power runs).
        self.accounting = "awake"
        self.idle_w = None
        # A measured scale per kernel (group): the mean of measured / predicted dynamic energy over a few configurations of
        # it (kernel_scale()). It multiplies the kernel's dynamic energies and replaces its bias in comparisons with other
        # kernels by what is left of it (independent_sd). group -> (scale, number of measurements).
        self.kernel_scales = {}

    @classmethod
    def load(cls, path):
        try:
            with open(path) as file:
                data = json.load(file)
        except ValueError as error:
            raise ModelError(f"{path}: not JSON ({error})")
        return cls(data, path)

    def baseline_w(self, temperature_c=None):
        """The power of the GPU awake at the model's clock without work, at a GPU temperature (default: the mean
        temperature of the calibration)."""
        base = float(self.baseline["mean_w"])
        if temperature_c is not None and not math.isnan(temperature_c):
            base += float(self.baseline["slope_w_per_c"]) * (temperature_c - float(self.baseline["mean_temperature_c"]))
        return base

    def idle_power_w(self):
        """The idle power that "above-idle" subtracts: idle_w if set, else the file's idle power with a context."""
        if self.idle_w is not None:
            return float(self.idle_w)
        value = self.baseline.get("context_idle_w")
        if value is None:
            raise ModelError(f"{self.path or 'model'}: no idle power in the file (rebuild it with tools/build_model.py, or "
                             "give the idle power: idle_w / --idle-w)")
        return float(value)

    def counter_names(self):
        """Every counter name the events read: the required ones, the group columns, the lsu instruction counters, the
        optional lane and fp16 counters, and their aliases (the same list as EnergyModel::Counters())."""
        names = set(self.required) | set(c for c in self.group_columns.values() if c) | set(self.lsu) | set(self.optional)
        names |= {alias for alias, canonical in self.aliases.items() if canonical in names}
        return sorted(names)

    def missing_counters(self, counters):
        """The counters the model needs that `counters` has no value of."""
        return [name for name in self.required if math.isnan(self._value(counters, name))]

    def _value(self, counters, name):
        """A counter by its canonical name or one of the file's aliases; NaN if absent."""
        for candidate in [name] + [alias for alias, canonical in self.aliases.items() if canonical == name]:
            value = counter_table.number(counters.get(candidate))
            if not math.isnan(value):
                return value
        return math.nan

    def _canonical(self, counters):
        """The counters under their canonical names (the file's aliases resolved), for energy_events."""
        result = dict(counters)
        for alias, canonical in self.aliases.items():
            if math.isnan(counter_table.number(result.get(canonical))) \
                    and not math.isnan(counter_table.number(result.get(alias))):
                result[canonical] = result[alias]
        return result

    def events(self, counters, kernel_s):
        counts = energy_events.events(self._canonical(counters), fma_halves=bool(self.options.get("fma_halves")))
        return energy_events.add_busy(counts, kernel_s * 1e9, busy=bool(self.options.get("busy", True)),
                                      scale_gbs=float(self.options.get("busy_scale_gbs", energy_events.BUSY_SCALE_GBS)))

    def predict(self, counters, kernel_s, temperature_c=None, sm_clock_mhz=None, label=None, strict=True, group=None,
                device_name=None):
        """Energy of one launch. `counters`: CUPTI counter values summed over the launch, by name (a dict; values may be
        text). `kernel_s`: the launch's time on the GPU. `temperature_c`: GPU temperature if known. `sm_clock_mhz`: the
        SM clock during the launch if known, to check it against the model's. `device_name`: the GPU's name if known,
        to check it against the model's (a model is calibrated on one GPU type). `group`: the kernel (recommend() compares
        predictions of different groups with a wider sd). With `strict`, counters the model needs and does not find, and
        counter values that are not finite, raise ModelError (missing ones would count as 0 and the kernel would look
        cheaper than it is); without, they become a warning and count as 0. Negative values always count as 0 with a
        warning (CUPTI reports some, see below)."""
        if not kernel_s > 0 or math.isinf(kernel_s):
            raise ModelError("kernel_s must be positive and finite")
        if self.accounting not in ACCOUNTINGS:
            raise ModelError(f"accounting must be one of {', '.join(ACCOUNTINGS)}")
        warnings = []
        # Negative values occur in real runs: CUPTI reports the two L2 reduction counters negative now and then in
        # kernels without reductions (24 of 1900 validation rows, 9 probe rows of ProfileModel1). They count as 0 (as in
        # energy_events.counter) with a warning. A value that is not finite is an error of the caller.
        values = {name: counter_table.number(counters[name]) for name in self.counter_names() if name in counters}
        negative = sorted(name for name, value in values.items() if value < 0 and not math.isinf(value))
        infinite = sorted(name for name, value in values.items() if math.isinf(value))
        if infinite and strict:
            raise ModelError("counter values that are not finite: " + ", ".join(f"{n} = {counters[n]}" for n in infinite))
        if negative or infinite:
            warnings.append("counter values that are negative or not finite count as 0: "
                            + ", ".join(f"{n} = {counters[n]}" for n in negative + infinite))
            counters = dict(counters)
            for name in negative + infinite:
                counters[name] = 0.0
        missing = self.missing_counters(counters)
        if missing:
            text = "counters the model needs are missing: " + ", ".join(missing)
            if strict:
                raise ModelError(text + " (collect the counters of tools/counters.json, or predict with strict=False)")
            warnings.append(text + "; their events count as 0")
        for group_name in self.groups:
            column = self.group_columns.get(group_name)
            if column and math.isnan(self._value(counters, column)):
                text = f"the model was fitted with the {group_name}, which these counters do not have"
                if strict:
                    raise ModelError(text)
                warnings.append(text)

        counts = self.events(counters, kernel_s)
        scale = float(self.kernel_scales[group][0]) if group in self.kernel_scales else 1.0
        parts = {name: counts[name] * self.pj.get(name, 0.0) * 1e-12 * scale for name in energy_events.EVENTS}
        dynamic = sum(parts.values())
        unscaled = dynamic / scale
        awake = self.baseline_w(temperature_c) * kernel_s

        instructions = energy_events.counter(self._canonical(counters), "smsp__inst_executed.sum")
        for name in sorted(self.unpriced):
            if counts.get(name, 0.0) > 0 and name.startswith("inst") and instructions > 0 \
                    and counts[name] / instructions >= UNPRICED_SHARE:
                warnings.append(f"{name}: {100 * counts[name] / instructions:.1f} % of the warp instructions, but no probe "
                                "priced it (counts as free)")
            elif counts.get(name, 0.0) > 0 and not name.startswith("inst"):
                warnings.append(f"{name}: {counts[name]:.3g} per launch, but no probe priced it (counts as free)")
        if device_name and self.device.get("name") and device_name != self.device["name"]:
            warnings.append(f"the model was calibrated on {self.device['name']}, this GPU is {device_name}: check it with "
                            "tools/check_model.py")
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
        dynamic_sd = relative * dynamic
        clamped = bool(self.power_limit_w) and awake + dynamic > self.power_limit_w * kernel_s
        if clamped:
            warnings.append(f"predicted power {(awake + dynamic) / kernel_s:.0f} W is above the GPU's limit of "
                            f"{self.power_limit_w:.0f} W: the GPU will lower its clock; energy taken as the limit times "
                            "the time")
            dynamic = max(self.power_limit_w * kernel_s - awake, 0.0)
            # The energy is now the limit times the measured time: its uncertainty is how closely the GPU holds the
            # limit, not the model's relative error of a dynamic energy that no longer enters.
            dynamic_sd = LIMIT_RELATIVE_SD * self.power_limit_w * kernel_s
        # The baseline under the accounting. Its error enters the energy with this weight: fully when a baseline is
        # charged, not at all under "dynamic"; at the limit the total is fixed, so under "dynamic" it enters negatively.
        subtract = {"awake": 0.0, "above-idle": self.idle_power_w() * kernel_s if self.accounting == "above-idle" else 0.0,
                    "dynamic": awake}[self.accounting]
        weight = (0.0 if clamped else 1.0) - (1.0 if self.accounting == "dynamic" else 0.0)
        baseline = awake - subtract
        return Prediction(energy_j=baseline + dynamic, sd_j=math.hypot(dynamic_sd, weight * baseline_sd), dynamic_j=dynamic,
                          baseline_j=baseline, kernel_s=kernel_s, power_w=(awake + dynamic) / kernel_s, events=counts,
                          shares=shares, warnings=warnings, label=label, dynamic_sd_j=dynamic_sd, baseline_weight=weight,
                          clamped=clamped, group=group, unscaled_dynamic_j=unscaled)

    def kernel_scale(self, group, predictions, measured_dynamic_j, minimum=3):
        """Sets and returns the scale of a kernel from configurations measured with NVML: the mean of measured /
        predicted dynamic energy. `predictions`: Prediction objects (their unscaled_dynamic_j is used, so a second call
        cannot compound a scale) or plain unscaled dynamic energies (J). None (and nothing set) with fewer than
        `minimum` measurements. One factor, not one per event: on the RTX 3090 two factors (compute, memory) were worse
        than one up to 8 measurements. Choose the measured configurations spread over the kernel's predicted dynamic
        energies (evenly spaced quantiles), not the cheapest: README, "A kernel's scale"."""
        predicted = [p.unscaled_dynamic_j if isinstance(p, Prediction) else p for p in predictions]
        ratios = [m / p for p, m in zip(predicted, measured_dynamic_j) if p and p > 0 and m > 0]
        if len(ratios) < minimum:
            return None
        self.kernel_scales[group] = (sum(ratios) / len(ratios), len(ratios))
        return self.kernel_scales[group][0]

    def between_kernels_sd(self):
        """The relative sd of the bias between kernels; the within-kernel sd if the file has none (an assumption that
        doubles the variance, rather than none)."""
        value = self.uncertainty.get("dynamic_bias_sd_between_kernels")
        return float(value) if value is not None else float(self.uncertainty["dynamic_relative_sd"])

    def independent_sd(self, prediction, other=None):
        """The part of a prediction's sd that is independent between two configurations (J). For two predictions of
        different groups (kernels) the model's bias differs too: the relative sd widens by the spread of the bias
        between kernels. A prediction at the power limit keeps its own sd: the limit, not the model, sets it."""
        relative = float(self.uncertainty["dynamic_relative_sd"])
        if prediction.clamped:
            return prediction.dynamic_sd_j
        if other is not None and prediction.group != other.group:
            # The kernel's bias against the other: the spread between kernels, or for a kernel with a measured scale
            # what is left of its bias (the scale's error from n measurements and the repeat of a measurement).
            if prediction.group in self.kernel_scales:
                bias = math.hypot(relative / math.sqrt(self.kernel_scales[prediction.group][1]), SCALE_RUN_SD)
            else:
                bias = self.between_kernels_sd()
            relative = math.hypot(relative, bias)
        return relative * prediction.dynamic_j

    def pair_sd(self, a, b, scale_a=1.0, scale_b=1.0):
        """Standard deviation of objective(a) - objective(b), with objective = scale x energy (1 for energy). The
        independent parts add in quadrature; a baseline error is common to both and acts on the difference of their
        times (weighted by how the baseline enters each energy)."""
        baseline = math.hypot(float(self.uncertainty.get("baseline_sd_w", 0.0)),
                              float(self.uncertainty.get("baseline_unknown_temperature_sd_w", 0.0)))
        return math.sqrt((self.independent_sd(a, b) * scale_a) ** 2 + (self.independent_sd(b, a) * scale_b) ** 2
                         + (baseline * (a.baseline_weight * a.kernel_s * scale_a
                                        - b.baseline_weight * b.kernel_s * scale_b)) ** 2)


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
    """Indices of the configurations that no other one beats in both time and predicted energy, by time. Over all
    the configurations given, also those that a limit (--max-slowdown, --max-energy-increase) excludes from the choice:
    the front shows the trade-off itself (Python only)."""
    order = sorted(range(len(predictions)), key=lambda i: (predictions[i].kernel_s, predictions[i].energy_j))
    front, best = [], math.inf
    for i in order:
        if predictions[i].energy_j < best:
            front.append(i)
            best = predictions[i].energy_j
    return front


def admissible_indices(predictions, max_slowdown=None, max_energy_increase=None):
    """The configurations both constraints admit (the same rule in EnergyModel::Recommend). Raises ModelError, saying
    what each constraint left, when none is."""
    if not predictions:
        raise ValueError("no predictions")
    fastest = min(p.kernel_s for p in predictions)
    cheapest = min(p.energy_j for p in predictions)
    by_time = {i for i, p in enumerate(predictions) if max_slowdown is None or p.kernel_s <= fastest * (1 + max_slowdown)}
    by_energy = {i for i, p in enumerate(predictions)
                 if max_energy_increase is None or p.energy_j <= cheapest * (1 + max_energy_increase)}
    admissible = sorted(by_time & by_energy)
    if not admissible:
        raise ModelError(f"no configuration is admissible: max_slowdown {100 * max_slowdown:g} % admits {len(by_time)} of "
                         f"{len(predictions)}, max_energy_increase {100 * max_energy_increase:g} % admits {len(by_energy)}, "
                         "and none is in both")
    return admissible


def recommend(model, predictions, objective="energy", z=2.0, max_slowdown=None, max_energy_increase=None):
    """The configuration to use, and how sure that is.

    objective: what to minimise (OBJECTIVES). max_slowdown: only configurations at most this fraction slower than the
    fastest are admissible (0.05 = within 5 % of the best time; "the least energy at nearly the best speed").
    max_energy_increase: the same against the lowest predicted energy ("the fastest at nearly the least energy", with
    objective="time"). Both are taken against all configurations; when no configuration meets both, ModelError.
    z: configurations whose objective is within z standard deviations (of the difference) of the best one are
    contenders; 2 leaves about 1 wrong order in 5000 decided pairs on the RTX 3090 validation runs. Predictions of
    different groups (kernels) are compared with the wider sd of model.independent_sd().
    """
    admissible = admissible_indices(predictions, max_slowdown, max_energy_increase)
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
        b = predictions[best]
        for i in admissible:
            if i == best:
                continue
            a = predictions[i]
            sd = model.pair_sd(a, b, values[i] / a.energy_j, values[best] / b.energy_j)
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


def ktt_device(path):
    """The GPU a KTT output file was made on (its metadata), or None."""
    with open(path) as file:
        return json.load(file).get("Metadata", {}).get("Device")


def from_validation_csv(path):
    """The same from a --validate run's CSV (PREFIX_validation.csv); the row itself serves as the counters."""
    kernels = collections.OrderedDict()
    for row in counter_table.read_rows(path):
        seconds = counter_table.number(row.get("kernel_s"))
        if row.get("valid") != "1" or not seconds > 0 or not row.get("smsp__inst_executed.sum") \
                or row.get("repeat") not in (None, "", "0"):
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
    parser.add_argument("--accounting", choices=ACCOUNTINGS, default="awake", help="which baseline goes into the energy: "
                        "awake (default; kernels back to back), above-idle (the GPU idles with a context between runs of "
                        "the work) or dynamic (none)")
    parser.add_argument("--idle-w", type=float, help="the idle power for --accounting above-idle (default: the model "
                        "file's idle power with a CUDA context)")
    parser.add_argument("--lenient", action="store_true", help="missing counters and bad counter values become warnings "
                        "instead of an error")
    parser.add_argument("--all", action="store_true", help="print every configuration, not only the choice")
    arguments = parser.parse_args()

    try:
        model = Model.load(arguments.model)
        model.power_limit_w = arguments.power_limit
        model.accounting = arguments.accounting
        model.idle_w = arguments.idle_w
        kernels = from_ktt_json(arguments.ktt) if arguments.ktt else from_validation_csv(arguments.csv)
        device = ktt_device(arguments.ktt) if arguments.ktt else None
        if not kernels:
            raise ModelError("no configuration with counters and a duration in the input")
        print(f"Model: {model.device.get('name')} at {model.device.get('sm_clock_mhz')} MHz, "
              f"{sum(1 for v in model.pj.values() if v > 0)} priced events, relative sd of the dynamic energy "
              f"{100 * float(model.uncertainty['dynamic_relative_sd']):.1f} %, accounting {model.accounting}"
              + (f" (idle {model.idle_power_w():.1f} W)" if model.accounting == "above-idle" else ""))
        for kernel, configurations in kernels.items():
            predictions = [model.predict(counters, seconds, temperature, clock, label, strict=not arguments.lenient,
                                         group=kernel, device_name=device)
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
