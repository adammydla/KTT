#!/usr/bin/env python3
"""Turn calibration runs into one model file for energy_predictor.py (and EnergyModel.h).

    python3 tools/build_model.py --power Probes --counters Profile --calibration Validation \
        --output rtx3090_1200mhz.model.json

The file holds everything a prediction needs and where it came from:
  - the GPU, its compute capability and the SM clock of the calibration (a model is valid for that clock);
  - the energy per event (the fit of energy_model.py) and which events no probe priced;
  - the baseline: power of the GPU awake without work, its slope with the temperature, and the idle power;
  - the uncertainty: the standard deviation of the model's relative error of the dynamic energy between the
    configurations of one kernel, measured on the --calibration runs (--validate runs: power and counters of real
    kernels), its mean per kernel (the bias) and the spread of that bias between kernels;
  - the counters the events need;
  - every probe's energy per access at full rate, its time per access and its counts, so that tools/check_model.py
    can test the model on another GPU from a few probes;
  - the runs it was built from, with the SHA-256 of their files.

Without --calibration the uncertainty is the RTX 3090 value (6 %), marked as assumed.
"""

import argparse
import contextlib
import datetime
import hashlib
import io
import itertools
import json
import math
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import energy_model as m  # noqa: E402
import energy_predictor  # noqa: E402

ASSUMED_RELATIVE_SD = 0.06  # RTX 3090
# Baseline uncertainty when the temperature of a prediction is unknown: the slope times this many degrees.
UNKNOWN_TEMPERATURE_C = 5.0
# Calibration rows whose SM clock is further than this from the model's are left out (throttled launches: a few gemm
# configurations ran at 1105-1170 MHz under the power cap).
CLOCK_TOLERANCE = 0.02


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source(prefix, suffix):
    path = prefix + suffix
    return {"run": os.path.basename(prefix), "file": os.path.basename(path), "sha256": sha256(path)}


def baseline_scatter(prefixes):
    """Rms of the sleep rows around each power run's own temperature line (W): how well the baseline is known at a
    known temperature."""
    residuals = []
    for prefix in prefixes:
        sleeps = [(m.number(r["temperature_c"]), m.number(r["power_w"])) for r in m.read_rows(prefix + ".csv")
                  if r["level"] == "baseline" and r["probe"].startswith("sleep-") and not r["probe"].startswith("sleep-sm")]
        if len(sleeps) >= 3:
            t, p = np.array(sleeps).T
            residuals += list(p - np.polyval(np.polyfit(t, p, 1), t))
    return float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.5


def idle_power(table, context):
    """The idle power of the first power run's energy table (W), without or with a CUDA context; None if absent."""
    rows = [r for r in table.get("idle") or [] if (r.get("label") != "idle-noctx") == context
            and r.get("power_w") is not None]
    return float(np.mean([r["power_w"] for r in rows])) if rows else None


def probe_records(power_prefixes, probes, energies):
    """Per fitted probe: label, the energy the fit used, the model's energy, and its full-rate row (first power run that
    has it): energy per access at duty 100, time per access, power above the baseline."""
    rows = {}
    for prefix in power_prefixes:
        for r in m.probe_rows(prefix):
            rows.setdefault(m.label_of(m.key_of(r)), r)
    records = []
    for probe in probes:
        record = {"label": probe.label, "pj_per_access": probe.energy,
                  "model_pj_per_access": sum(probe.counts[name] * energies[name] for name in m.EVENTS),
                  # Events per access of the counter run, for tools/check_model.py --counters.
                  "counts": {name: float(value) for name, value in probe.counts.items()
                             if name != "DRAM busy ns" and value and not math.isnan(value)}}
        row = rows.get(probe.label)
        if row is not None and m.number(row["accesses_per_launch"]) > 0:
            record.update({
                "probe": row["probe"], "level": row["level"], "operation": row["operation"],
                "width_bytes": int(row["width_bytes"]), "footprint_bytes": int(row["footprint_bytes"]),
                "full_rate_pj_per_access": m.number(row["pj_per_access"]),
                "ns_per_access": m.number(row["active_s"]) / m.number(row["accesses_per_launch"]) * 1e9,
                "dynamic_w": m.number(row["power_w"]) - m.number(row["baseline_w"]),
            })
        records.append(rounded(record))
    return records


def rounded(value):
    """Floats to six significant digits (the check compares within percents), NaN to null."""
    if isinstance(value, dict):
        return {key: rounded(entry) for key, entry in value.items()}
    if isinstance(value, float):
        return None if math.isnan(value) else float(f"{value:.6g}")
    return value


def tools_commit():
    """The git commit of the tools (with "+changes" when the checkout has uncommitted changes), or None outside one."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                universal_newlines=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", ".."], cwd=here, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, universal_newlines=True, check=True).stdout.strip()
        return commit + ("+changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def calibrate(model, prefixes, table, z=2.0):
    """The model's errors on validation runs, through the predictor itself. Returns the uncertainty entries and the lines
    of a report."""
    groups = []  # (run, kernel, relative dynamic errors, predictions, measured total energies)
    clock = m.number(table.get("sm_clock_mhz"))
    off_clock = []
    for prefix in prefixes:
        with contextlib.redirect_stdout(io.StringIO()):
            rows = m.load_validation(prefix, table)
        for kernel in dict.fromkeys(r["kernel"] for r in rows):
            group = [r for r in rows if r["kernel"] == kernel]
            # Launches at another clock than the model's (throttled) are not what the model describes.
            if not math.isnan(clock):
                kept = [r for r in group if math.isnan(m.number(r.get("sm_clock_mhz")))
                        or abs(m.number(r["sm_clock_mhz"]) / clock - 1) <= CLOCK_TOLERANCE]
                off_clock += [(os.path.basename(prefix), kernel, m.number(r["sm_clock_mhz"]))
                              for r in group if r not in kept]
                group = kept
            predictions = [model.predict(r, m.number(r["kernel_s"]), strict=False) for r in group]
            # The measured energies use the validation run's own baseline, so the comparison is on the dynamic part.
            errors = np.array([p.dynamic_j / m.number(r["dynamic_energy_j"]) - 1 for p, r in zip(predictions, group)])
            baseline = np.array([m.number(r["baseline_w"]) * m.number(r["kernel_s"]) for r in group])
            groups.append((os.path.basename(prefix), kernel, errors, predictions,
                           np.array([m.number(r["energy_j"]) for r in group]), baseline))

    def pooled(selection):
        variances = [g[2].var(ddof=1) for g in selection if len(g[2]) > 2]
        return float(np.sqrt(np.mean(variances)))

    if not groups:
        raise SystemExit("no calibration rows at the model's clock")
    kernels = sorted(set(g[1] for g in groups))
    relative = pooled(groups)
    per_kernel = {k: pooled([g for g in groups if g[1] == k]) for k in kernels}
    bias = float(np.mean([g[2].mean() for g in groups]))
    # The bias changes from kernel to kernel: its spread is the uncertainty of an absolute prediction.
    kernel_bias = {k: float(np.mean([g[2].mean() for g in groups if g[1] == k])) for k in kernels}
    between = float(np.std(list(kernel_bias.values()), ddof=1)) if len(kernels) > 1 else None

    # How the two-sd rule would have decided: pairs of configurations of one kernel, totals with the measured baseline.
    decided = right_decided = undecided = right_undecided = picks = confident = worst = 0
    regrets = []
    for _, _, _, predictions, measured, baseline in groups:
        totals = [p.dynamic_j + b for p, b in zip(predictions, baseline)]
        for i, j in itertools.combinations(range(len(totals)), 2):
            sd = relative * math.hypot(predictions[i].dynamic_j, predictions[j].dynamic_j)
            right = (totals[i] < totals[j]) == (measured[i] < measured[j])
            if abs(totals[i] - totals[j]) > z * sd:
                decided += 1
                right_decided += right
            else:
                undecided += 1
                right_undecided += right
        best = int(np.argmin(totals))
        regrets.append(measured[best] / measured.min() - 1)
        contenders = [i for i in range(len(totals)) if i != best and totals[i] - totals[best]
                      <= z * relative * math.hypot(predictions[i].dynamic_j, predictions[best].dynamic_j)]
        picks += 1
        confident += not contenders
        # With the contenders measured, the choice is the measured best among them and the first pick.
        candidates = [best] + contenders
        worst = max(worst, measured[candidates].min() / measured.min() - 1)

    lines = [f"Calibration on {len(groups)} kernel runs ({sum(len(g[2]) for g in groups)} configurations, kernels: "
             f"{', '.join(kernels)}):",
             f"  relative sd of the dynamic energy within a kernel: {100 * relative:.1f} %  (per kernel: "
             + ", ".join(f"{k} {100 * v:.1f}" for k, v in per_kernel.items()) + ")"]
    # The pooled sd is what the predictors use; the largest kernel's is stored next to it as the cautious value for a
    # kernel unlike the calibration's.
    largest = max(per_kernel, key=per_kernel.get)
    lines.append(f"  largest within one kernel: {100 * per_kernel[largest]:.1f} % ({largest}); the predictors use the pooled "
                 f"{100 * relative:.1f} %")
    lines += [f"  bias of the dynamic energy: {100 * bias:+.1f} %  (per kernel: "
              + ", ".join(f"{k} {100 * v:+.1f}" for k, v in kernel_bias.items()) + ")",
              f"  pairs told apart at {z:g} sd: {decided} of {decided + undecided} "
              f"({100 * decided / max(decided + undecided, 1):.0f} %), ordered right {100 * right_decided / max(decided, 1):.2f} %; "
              f"the rest ordered right {100 * right_undecided / max(undecided, 1):.0f} %",
              f"  first choice: regret mean {100 * np.mean(regrets):.2f} %, max {100 * max(regrets):.1f} %; confident in "
              f"{confident} of {picks} kernel runs; with the contenders measured, max regret {100 * worst:.1f} %"]
    if between is not None:
        lines.append(f"  bias between kernels: sd {100 * between:.1f} % (pairs of configurations of different kernels are "
                     f"compared with {100 * math.hypot(relative, between):.1f} % per prediction)")
    if off_clock:
        lines.append(f"  left out: {len(off_clock)} configurations more than {100 * CLOCK_TOLERANCE:.0f} % off the model's "
                     "clock (" + ", ".join(f"{run} {kernel} {c:.0f} MHz" for run, kernel, c in off_clock[:6])
                     + (", ..." if len(off_clock) > 6 else "") + ")")
    uncertainty = {
        "dynamic_relative_sd": relative, "dynamic_bias": bias,
        "dynamic_relative_sd_largest_kernel": {"kernel": largest, "sd": per_kernel[largest]},
        "dynamic_bias_sd_between_kernels": between,
        "calibration": {
            "runs": [os.path.basename(p) for p in prefixes], "kernels": kernels,
            "configurations": int(sum(len(g[2]) for g in groups)),
            "relative_sd_per_kernel": per_kernel,
            "bias_per_kernel": kernel_bias,
            "pairs_decided_share": decided / max(decided + undecided, 1),
            "pairs_decided_right": right_decided / max(decided, 1),
            "pairs_undecided_right": right_undecided / max(undecided, 1),
            "regret_mean": float(np.mean(regrets)), "regret_max": float(max(regrets)), "z": z,
            "left_out_off_clock": len(off_clock), "clock_tolerance": CLOCK_TOLERANCE,
        },
    }
    return uncertainty, lines


def dump(value, indent=0, width=150):
    """JSON with one line per entry: objects are spread over lines when they do not fit, lists hold one element per
    line (a probe, a source file)."""
    text = json.dumps(value)
    if len(text) + indent <= width or not isinstance(value, (dict, list)) or not value:
        return text
    pad = " " * (indent + 1)
    if isinstance(value, dict):
        return "{\n" + ",\n".join(pad + json.dumps(k) + ": " + dump(v, indent + 1, width) for k, v in value.items()) \
            + "\n" + " " * indent + "}"
    return "[\n" + ",\n".join(pad + json.dumps(v) for v in value) + "\n" + " " * indent + "]"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power", required=True, help="probe runs, comma-separated (as for energy_model.py)")
    parser.add_argument("--counters", required=True, help="--profile runs of the probes, comma-separated")
    parser.add_argument("--calibration", help="--validate runs for the uncertainty, comma-separated")
    parser.add_argument("--output", required=True, help="the model file to write (e.g. rtx3090_1200mhz.model.json)")
    parser.add_argument("--busy-scale", type=float, default=m.energy_events.BUSY_SCALE_GBS, help="as in energy_model.py")
    arguments = parser.parse_args()
    power, counters = arguments.power.split(","), arguments.counters.split(",")
    calibration = arguments.calibration.split(",") if arguments.calibration else []

    with contextlib.redirect_stdout(io.StringIO()) as notes:
        probes, table = m.probe_table(power, counters, arguments.busy_scale)
    if not probes:
        raise SystemExit("no probe has both an energy and counters; check the prefixes")
    energies, fitted, unused = m.fit(probes)
    present = m.counters_present([r for prefix in counters for r in m.probe_rows(prefix)])
    groups = sorted(g for g, column in m.counter_table.GROUPS.items() if column in present)
    provenance = table.get("provenance") or {}
    clocks = [m.number(r["memory_clock_mhz"]) for prefix in power for r in m.read_rows(prefix + ".csv")
              if r["level"] == "baseline"]
    errors = np.array([f / p.energy - 1 for f, p in zip(fitted, probes)])
    data = {
        "format": energy_predictor.FORMAT,
        "version": energy_predictor.VERSION,
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": {
            "name": table.get("device"),
            "compute_capability": table.get("compute_capability")
            or m.counter_table.device_compute_capability(table.get("device") or ""),
            "sm_clock_mhz": table.get("sm_clock_mhz"),
            "memory_clock_mhz": float(np.nanmedian(clocks)) if clocks and not np.all(np.isnan(clocks)) else None,
            "driver_version": provenance.get("driver_version"), "cuda_driver_version": provenance.get("cuda_driver_version"),
            "uuid": provenance.get("uuid"), "serial": provenance.get("serial"),
            "power_limit_w": provenance.get("power_limit_w"),
        },
        "tools": {"commit": tools_commit(), "events_sha256": sha256(os.path.join(os.path.dirname(os.path.abspath(
            __file__)), "energy_events.py"))},
        "options": {"busy_scale_gbs": arguments.busy_scale},
        "events": {name: {"pj": float(energies[name]),
                          "status": "no probe" if name in unused else ("fitted" if energies[name] > 0 else "zero")}
                   for name in m.EVENTS},
        "baseline": {"mean_w": float(table["baseline"]["mean_w"]),
                     "mean_temperature_c": float(table["baseline"]["mean_temperature_c"]),
                     "slope_w_per_c": float(table["baseline"]["slope_w_per_c"]),
                     # The idle GPU (for the accountings of energy_predictor.py): without a CUDA context (memory clock
                     # down), and with a context but no kernel (the mean of the run's idle rows with a context).
                     "idle_w": idle_power(table, context=False), "context_idle_w": idle_power(table, context=True)},
        # The names EnergyModel.h needs without the counter table: aliases (the same counter under another chip's name)
        # and the lsu pipe's instruction counters by memory space.
        "counters": {"required": sorted(name for name in present if "group" not in m.counter_table.ENTRIES[name]),
                     "groups": groups, "group_columns": {g: m.counter_table.GROUPS[g] for g in groups},
                     "aliases": dict(m.counter_table.CANONICAL),
                     "lsu_shared": list(m.energy_events.SHARED_INSTRUCTIONS),
                     "lsu_global": list(m.energy_events.GLOBAL_INSTRUCTIONS),
                     "optional": list(energy_predictor.OPTIONAL_COUNTERS)},
        "fit": {"probes": len(probes), "median_abs_error": float(np.median(np.abs(errors))),
                "within_10_pct": int((np.abs(errors) <= 0.10).sum())},
        "probes": probe_records(power, probes, energies),
        "sources": {"power": [source(p, ".csv") for p in power] + [source(p, "_energy_table.json") for p in power],
                    "counters": [source(p, ".csv") for p in counters],
                    "calibration": [source(p, "_validation.csv") for p in calibration]},
    }
    data["uncertainty"] = {"dynamic_relative_sd": ASSUMED_RELATIVE_SD, "dynamic_bias": None, "calibration": None,
                           "assumed": True, "baseline_sd_w": baseline_scatter(power),
                           "baseline_unknown_temperature_sd_w": UNKNOWN_TEMPERATURE_C * data["baseline"]["slope_w_per_c"]}

    print(f"{data['device']['name']} (compute capability {data['device']['compute_capability']}) at "
          f"{data['device']['sm_clock_mhz']} MHz: {len(probes)} probes, "
          f"{sum(1 for e in data['events'].values() if e['status'] == 'fitted')} events priced, fit median error "
          f"{100 * data['fit']['median_abs_error']:.1f} %, {data['fit']['within_10_pct']} probes within 10 %")
    for line in notes.getvalue().splitlines():
        print("  " + line)
    if unused:
        print("  no probe prices: " + ", ".join(sorted(unused)))
    if calibration:
        uncertainty, lines = calibrate(energy_predictor.Model(data), calibration, table)
        data["uncertainty"].update(uncertainty, assumed=False)
        print("\n".join(lines))
    else:
        print(f"No --calibration: the relative sd is assumed ({100 * ASSUMED_RELATIVE_SD:.0f} %, from an RTX 3090).")

    with open(arguments.output, "w") as file:
        file.write(dump(data) + "\n")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
