#!/usr/bin/env python3
"""Turn calibration runs into one model file for energy_predictor.py (and EnergyModel.h).

    python3 tools/build_model.py --power Preset3 --counters ProfileModel1 \
        --calibration Validation11,Validation12,Validation13,HeldOut1,HeldOut2 --output rtx3090_1200mhz.model.json

The file holds everything a prediction needs and where it came from:
  - the GPU, its compute capability and the SM clock of the calibration (a model is valid for that clock);
  - the energy per event (the fit of energy_model.py with the same options) and which events no probe priced;
  - the baseline: power of the GPU awake without work, and its slope with the temperature;
  - the uncertainty: the standard deviation of the model's relative error of the dynamic energy between the
    configurations of one kernel, measured on the --calibration runs (validation runs: power and counters of real
    kernels), its mean (the bias), and the same with each kernel left out in turn (how much it depends on the kernels);
  - the counters the events need and the optional counter groups the fit had;
  - every probe's energy per access at full rate, its time per access and its power above the baseline, so that
    tools/check_model.py can test the model on a GPU from a few probes without profiling;
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
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import energy_model as m  # noqa: E402
import energy_predictor  # noqa: E402

ASSUMED_RELATIVE_SD = 0.06  # RTX 3090, 5 validation runs of 7 kernels (2026-11-11)
# Baseline uncertainty when the temperature of a prediction is unknown: the slope times this many degrees.
UNKNOWN_TEMPERATURE_C = 5.0


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
                  if r["level"] == "baseline" and r["active_sms"] == "all" and r["probe"].startswith("sleep-")
                  and not r["probe"].startswith("sleep-sm")]
        if len(sleeps) >= 3:
            t, p = np.array(sleeps).T
            residuals += list(p - np.polyval(np.polyfit(t, p, 1), t))
    return float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.5


def probe_records(power_prefixes, probes, energies):
    """Per fitted probe: label, the energy the fit used, the model's energy, and its full-rate row (first power run that
    has it): energy per access at duty 100, time per access, power above the baseline."""
    rows = {}
    for prefix in power_prefixes:
        for r in m.read_rows(prefix + ".csv"):
            if r["payload"] == "random" and r["active_sms"] == "all" and r["duty_pct"] == "100":
                label = f"{r['level']} {r['operation']}" + (f" w{int(r['width_bytes'])}" if int(r["width_bytes"]) else "")
                rows.setdefault(label, r)
    records = []
    for probe in probes:
        record = {"label": probe.label, "pj_per_access": probe.energy,
                  "model_pj_per_access": sum(probe.counts[name] * energies[name] for name in m.EVENTS)}
        row = rows.get(probe.label)
        if row is not None and m.number(row["accesses_per_launch"]) > 0:
            record.update({
                "probe": row["probe"], "level": row["level"], "operation": row["operation"],
                "width_bytes": int(row["width_bytes"]), "footprint_bytes": int(row["footprint_bytes"]),
                "full_rate_pj_per_access": m.number(row["pj_per_access"]),
                "ns_per_access": m.number(row["active_s"]) / m.number(row["accesses_per_launch"]) * 1e9,
                "dynamic_w": m.number(row["power_w"]) - m.number(row["baseline_w"]),
            })
            record = {key: (None if isinstance(value, float) and math.isnan(value) else value)
                      for key, value in record.items()}
        records.append(record)
    return records


def calibrate(model, prefixes, table, z=2.0):
    """The model's errors on validation runs, through the predictor itself. Returns the uncertainty entries and the lines
    of a report."""
    groups = []  # (run, kernel, relative dynamic errors, predictions, measured total energies)
    for prefix in prefixes:
        with contextlib.redirect_stdout(io.StringIO()):
            rows = m.load_validation(prefix, table)
        for kernel in dict.fromkeys(r["kernel"] for r in rows):
            if kernel == m.CONTROL:
                continue
            group = [r for r in rows if r["kernel"] == kernel]
            predictions = [model.predict(r, m.number(r["kernel_s"]), strict=False) for r in group]
            # The measured energies use the validation run's own baseline, so the comparison is on the dynamic part.
            errors = np.array([p.dynamic_j / m.number(r["dynamic_energy_j"]) - 1 for p, r in zip(predictions, group)])
            baseline = np.array([m.number(r["baseline_w"]) * m.number(r["kernel_s"]) for r in group])
            groups.append((os.path.basename(prefix), kernel, errors, predictions,
                           np.array([m.number(r["energy_j"]) for r in group]), baseline))

    def pooled(selection):
        variances = [g[2].var(ddof=1) for g in selection if len(g[2]) > 2]
        return float(np.sqrt(np.mean(variances)))

    kernels = sorted(set(g[1] for g in groups))
    relative = pooled(groups)
    left_out = {k: pooled([g for g in groups if g[1] != k]) for k in kernels} if len(kernels) > 1 else {}
    per_kernel = {k: pooled([g for g in groups if g[1] == k]) for k in kernels}
    bias = float(np.mean([g[2].mean() for g in groups]))
    # The bias changes from kernel to kernel: its spread is the uncertainty of an absolute prediction.
    kernel_bias = {k: float(np.mean([g[2].mean() for g in groups if g[1] == k])) for k in kernels}

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
    if left_out:
        lines.append(f"  with one kernel left out: {100 * min(left_out.values()):.1f} to {100 * max(left_out.values()):.1f} %")
    lines += [f"  bias of the dynamic energy: {100 * bias:+.1f} %  (per kernel: "
              + ", ".join(f"{k} {100 * v:+.1f}" for k, v in kernel_bias.items()) + ")",
              f"  pairs told apart at {z:g} sd: {decided} of {decided + undecided} "
              f"({100 * decided / max(decided + undecided, 1):.0f} %), ordered right {100 * right_decided / max(decided, 1):.2f} %; "
              f"the rest ordered right {100 * right_undecided / max(undecided, 1):.0f} %",
              f"  first choice: regret mean {100 * np.mean(regrets):.2f} %, max {100 * max(regrets):.1f} %; confident in "
              f"{confident} of {picks} kernel runs; with the contenders measured, max regret {100 * worst:.1f} %"]
    uncertainty = {
        "dynamic_relative_sd": relative, "dynamic_bias": bias,
        "dynamic_bias_sd_between_kernels": float(np.std(list(kernel_bias.values()), ddof=1)) if len(kernels) > 1 else None,
        "calibration": {
            "runs": [os.path.basename(p) for p in prefixes], "kernels": kernels,
            "configurations": int(sum(len(g[2]) for g in groups)),
            "relative_sd_per_kernel": per_kernel, "relative_sd_leaving_one_kernel_out": left_out,
            "bias_per_kernel": kernel_bias,
            "pairs_decided_share": decided / max(decided + undecided, 1),
            "pairs_decided_right": right_decided / max(decided, 1),
            "pairs_undecided_right": right_undecided / max(undecided, 1),
            "regret_mean": float(np.mean(regrets)), "regret_max": float(max(regrets)), "z": z,
        },
    }
    return uncertainty, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power", required=True, help="probe power runs, comma-separated (as for energy_model.py)")
    parser.add_argument("--counters", required=True, help="profiled probe runs, comma-separated")
    parser.add_argument("--calibration", help="validation runs for the uncertainty, comma-separated")
    parser.add_argument("--output", required=True, help="the model file to write (e.g. rtx3090_1200mhz.model.json)")
    parser.add_argument("--weighted", action="store_true", help="as in energy_model.py")
    parser.add_argument("--fma-halves", action="store_true", help="as in energy_model.py")
    parser.add_argument("--no-busy", action="store_true", help="as in energy_model.py")
    parser.add_argument("--busy-scale", type=float, default=m.BUSY_SCALE_GBS, help="as in energy_model.py")
    arguments = parser.parse_args()
    m.configure(arguments.fma_halves, not arguments.no_busy, arguments.busy_scale)
    power, counters = arguments.power.split(","), arguments.counters.split(",")
    calibration = arguments.calibration.split(",") if arguments.calibration else []

    with contextlib.redirect_stdout(io.StringIO()) as notes:
        probes, table = m.probe_table(power, counters)
    if not probes:
        raise SystemExit("no probe has both an energy and counters; check the prefixes")
    if any(p.label == "compute ffma3" for p in probes):
        probes = [p for p in probes if p.label not in m.TWO_REGISTER_FFMA]
    probes = [p for p in probes if p.label not in m.CHECK_ONLY]
    energies, fitted, unused = m.fit(probes, m.EVENTS, arguments.weighted)

    counter_rows = m.counter_rows(counters)
    present = m.counters_present(counter_rows)
    groups = sorted(set.intersection(*[set(p.groups) for p in probes]) - (set() if arguments.fma_halves
                                                                         else {"fma pipe halves"}))
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
        },
        "options": {"busy": not arguments.no_busy, "busy_scale_gbs": arguments.busy_scale,
                    "fma_halves": arguments.fma_halves, "weighted": arguments.weighted},
        "events": {name: {"pj": float(energies[name]),
                          "status": "no probe" if name in unused else ("fitted" if energies[name] > 0 else "zero")}
                   for name in m.EVENTS},
        "baseline": {"mean_w": float(table["baseline"]["mean_w"]),
                     "mean_temperature_c": float(table["baseline"]["mean_temperature_c"]),
                     "slope_w_per_c": float(table["baseline"]["slope_w_per_c"])},
        # The names EnergyModel.h needs without the counter table: aliases (the same counter under another chip's name)
        # and the lsu pipe's instruction counters by memory space.
        "counters": {"required": sorted(name for name in m.COUNTER_EVENTS if name in present
                                        and name not in m.OPTIONAL_COUNTERS), "groups": groups,
                     "group_columns": {g: m.COUNTER_GROUPS[g] for g in groups},
                     "aliases": dict(m.counter_table.CANONICAL),
                     "lsu_shared": list(m.SHARED_INSTRUCTIONS), "lsu_global": list(m.GLOBAL_INSTRUCTIONS)},
        "fit": {"probes": len(probes), "median_abs_error": float(np.median(np.abs(errors))),
                "within_10_pct": int((np.abs(errors) <= 0.10).sum())},
        "probes": probe_records(power, probes, energies),
        "sources": {"power": [source(p, ".csv") for p in power] + [source(p, "_energy_table.json") for p in power],
                    "counters": [source(p, ".csv") for p in counters],
                    "calibration": [source(p, "_validation.csv") for p in calibration]},
    }
    data["uncertainty"] = {"dynamic_relative_sd": ASSUMED_RELATIVE_SD, "dynamic_bias": None, "calibration": None,
                           "assumed": True}
    data["uncertainty"]["baseline_sd_w"] = baseline_scatter(power)
    data["uncertainty"]["baseline_unknown_temperature_sd_w"] = UNKNOWN_TEMPERATURE_C * data["baseline"]["slope_w_per_c"]

    print(f"{data['device']['name']} (compute capability {data['device']['compute_capability']}) at "
          f"{data['device']['sm_clock_mhz']} MHz: {len(probes)} probes, "
          f"{sum(1 for e in data['events'].values() if e['status'] == 'fitted')} events priced, fit median error "
          f"{100 * data['fit']['median_abs_error']:.1f} %, {data['fit']['within_10_pct']} probes within 10 %")
    for line in notes.getvalue().splitlines():
        print("  " + line)
    if unused:
        print("  no probe prices: " + ", ".join(sorted(unused)))
    if calibration:
        model = energy_predictor.Model(data)
        uncertainty, lines = calibrate(model, calibration, table)
        data["uncertainty"].update(uncertainty)
        data["uncertainty"]["assumed"] = False
        print("\n".join(lines))
    else:
        print(f"No --calibration: the relative sd is assumed ({100 * ASSUMED_RELATIVE_SD:.0f} %, from an RTX 3090).")

    with open(arguments.output, "w") as file:
        json.dump(data, file, indent=1)
        file.write("\n")
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
