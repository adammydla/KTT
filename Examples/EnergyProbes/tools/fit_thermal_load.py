#!/usr/bin/env python3
"""Power over temperature under load, from a --thermal-load run (NEXT_STEPS_PLAN M1).

Every probe (and the sleep baseline) was held twice, once starting cool and once starting hot, so its temperature ran
through the same range upwards and downwards. In 2 s windows of the power readings and the sensor (whole
degrees, averaged over the window), this fits per probe and duty

    P = a + b x T + c x t        (t: seconds since the hold began; skips the first --skip s)

over both holds together: the opposite directions of the temperature separate b from a drift in time c. b of the sleep
baseline is the baseline's own slope; b - b_sleep is what a working GPU adds per kelvin. Then three forms of that excess
are compared over the probes (one coefficient k each):

    flat         excess = k                    (the same at every duty)
    duty         excess = k x duty / 100
    dynamic      excess = k x (P - P_sleep) / 100 W   (proportional to the dynamic power)

    python3 tools/fit_thermal_load.py ThermalLoad1 [--skip 10] [--window 2]
"""
import argparse
import collections
import csv
import re
import sys

import numpy as np


def windows(samples, skip, width, counter=False):
    """(t since start, power, temperature) per window: the mean of NVML's power readings (what the probes' and the
    model's power is), or with counter the energy counter's difference (it covers other rails: lower by about 20 W)."""
    samples.sort()
    t0 = samples[0][0]
    times = np.array([s[0] for s in samples]); energy = np.array([s[1] for s in samples])
    temperature = np.array([s[2] for s in samples]); power = np.array([s[3] for s in samples])
    out = []
    for start in np.arange(t0 + skip, times[-1] - width, width):
        i, j = np.searchsorted(times, start), np.searchsorted(times, start + width)
        if j - i < 5 or times[j - 1] <= times[i]:
            continue
        value = (energy[j - 1] - energy[i]) / (times[j - 1] - times[i]) if counter else power[i:j].mean()
        out.append(((times[i] + times[j - 1]) / 2 - t0, value, temperature[i:j].mean()))
    return out


def fit(points):
    """Least squares P = a + b T + c t; returns (a, b, c, residual sd, T range)."""
    t = np.array([p[0] for p in points]); power = np.array([p[1] for p in points]); temp = np.array([p[2] for p in points])
    design = np.column_stack([np.ones_like(t), temp, t])
    coef, *_ = np.linalg.lstsq(design, power, rcond=None)
    residual = power - design @ coef
    return coef[0], coef[1], coef[2], float(residual.std()), (float(temp.min()), float(temp.max())), float(power.mean()), \
        float(temp.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="--output prefix of the --thermal-load run (its _trace.csv)")
    parser.add_argument("--skip", type=float, default=10.0, help="seconds skipped at the start of each hold (default 10)")
    parser.add_argument("--window", type=float, default=2.0, help="window length in s (default 2)")
    parser.add_argument("--energy-counter", action="store_true", help="power from the energy counter, not the readings")
    args = parser.parse_args()

    traced = collections.defaultdict(list)
    with open(args.run + "_trace.csv") as file:
        for row in csv.DictReader(file):
            name = row["configuration"].split(" ")[0]  # the trace appends the configuration's parameters
            if name.startswith("Hold_"):
                traced[name].append((float(row["time_s"]), float(row["energy_j"]), float(row["temperature_c"]),
                                     float(row["power_w"])))
    groups = collections.defaultdict(dict)
    for name, samples in traced.items():
        found = re.fullmatch(r"Hold_(.+?)(?:_d(\d+))?_(cool|hot)", name)
        if found:
            groups[(found.group(1), int(found.group(2) or 0))][found.group(3)] = windows(samples, args.skip, args.window,
                                                                                        args.energy_counter)
    if ("sleep", 0) not in groups:
        raise SystemExit(f"no sleep holds in {args.run}_trace.csv")

    results = {}
    print(f"{args.run}: P = a + b T + c t over the cool and the hot hold of each probe ({args.window:g} s windows, first "
          f"{args.skip:g} s skipped)")
    print(f"  {'probe':10} {'duty':>4} {'P [W]':>7} {'b [W/K]':>8} {'c [W/s]':>8} {'T up':>11} {'T down':>11} {'sd [W]':>6}  "
          "b per hold (up, down)")
    for key in sorted(groups, key=lambda k: (k[0] != "sleep", k)):
        holds = groups[key]
        points = holds.get("cool", []) + holds.get("hot", [])
        if len(points) < 8:
            continue
        a, b, c, sd, _, mean_p, mean_t = fit(points)
        single = []
        for start in ("cool", "hot"):
            p = holds.get(start, [])
            if len(p) >= 5 and np.ptp([x[2] for x in p]) >= 1:
                temps = np.array([x[2] for x in p]); powers = np.array([x[1] for x in p])
                single.append(f"{np.polyfit(temps, powers, 1)[0]:+.2f}")
            else:
                single.append("-")
        ranges = [f"{min(x[2] for x in holds[s]):.0f}-{max(x[2] for x in holds[s]):.0f} C" if holds.get(s) else "-"
                  for s in ("cool", "hot")]
        results[key] = (b, mean_p, mean_t, a)
        print(f"  {key[0]:10} {key[1]:4d} {mean_p:7.1f} {b:+8.3f} {c:+8.4f} {ranges[0]:>11} {ranges[1]:>11} {sd:6.2f}  "
              f"{', '.join(single)}")

    b_sleep, p_sleep, t_sleep, a_sleep = results[("sleep", 0)]
    probes = [(k, v) for k, v in results.items() if k[0] != "sleep"]
    if not probes:
        return 0
    excess = np.array([v[0] - b_sleep for _, v in probes])
    duty = np.array([k[1] / 100.0 for k, _ in probes])
    # The dynamic power at the probe's mean temperature, against the sleep baseline at that temperature.
    dynamic = np.array([(v[1] - (a_sleep + b_sleep * v[2])) / 100.0 for _, v in probes])
    print(f"\nSleep baseline: b = {b_sleep:+.3f} W/K at {p_sleep:.1f} W. Excess slope of the probes (b - b_sleep): "
          f"mean {excess.mean():+.3f} W/K, range {excess.min():+.3f} to {excess.max():+.3f}")
    print("Forms of the excess (one coefficient over the probes; rms of the residual):")
    for name, x in (("flat", np.ones_like(excess)), ("duty", duty), ("dynamic", dynamic)):
        k = float(x @ excess / (x @ x))
        rms = float(np.sqrt(np.mean((excess - k * x) ** 2)))
        unit = "W/K" if name != "dynamic" else "W/K per 100 W of dynamic power"
        print(f"  {name:8} k = {k:+.3f} {unit}, rms {rms:.3f} W/K")
    return 0


if __name__ == "__main__":
    sys.exit(main())
