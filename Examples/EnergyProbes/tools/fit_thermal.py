#!/usr/bin/env python3
"""The sleep baseline's power over temperature from a --thermal-curve run.

    python3 tools/fit_thermal.py Thermal1        # the --output prefix (its _trace.csv)

In a probe run the baseline is a line through sleeps of 10 s each, a few degrees apart. Here a sleep lasts 180 s while
the GPU cools, traced every 50 ms, so power and temperature come from one continuous measurement. For each cooling
curve (after compute heating, after DRAM heating) and for the staircase of short sleeps, this prints:
- the slope of a line (W/K) with its 95 % interval, and its residual;
- the leakage form of Price et al. (arXiv 1407.8116, a K20): P = c + a T^2 exp(-b / T), T in kelvin, b = 3209.7 K as
  there (over 10-15 K the exponent cannot be fitted on its own), against the line;
- a fit with the fan speed as a second variable (the fans draw board power and follow the temperature);
- the difference of the two cooling curves at the same GPU temperature: if heating DRAM leaves the sleep power higher
  than heating the SMs did, the baseline depends on the memory's temperature too, which NVML does not report.
The first SKIP_S seconds of every sleep are left out: the power reading is a 1 s average that still holds the heater.
"""

import argparse
import csv
import math
import sys

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

SKIP_S = 3.0
PRICE_B = 3209.7  # K


def segments(prefix):
    """The traced samples of every configuration of the run, by label, in run order."""
    result = {}
    with open(prefix + "_trace.csv") as file:
        for row in csv.DictReader(file):
            result.setdefault(row["configuration"], []).append(row)
    return result


def arrays(samples, skip=SKIP_S):
    t = np.array([float(s["time_s"]) for s in samples])
    keep = t >= skip
    power = np.array([float(s["power_w"]) for s in samples])[keep]
    temperature = np.array([float(s["temperature_c"]) for s in samples])[keep]
    fan = np.array([float(s["fan_pct"]) if s.get("fan_pct") not in ("", None) else np.nan for s in samples])[keep]
    return t[keep], power, temperature, fan


def line(temperature, power):
    result = stats.linregress(temperature, power)
    half = stats.t.ppf(0.975, len(power) - 2) * result.stderr
    rms = math.sqrt(np.mean((power - result.intercept - result.slope * temperature) ** 2))
    return result.slope, half, rms, result


def leakage(temperature, power):
    kelvin = temperature + 273.15
    shape = kelvin ** 2 * np.exp(-PRICE_B / kelvin)
    a = np.c_[np.ones_like(shape), shape]
    coefficients, *_ = np.linalg.lstsq(a, power, rcond=None)
    rms = math.sqrt(np.mean((power - a @ coefficients) ** 2))
    # Slope dP/dT of the fitted form at the mean temperature.
    mean = kelvin.mean()
    slope = coefficients[1] * (2 * mean + PRICE_B) * math.exp(-PRICE_B / mean)  # d/dT of T^2 exp(-b / T)
    return slope, rms


def describe(name, temperature, power, fan):
    span = temperature.max() - temperature.min()
    print(f"\n{name}: {len(power)} samples, {temperature.min():.0f}-{temperature.max():.0f} C, {power.min():.1f}-"
          f"{power.max():.1f} W")
    if span < 2:
        print("  spans less than 2 C: no slope")
        return None
    slope, half, rms, fit = line(temperature, power)
    print(f"  line:            {slope:.3f} W/K (95 % {slope - half:.3f} to {slope + half:.3f}), rms {rms:.3f} W")
    leak_slope, leak_rms = leakage(temperature, power)
    print(f"  leakage form:    {leak_slope:.3f} W/K at the mean, rms {leak_rms:.3f} W (b = {PRICE_B} K fixed)")
    quadratic = np.polyfit(temperature - temperature.mean(), power, 2)
    q_rms = math.sqrt(np.mean((power - np.polyval(quadratic, temperature - temperature.mean())) ** 2))
    print(f"  quadratic:       slope {quadratic[1] + 2 * quadratic[0] * (temperature.min() - temperature.mean()):.3f} "
          f"W/K at {temperature.min():.0f} C, {quadratic[1] + 2 * quadratic[0] * (temperature.max() - temperature.mean()):.3f}"
          f" at {temperature.max():.0f} C, rms {q_rms:.3f} W")
    if not np.isnan(fan).any() and np.ptp(fan) > 0:
        a = np.c_[np.ones_like(power), temperature - temperature.mean(), fan - fan.mean()]
        coefficients, *_ = np.linalg.lstsq(a, power, rcond=None)
        f_rms = math.sqrt(np.mean((power - a @ coefficients) ** 2))
        print(f"  with the fan:    {coefficients[1]:.3f} W/K + {coefficients[2]:.3f} W per fan %, rms {f_rms:.3f} W "
              f"(fan {np.nanmin(fan):.0f}-{np.nanmax(fan):.0f} %, correlation with T {np.corrcoef(temperature, fan)[0, 1]:.2f})")
    return fit


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="--output prefix of a --thermal-curve run")
    arguments = parser.parse_args()
    traced = segments(arguments.run)
    curves = {}
    for label, title in (("CoolAfterSpin", "Cooling after 120 s of compute (spin)"),
                         ("CoolAfterDram", "Cooling after 120 s of DRAM loads")):
        found = [k for k in traced if k.startswith(label)]
        if not found:
            print(f"no trace of {label} in {arguments.run}_trace.csv")
            continue
        _, power, temperature, fan = arrays(traced[found[0]])
        curves[label] = (temperature, power)
        describe(title, temperature, power, fan)

    # The staircase: each 10 s sleep after 20 s of DRAM loads, as one point (mean of its samples after SKIP_S).
    points = []
    for label in sorted(k for k in traced if k.startswith("StepSleep")):
        _, power, temperature, _ = arrays(traced[label])
        points.append((temperature.mean(), power.mean()))
    if len(points) >= 3:
        temperature, power = np.array(points).T
        print(f"\nStaircase of {len(points)} sleeps of 10 s after 20 s of DRAM loads each (one point per sleep):")
        for t, p in points:
            print(f"  {t:5.1f} C {p:7.2f} W")
        if np.ptp(temperature) >= 2:
            slope, half, rms, _ = line(temperature, power)
            print(f"  line: {slope:.3f} W/K (95 % {slope - half:.3f} to {slope + half:.3f}), rms {rms:.3f} W")

    # All sleep samples together (the settling sleep, both cooling curves and the staircase): the fan speed varies far
    # more across them than within one curve (DRAM loads drove it to 99 % while the GPU cooled), so temperature and fan
    # separate. Fan power grows with the cube of the speed.
    temperature, power, fan = [], [], []
    for label, samples in traced.items():
        if label.startswith(("SleepSettle", "CoolAfter", "StepSleep")):
            _, p, t, f = arrays(samples)
            temperature += list(t); power += list(p); fan += list(f)
    temperature, power, fan = np.array(temperature), np.array(power), np.array(fan)
    if len(power) > 10 and not np.isnan(fan).any() and np.ptp(fan) > 5:
        print(f"\nAll sleep samples: {len(power)}, {temperature.min():.0f}-{temperature.max():.0f} C, fan {fan.min():.0f}-"
              f"{fan.max():.0f} % (correlation with T {np.corrcoef(temperature, fan)[0, 1]:.2f})")
        for name, columns in (("temperature only", [temperature]), ("temperature + fan", [temperature, fan]),
                              ("temperature + fan^3", [temperature, (fan / 100) ** 3])):
            a = np.column_stack([np.ones(len(power))] + columns)
            c, *_ = np.linalg.lstsq(a, power, rcond=None)
            residual = power - a @ c
            sigma = math.sqrt(residual @ residual / (len(power) - a.shape[1]))
            se = np.sqrt(np.diag(sigma ** 2 * np.linalg.inv(a.T @ a)))
            terms = [f"{c[1]:.3f} +- {1.96 * se[1]:.3f} W/K"]
            if len(columns) > 1:
                unit = "W per fan %" if name.endswith("fan") else "W at full fan speed"
                terms.append(f"{c[2]:.3f} +- {1.96 * se[2]:.3f} {unit}")
            print(f"  {name:22} " + ", ".join(terms) + f", rms {math.sqrt(np.mean(residual ** 2)):.3f} W")

    # The two cooling curves at the same temperature (means per whole degree, as the sensor reads).
    if len(curves) == 2:
        means = {}
        for label, (temperature, power) in curves.items():
            means[label] = {int(t): power[temperature == t].mean() for t in np.unique(temperature)}
        common = sorted(set(means["CoolAfterSpin"]) & set(means["CoolAfterDram"]))
        if common:
            print("\nSleep power after DRAM heating minus after compute heating, at the same GPU temperature:")
            differences = [means["CoolAfterDram"][t] - means["CoolAfterSpin"][t] for t in common]
            for t, d in zip(common, differences):
                print(f"  {t:3d} C {d:+6.2f} W")
            print(f"  mean {np.mean(differences):+.2f} W over {len(common)} degrees")
        else:
            print("\nThe two cooling curves share no temperature.")


if __name__ == "__main__":
    sys.exit(main())
