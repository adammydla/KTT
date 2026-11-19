#!/usr/bin/env python3
"""The DRAM busy event's scale on this GPU: which scales its probes allow (Kimi review P3, 2026-11-16).

The model prices a fixed power of DRAM kept busy, `DRAM busy ns` = time x (1 - exp(-rate / scale)), with the scale of an
RTX 3090 (95 GB/s, energy_events.BUSY_SCALE_GBS). The NNLS fit prices the event on every GPU, but the scale is a shape,
not a price. This refits the probes for every scale of a log grid (10 to 2560 GB/s) and without the event, and prints:

  - the scan: the sum of squared relative errors of the probe fit per scale;
  - the best scale and a profile-likelihood 95 % interval (sum within 3.84 x the residual variance of the smallest);
  - whether the probes need the event at all (the fit without it is outside that bound);
  - the scale build_model.py uses: the RTX 3090's if it lies in the interval, else the best one;
  - a cross-check from the dependent-load probes alone: power above the baseline of the DRAM chase rows against their
    rate, e x rate + F x (1 - exp(-rate / scale)), with 3 parameters on the rows there are (4 in the lean program, so
    usually not identifiable).

Only probes enter; no validation kernel. On an RTX 3090 and an RTX 2080 Ti (2026-11-16) the best scale is 50-64 GB/s,
the interval reaches from below 10 to 200-320 GB/s, and both need the event: the probes bound the scale from above
but hardly from below, so 95 GB/s is kept there.

    python3 tools/fit_busy.py --power Probes --counters Profile
"""

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import energy_model as m  # noqa: E402


def chase_fit(power_prefixes):
    """The direct fit on the DRAM dependent-load rows (duty 100, at the baseline's clock): a line of text."""
    from scipy.optimize import curve_fit
    rows = [r for prefix in power_prefixes for r in m.read_rows(prefix + ".csv")
            if r["operation"].startswith("chase") and r["level"] == "DRAM" and r["duty_pct"] == "100"
            and abs(m.number(r.get("clock_difference_pct")) or 0.0) <= 100 * m.CLOCK_TOLERANCE]
    rate = np.array([m.number(r["bandwidth_gbs"]) for r in rows])
    dynamic = np.array([m.number(r["power_w"]) - m.number(r["baseline_w"]) for r in rows])
    keep = np.isfinite(rate) & np.isfinite(dynamic)
    rate, dynamic = rate[keep], dynamic[keep]
    if len(rate) < 4:
        return f"{len(rate)} DRAM chase rows: too few for the cross-check"
    shape = lambda r, e, f, scale: e * r + f * -np.expm1(-r / scale)  # noqa: E731
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # an undetermined covariance is reported below
            p, covariance = curve_fit(shape, rate, dynamic, p0=(0.17, 10.0, 100.0), maxfev=20000)
    except RuntimeError:
        return f"{len(rate)} DRAM chase rows ({rate.min():.0f}-{rate.max():.0f} GB/s): no fit"
    sd = np.sqrt(np.diag(covariance))
    text = (f"{len(rate)} DRAM chase rows ({rate.min():.0f}-{rate.max():.0f} GB/s): {1000 * p[0]:.0f} pJ/B + {p[1]:.1f} W "
            f"x (1 - exp(-rate / {p[2]:.0f} GB/s))")
    if not np.all(np.isfinite(sd)) or sd[2] > p[2] or p[2] > rate.max() or p[2] <= 0:
        return text + " - not identifiable from these rows"
    return text + f" (sd of the scale {sd[2]:.0f} GB/s)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power", required=True, help="probe runs, comma-separated (as for energy_model.py)")
    parser.add_argument("--counters", required=True, help="--profile runs of the probes, comma-separated")
    arguments = parser.parse_args()
    power, counters = arguments.power.split(","), arguments.counters.split(",")
    scale, source, result = m.busy_scale_choice(power, counters)
    print("Probe fit per DRAM busy scale (sum of squared relative errors; * inside the 95 % interval):")
    for value, error in result["scan"]:
        print(f"  {value:7g} GB/s  {error:.4f}" + ("  *" if error <= result["bound"] else ""))
    print(f"  without the event: {result['none']:.4f}")
    print(m.describe_busy_scan(result))
    print(f"build_model.py uses {scale:g} GB/s: {source}")
    print("Cross-check, dependent loads alone: " + chase_fit(power))


if __name__ == "__main__":
    main()
