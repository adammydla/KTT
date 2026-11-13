#!/usr/bin/env python3
"""The fixed power of steady DRAM traffic, from the dependent-load probes (--latency, and with --sm-sweep on fewer SMs).

Every chase row at a level (all SMs and SM-sweep rows, duty 100, random data) gives the power above its sleep baseline at
a steady access rate. This fits

    P - baseline = e x rate + F x (1 - exp(-rate / r0))

per level, next to the alternatives: a ramp F x min(1, rate / r0), a fixed power that does not depend on the rate, and
one per working SM. On an RTX 3090 (runs Instr and LowRate) DRAM gives e = 172 pJ/B, F = 13.4 W, r0 = 95 GB/s (rms 2.4 W);
r0 is BUSY_SCALE_GBS of energy_model.py (its --busy-scale), F is what the fit of the probes prices "DRAM busy ns" at.

    python3 tools/fit_busy.py LowRate[,Instr]     # the --output prefixes of runs with --latency (their .csv)
"""

import argparse
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_table  # noqa: E402


CLOCK_TOLERANCE_PCT = 2.0  # as ClockTolerance in Common.h


def chase_rows(prefixes):
    rows = []
    for prefix in prefixes:
        for row in counter_table.read_rows(prefix + ".csv"):
            # Not the rows that ran at another SM clock than the baseline (throttled below the lock; their energy is
            # withheld, but they still have a power and a baseline): they are not measurements at the baseline's clock.
            throttled = abs(counter_table.number(row.get("clock_difference_pct")) or 0.0) > CLOCK_TOLERANCE_PCT
            if row["operation"].startswith("chase") and row["payload"] == "random" and row["duty_pct"] == "100" \
                    and row["power_w"] and row["baseline_w"] and row["bandwidth_gbs"] and not throttled:
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", help="--output prefixes of runs with --latency, comma-separated")
    arguments = parser.parse_args()
    rows = chase_rows(arguments.runs.split(","))
    if not rows:
        raise SystemExit("no dependent-load rows (operation chase-*) with power in these runs; measure with --latency")

    shapes = {
        "exponential": (lambda r, e, f, r0: e * r + f * -np.expm1(-r / r0), (0.17, 10.0, 100.0)),
        "ramp": (lambda r, e, f, r0: e * r + f * np.minimum(1.0, r / r0), (0.17, 10.0, 150.0)),
    }
    for level in ("L2", "DRAM"):
        group = [r for r in rows if r["level"] == level]
        if len(group) < 4:
            continue
        rate = np.array([float(r["bandwidth_gbs"]) for r in group])
        dynamic = np.array([float(r["power_w"]) - float(r["baseline_w"]) for r in group])
        sms = np.array([82.0 if r["active_sms"] == "all" else float(r["active_sms"]) for r in group])
        if any(r["active_sms"] == "all" for r in group):
            # "all" SMs: take the largest SM count of the sweep rows if there are any, else leave the per-SM fit out.
            counts = [float(r["active_sms"]) for r in group if r["active_sms"] != "all"]
            sms[[r["active_sms"] == "all" for r in group]] = max(counts) if counts else np.nan
        print(f"{level}: {len(group)} rows, {rate.min():.1f}-{rate.max():.0f} GB/s, {dynamic.min():.1f}-{dynamic.max():.1f} W "
              "above the baseline")
        for name, (function, start) in shapes.items():
            try:
                p, covariance = curve_fit(function, rate, dynamic, p0=start, maxfev=20000)
            except RuntimeError:
                print(f"  {name:16} no fit")
                continue
            sd = np.sqrt(np.diag(covariance))
            rms = np.sqrt(np.mean((dynamic - function(rate, *p)) ** 2))
            if not np.all(np.isfinite(sd)) or sd[2] > p[2] or p[2] > rate.max():
                print(f"  {name:16} no fixed power identifiable (scale {p[2]:.0f} GB/s, beyond the measured rates or "
                      f"undetermined)   rms {rms:.2f} W")
                continue
            print(f"  {name:16} {1000 * p[0]:6.1f} pJ/B + {p[1]:5.2f} W fixed, scale {p[2]:7.1f} GB/s "
                  f"(sd {1000 * sd[0]:.1f}, {sd[1]:.2f}, {sd[2]:.1f})   rms {rms:.2f} W")
        a = np.c_[rate, np.ones_like(rate)]
        c = np.linalg.lstsq(a, dynamic, rcond=None)[0]
        print(f"  {'constant fixed':16} {1000 * c[0]:6.1f} pJ/B + {c[1]:5.2f} W fixed"
              f"{'':37}rms {np.sqrt(np.mean((dynamic - a @ c) ** 2)):.2f} W")
        if not np.isnan(sms).any():
            a = np.c_[rate, sms]
            c = np.linalg.lstsq(a, dynamic, rcond=None)[0]
            print(f"  {'per working SM':16} {1000 * c[0]:6.1f} pJ/B + {1000 * c[1]:5.1f} mW per SM"
                  f"{'':34}rms {np.sqrt(np.mean((dynamic - a @ c) ** 2)):.2f} W")
    print("The model's DRAM busy event uses the exponential shape's scale (energy_model.py --busy-scale, default 95 GB/s "
          "from an RTX 3090).\nRows only a few watts above the baseline are where power readings are sublinear "
          "(README, known limitations).")


if __name__ == "__main__":
    main()
