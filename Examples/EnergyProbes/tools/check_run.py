#!/usr/bin/env python3
"""Checks of a probe run against what the probes are designed to do: the validation checklist of README.md as tests.

    python3 tools/check_run.py RUN [--counters PROFILED_RUN]

RUN is the --output prefix of a probe run (RUN.csv, RUN_fits.csv), PROFILED_RUN that of a run with --profile (its CUPTI
counts per row). Every check prints PASS, WARN or FAIL with its numbers; the exit code is 1 if a check fails.

These checks cannot show that an energy is the true energy of a component (NVML measures the whole board). They show
that the measurement is sound (clock, temperature, linearity, placement), that every probe exercises the component it is
named after (counters), and that the energies relate as the probes' construction implies (ordering of levels, equal
paths, payload, widths, conflict degrees). Whether the energies add up to the energy of other kernels is checked by
energy_model.py on a --validate run; repeatability by compare_runs.py.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_table  # noqa: E402

CLOCK_TOLERANCE_PCT = 2.0  # as ClockTolerance in Common.h
SAME_PATH_TOLERANCE = 0.10  # probes that take the same path must agree within this
ORDER_TOLERANCE = 0.05     # a "not less than" may miss by this much (noise), at least
NOISE_FACTOR = 3.0         # a fit's residual counts as more than noise above this many times the baseline noise
NOISE_LIMIT = 0.25         # an order check whose noise allows more than this says nothing (WARN)


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def read(path):
    return counter_table.read_rows(path)


class Report:
    def __init__(self):
        self.counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    def add(self, status, name, detail):
        self.counts[status] += 1
        print(f"{status:4}  {name:34} {detail}")

    def check(self, passed, name, detail, failure="FAIL"):
        self.add("PASS" if passed else failure, name, detail)


def r_blocks(row):
    """BLOCKS_PER_SM of a row's note "registers: N per thread, M of K blocks fit per SM"."""
    note = row["note"].split("registers:", 1)[1]
    return note.split(" of ", 1)[1].split()[0] if " of " in note else "?"


def is_probe(row):
    return row["level"] not in ("idle", "baseline", "reference")


def full_rate(rows):
    """Rows at duty 100 on random data and all SMs, by (level, operation, width)."""
    return {(r["level"], r["operation"], int(r["width_bytes"])): r for r in rows
            if is_probe(r) and r["duty_pct"] == "100" and r["payload"] == "random" and r["active_sms"] == "all"}


def fit_table(fits, payload="random"):
    return {(f["level"], f["operation"], int(f["width_bytes"])): f for f in fits if f["payload"] == payload}


def energy(fits, level, operation, width):
    fit = fits.get((level, operation, width))
    return number(fit["pj_per_access"]) if fit else math.nan


# --- the measurement ---------------------------------------------------------------------------------------------------

def baseline_noise(rows):
    """The power noise of the run (W): the sd of the sleep baselines around their temperature line (around their mean
    if they span less than 2 C). On an RTX 3090 0.14-0.53 W, on an RTX 2080 Ti 0.8-0.9 W."""
    sleeps = [r for r in rows if r["probe"].startswith("sleep-") and not r["probe"].startswith("sleep-sm")]
    points = [(number(r["temperature_c"]), number(r["power_w"])) for r in sleeps]
    points = [(t, p) for t, p in points if not (math.isnan(t) or math.isnan(p))]
    if len(points) < 3:
        return math.nan
    mean_t = sum(t for t, _ in points) / len(points)
    mean_p = sum(p for _, p in points) / len(points)
    sxx = sum((t - mean_t) ** 2 for t, _ in points)
    span = max(t for t, _ in points) - min(t for t, _ in points)
    slope = sum((t - mean_t) * (p - mean_p) for t, p in points) / sxx if sxx > 0 and span >= 2.0 else 0.0
    residual = sum((p - mean_p - slope * (t - mean_t)) ** 2 for t, p in points)
    return math.sqrt(residual / (len(points) - 2))


def fit_rows(rows, fit):
    """The rows a fit of the run's _fits.csv went through (clock-matched rows of its duties)."""
    duties = fit["duties"].split("/")
    return [r for r in rows if r["operation"] == fit["operation"] and r["level"] == fit["level"]
            and r["width_bytes"] == fit["width_bytes"] and r["payload"] == fit["payload"] and r["active_sms"] == "all"
            and r["footprint_bytes"] == fit["footprint_bytes"] and r["duty_pct"] in duties
            and not math.isnan(number(r["pj_per_access"]))]


def fit_residual(rows, fit):
    """The sd of a fit's rows around its line (W), from its R2 and the spread of their power above the baseline."""
    y = [number(r["power_w"]) - number(r["baseline_w"]) for r in fit_rows(rows, fit)]
    r2 = number(fit["r2"])
    if len(y) < 3 or math.isnan(r2):
        return math.nan
    mean = sum(y) / len(y)
    return math.sqrt(max(1 - r2, 0.0) * sum((v - mean) ** 2 for v in y) / (len(y) - 2))


def slope_error(rows, fit, noise):
    """The relative error of a fit's energy per access that the baseline noise alone gives (sd of the slope / slope)."""
    x = [number(r["accesses_per_launch"]) / number(r["duration_s"]) for r in fit_rows(rows, fit)]
    energy = number(fit["pj_per_access"]) * 1e-12
    if len(x) < 2 or math.isnan(noise) or not energy > 0:
        return math.nan
    mean = sum(x) / len(x)
    sxx = sum((v - mean) ** 2 for v in x)
    return noise / math.sqrt(sxx) / energy if sxx > 0 else math.nan


def check_measurement(rows, report, noise):
    probes = [r for r in rows if is_probe(r)]
    invalid = [r for r in probes if r["valid"] != "1" or "run failed" in r["note"]]
    report.check(not invalid, "every configuration ran", f"{len(probes) - len(invalid)} of {len(probes)} probe rows valid")

    withheld = [r for r in probes if math.isnan(number(r["pj_per_access"]))]
    report.check(not withheld, "every row has an energy", f"{len(withheld)} withheld"
                 + (": " + ", ".join(sorted({f"{r['operation']} {r['level']}" for r in withheld}))[:120] if withheld else ""),
                 failure="WARN")

    clocks = [abs(number(r["clock_difference_pct"])) for r in probes if not math.isnan(number(r["clock_difference_pct"]))]
    worst = max(clocks) if clocks else math.nan
    report.check(clocks and worst <= CLOCK_TOLERANCE_PCT, "SM clock of every row = baseline's",
                 f"{len(clocks)} rows, largest difference {worst:.2f} % (limit {CLOCK_TOLERANCE_PCT} %)")
    limited = [r for r in probes if r["clock_limiters"] not in ("", "none")]
    report.check(not limited, "no clock limiter active", f"{len(limited)} rows with limiters"
                 + (f" ({limited[0]['clock_limiters']})" if limited else ""), failure="WARN")

    sleeps = [r for r in rows if r["probe"].startswith("sleep-") and not r["probe"].startswith("sleep-sm")]
    powers = [number(r["power_w"]) for r in sleeps]
    temperatures = [number(r["temperature_c"]) for r in sleeps]
    if powers:
        # They agree if their spread is small, or if the temperature line explains it (noise: their sd around it).
        spread = (max(powers) - min(powers)) / (sum(powers) / len(powers))
        explained = not math.isnan(noise) and 3 * noise / (sum(powers) / len(powers)) <= 0.03
        report.check(spread <= 0.03 or explained, "sleep baselines agree", f"{len(powers)} baselines "
                     f"{min(powers):.1f}-{max(powers):.1f} W ({100 * spread:.1f} %), "
                     f"{min(temperatures):.0f}-{max(temperatures):.0f} C; sd around the temperature line {noise:.2f} W",
                     failure="WARN")
        check_baseline_temperature(probes, powers, temperatures, report)

    placements = [r for r in probes if "placement:" in r["note"] and "placement: ok" not in r["note"]]
    registers = [r for r in placements if "registers:" in r["note"]]
    report.check(not placements, "blocks placed one stripe each", f"{len(placements)} rows with a placement problem"
                 + (f", {len(registers)} of them with more registers than {r_blocks(registers[0])} blocks per SM allow ("
                    + ", ".join(sorted({f"{r['operation']} w{r['width_bytes']}: {r.get('registers', '?')}" for r in registers}))
                    + " registers per thread)" if registers else ""))
    # Kernels whose registers let fewer blocks than BLOCKS_PER_SM run at once on an SM (runs since 2026-11-06).
    short = sorted({(r["probe"], r["operation"], r["width_bytes"], r.get("registers"), r.get("blocks_per_sm_fit")) for r in probes
                    if "registers:" in r.get("note", "")})
    if "blocks_per_sm_fit" in rows[0]:
        report.check(not short, "kernels fit their blocks per SM", f"{len(short)} kernels with fewer"
                     + (": " + ", ".join(f"{p} {o} w{w}: {g} registers, {b} blocks" for p, o, w, g, b in short[:5])
                        if short else ""), failure="WARN")

    # KTT ends a configuration at --max-time-ms when its power reading has not settled within --max-power-diff (runs
    # since 2026-11-06 record it). Those rows are measured, but less precisely; on an RTX 2080 Ti 266 of 816 were.
    judged = [r for r in rows if r.get("power_settled") in ("0", "1")]
    if judged:
        unsettled = [r for r in judged if r["power_settled"] == "0"]
        by_duty = {}
        for r in unsettled:
            by_duty[r["duty_pct"] or "-"] = by_duty.get(r["duty_pct"] or "-", 0) + 1
        spreads = [number(r["power_spread_pct"]) for r in unsettled if not math.isnan(number(r["power_spread_pct"]))]
        report.check(len(unsettled) <= 0.02 * len(judged), "KTT's power reading settled", f"{len(judged) - len(unsettled)} "
                     f"of {len(judged)} configurations" + (f"; not settled: " + ", ".join(f"duty {d}: {n}" for d, n in
                     sorted(by_duty.items())) + (f", spread up to {max(spreads):.1f} %" if spreads else "")
                     if unsettled else ""), failure="WARN")

    # NVML's energy counter and the sampled power measure different things (the counter misses some rails on an RTX
    # 3090), but their ratio must be the same in every row, or one of them is not a measurement of the probe.
    ratios = [number(r["counter_power_w"]) / number(r["power_w"]) for r in probes
              if number(r["power_w"]) > 0 and number(r["counter_power_w"]) > 0]
    if ratios:
        mean = sum(ratios) / len(ratios)
        cv = math.sqrt(sum((x - mean) ** 2 for x in ratios) / len(ratios)) / mean
        report.check(cv <= 0.05, "NVML counter / sampled power steady", f"ratio {mean:.3f}, CV {100 * cv:.1f} % over "
                     f"{len(ratios)} rows", failure="WARN")


def check_baseline_temperature(probes, powers, temperatures, report):
    # The analysis moves the sleep baseline to each row's temperature only when the sleeps span 2 C (FitBaseline in
    # Analysis.cpp); otherwise it uses their mean. Rows warmer than the sleeps then carry the leakage of the extra degrees
    # as dynamic power, and duty rows at different temperatures no longer lie on one line.
    hottest = max((number(r["temperature_c"]) for r in probes if not math.isnan(number(r["temperature_c"]))), default=math.nan)
    n = len(temperatures)
    mean_t = sum(temperatures) / n
    mean_p = sum(powers) / n
    sxx = sum((t - mean_t) ** 2 for t in temperatures)
    if n < 3 or sxx == 0:
        report.add("WARN", "baseline follows temperature", f"{n} sleeps at one temperature, no slope")
        return
    slope = sum((t - mean_t) * (p - mean_p) for t, p in zip(temperatures, powers)) / sxx
    residual = sum((p - mean_p - slope * (t - mean_t)) ** 2 for t, p in zip(temperatures, powers))
    error = math.sqrt(residual / (n - 2) / sxx)
    span = max(temperatures) - min(temperatures)
    detail = (f"sleeps {min(temperatures):.1f}-{max(temperatures):.1f} C, slope {slope:.2f} +- {error:.2f} W/C; hottest row "
              f"{hottest:.1f} C")
    # The temperature model is a line through the sleeps; beyond their range it is extrapolated, with an uncertainty of
    # the slope's error times the distance. Since 2026-11-06 a sleep follows every probe group, so that the sleeps run at
    # the temperatures the probes leave behind (on an RTX 3090: 14 sleeps over 64-71.5 C, 0.29 +- 0.02 W/C, against
    # 0.33 +- 0.10 W/C from 5 sleeps over 64-67.4 C).
    # (The SM sweep has baselines of its own next to its rows.)
    row_temperatures = [number(r["temperature_c"]) for r in probes
                        if r["active_sms"] == "all" and not math.isnan(number(r["temperature_c"]))]
    if row_temperatures and span >= 2.0:
        beyond = [t for t in row_temperatures if t > max(temperatures) + 2.0 or t < min(temperatures) - 2.0]
        distance = max(max(row_temperatures) - max(temperatures), min(temperatures) - min(row_temperatures), 0.0)
        report.check(error * distance <= 0.5, "sleeps cover the probes' temperatures",
                     f"rows {min(row_temperatures):.1f}-{max(row_temperatures):.1f} C, sleeps "
                     f"{min(temperatures):.1f}-{max(temperatures):.1f} C ({len(beyond)} of {len(row_temperatures)} rows more "
                     f"than 2 C outside); baseline at the farthest row +- {error * distance:.2f} W (limit 0.5 W)",
                     failure="WARN")
    if span >= 2.0:
        report.add("PASS", "baseline follows temperature", detail)
    elif error > 0 and abs(slope) / error >= 3:
        report.add("WARN", "baseline follows temperature", f"flat (sleeps span {span:.1f} C < 2 C) although {detail}, "
                   f"{slope * (hottest - mean_t):+.1f} W at that slope")
    else:
        report.add("PASS", "baseline follows temperature", f"flat, no clear slope: {detail}")


def check_fits(rows, fits, report, noise):
    # A low R2 counts only if the rows stray from the line by more than NOISE_FACTOR times the baseline noise: a probe
    # only a few watts above the baseline can have a low R2 from noise alone (on an RTX 2080 Ti, R2 0.68 with a residual
    # of 0.5 W, half the noise). On an RTX 3090 every fit below 0.99 strays by 6-48 times the noise.
    random_fits = [f for f in fits if f["payload"] == "random"]
    r2 = [(number(f["r2"]), f, fit_residual(rows, f)) for f in random_fits if not math.isnan(number(f["r2"]))]
    within = [(v, f, e) for v, f, e in r2 if v < 0.99 and not math.isnan(noise) and e <= NOISE_FACTOR * noise]
    low = [(v, f, e) for v, f, e in r2 if v < 0.99 and (v, f, e) not in within]
    report.check(not [1 for v, _, _ in low if v < 0.95], "power linear in access rate",
                 f"{len(r2)} fits, R2 >= 0.99 in {len(r2) - len(low) - len(within)}"
                 + (f", {len(within)} more within {NOISE_FACTOR:.0f}x the noise ({noise:.2f} W)" if within else "")
                 + (", lower: " + ", ".join(f"{f['operation']} {f['level']} w{f['width_bytes']} {v:.4f} ({e:.1f} W)"
                                            for v, f, e in sorted(low, key=lambda x: x[0])[:6]) if low else ""),
                 failure="FAIL")
    if low and not [1 for v, _, _ in low if v < 0.95]:
        report.add("WARN", "fits with 0.95 <= R2 < 0.99", f"{len(low)}")

    full = full_rate(rows)
    nonlinear = []
    for f in random_fits:
        row = full.get((f["level"], f["operation"], int(f["width_bytes"])))
        if row and abs(number(f["fixed_w"])) > 0.5 * (number(row["power_w"]) - number(row["baseline_w"])):
            nonlinear.append(f"{f['operation']} {f['level']} w{f['width_bytes']}")
    report.check(len(nonlinear) <= 2, "fixed power below half the dynamic", f"{len(nonlinear)} fits above (the summary uses "
                 f"their duty-100 row): {', '.join(nonlinear)}" if nonlinear else "all fits", failure="WARN")

    negative = [f for f in fits if number(f["pj_per_access"]) <= 0]
    report.check(not negative, "every energy above the baseline", f"{len(negative)} fits at or below 0"
                 + (": " + ", ".join(f"{f['operation']} {f['level']}" for f in negative[:5]) if negative else ""))


# --- relations the probes' construction implies ------------------------------------------------------------------------

def check_relations(rows, fits_list, report, noise):
    fits = fit_table(fits_list)
    full = full_rate(rows)

    # Deeper levels cost more and deliver less bandwidth (same SASS apart from the footprint).
    for operation in ("ld", "ld.cg", "ld.nc", "st", "tex", "red", "atom"):
        for width in (4, 8, 16):
            levels = [(lv, energy(fits, lv, operation, width), full.get((lv, operation, width)))
                      for lv in ("L1", "L2", "DRAM")]
            levels = [(lv, e, r) for lv, e, r in levels if not math.isnan(e)]
            if len(levels) < 2:
                continue
            ordered = all(a[1] < b[1] for a, b in zip(levels, levels[1:]))
            bandwidths = [number(r["bandwidth_gbs"]) if r else math.nan for _, _, r in levels]
            slower = all(a > b for a, b in zip(bandwidths, bandwidths[1:]) if not (math.isnan(a) or math.isnan(b)))
            report.check(ordered and slower, f"{operation} w{width}: L1 < L2 < DRAM",
                         " < ".join(f"{lv} {e:.0f} pJ" for lv, e, _ in levels) + "; GB/s "
                         + " > ".join(f"{b:.0f}" for b in bandwidths))

    # Probes that take the same path must agree.
    same = [(("L2", "ld.cg", 16), ("L2", "ld", 16), "ld.cg = ld at L2 (both skip or miss L1)"),
            (("DRAM", "ld.cg", 16), ("DRAM", "ld", 16), "ld.cg = ld at DRAM"),
            (("L1", "ld.ca", 16), ("L1", "ld", 16), "ld.ca = ld at L1 (default caches in L1)"),
            (("L1", "ld.nc", 16), ("L1", "ld", 16), "ld.nc = ld at L1 (unified L1/tex)"),
            (("L2", "ld.nc", 16), ("L2", "ld", 16), "ld.nc = ld at L2"),
            (("DRAM", "ld-stream", 16), ("DRAM", "ld", 16), "locality stream = global ld at DRAM"),
            (("DRAM", "st-stream", 16), ("DRAM", "st", 16), "locality stream = global st at DRAM"),
            (("L2", "ld-stream", 16), ("L2", "ld", 16), "locality stream = global ld at L2")]
    for a, b, label in same:
        ea, eb = energy(fits, *a), energy(fits, *b)
        if math.isnan(ea) or math.isnan(eb):
            continue
        difference = ea / eb - 1
        report.check(abs(difference) <= SAME_PATH_TOLERANCE, label, f"{ea:.1f} vs {eb:.1f} pJ ({100 * difference:+.1f} %)",
                     failure="WARN")

    # Shared memory and L1 are one SRAM; an L1 hit adds the tag lookup and the global address path.
    for operation, width in (("ld.shared", 16), ("ld.shared-1way", 4), ("ld.shared-1way", 8), ("ld.shared-1way", 16)):
        shared, hit = energy(fits, "shared", operation, width), energy(fits, "L1", "ld", width)
        if not (math.isnan(shared) or math.isnan(hit)):
            report.check(shared <= hit * (1 + ORDER_TOLERANCE), f"{operation} w{width} <= L1 hit (no tag lookup)",
                         f"{shared:.1f} vs {hit:.1f} pJ", failure="WARN")

    # Zeros switch fewer bits than random data: never more energy; the no-load loop moves no data at all.
    zero = fit_table(fits_list, "zero")
    above = [(k, energy(zero, *k), energy(fits, *k)) for k in zero if energy(zero, *k) > energy(fits, *k) * (1 + ORDER_TOLERANCE)]
    report.check(not above, "zeros cost no more than random data", f"{len(zero)} probes with both payloads"
                 + (", more on zeros: " + ", ".join(f"{k[1]} {k[0]} w{k[2]} {z:.1f} vs {r:.1f}" for k, z, r in above[:4])
                    if above else ""), failure="WARN")
    for width in (4, 8, 16):
        r, z = energy(fits, "none", "no-load", width), energy(zero, "none", "no-load", width)
        if not (math.isnan(r) or math.isnan(z)):
            report.check(abs(r - z) / r <= 0.05, f"no-load w{width}: datapath share 0", f"random {r:.2f}, zeros {z:.2f} pJ "
                         f"({100 * (r - z) / r:+.1f} %)", failure="WARN")

    # Wider accesses carry the per-request cost over more bytes: pJ per byte does not grow with the width.
    for level in ("L1", "L2", "DRAM"):
        per_byte = [(w, energy(fits, level, "ld", w) / w) for w in (4, 8, 16) if not math.isnan(energy(fits, level, "ld", w))]
        if len(per_byte) > 1:
            report.check(all(a[1] >= b[1] * (1 - ORDER_TOLERANCE) for a, b in zip(per_byte, per_byte[1:])),
                         f"ld {level}: pJ/B falls with width", ", ".join(f"w{w} {e:.2f}" for w, e in per_byte), failure="WARN")

    # More work per access never costs less: wavefronts of a conflict, L2 requests of scattered sectors, guarded moves.
    for operation in ("ld.shared", "st.shared"):
        for width in (4, 8, 16):
            keys = [(ways, ("shared", f"{operation}-{ways}way", width)) for ways in (1, 2, 4, 8, 16, 32)]
            series = [(w, energy(fits, *k), fits.get(k)) for w, k in keys]
            series = [(w, e, f) for w, e, f in series if not math.isnan(e)]
            # The tolerance grows with the noise of the two energies (twice their combined error, at least 5 %).
            def tolerance(a, b):
                errors = [slope_error(rows, f, noise) for f in (a[2], b[2])]
                errors = [e for e in errors if not math.isnan(e)]
                return max(ORDER_TOLERANCE, 2 * math.sqrt(sum(e * e for e in errors))) if errors else ORDER_TOLERANCE
            if len(series) > 1:
                pairs = list(zip(series, series[1:]))
                strict = all(b[1] >= a[1] * (1 - ORDER_TOLERANCE) for a, b in pairs)
                # A step down that the noise of the two energies allows is not a failure; but where the noise allows
                # more than NOISE_LIMIT, the energies are too noisy for the check to say anything (on an RTX 2080 Ti the
                # conflicting shared loads are 1-10 W above a baseline with 0.9 W of noise).
                misses = [(a, b) for a, b in pairs if b[1] < a[1] * (1 - ORDER_TOLERANCE)]
                needed = max((tolerance(a, b) for a, b in misses), default=ORDER_TOLERANCE)
                explained = not strict and all(b[1] >= a[1] * (1 - tolerance(a, b)) for a, b in misses)
                report.check(strict or (explained and needed <= NOISE_LIMIT), f"{operation} w{width}: rises with conflicts",
                             ", ".join(f"{w}x {e:.0f}" for w, e, _ in series)
                             + ("" if strict else f" (within the noise, {100 * needed:.0f} %)" if explained and
                                needed <= NOISE_LIMIT else f" (too noisy to check: the noise allows {100 * needed:.0f} %)"
                                if explained else f" (the noise allows {100 * needed:.0f} %)"),
                             failure="WARN")
    for level, operation in (("DRAM", "ld"), ("DRAM", "st")):
        for width in (4, 8):
            adjacent = energy(fits, level, f"{operation}-sub-adjacent", width)
            scattered = energy(fits, level, f"{operation}-sub-scattered", width)
            if not (math.isnan(adjacent) or math.isnan(scattered)):
                report.check(scattered >= adjacent, f"{operation}-sub w{width}: scattered >= adjacent",
                             f"{scattered:.0f} vs {adjacent:.0f} pJ", failure="WARN")
    static, indexed = energy(fits, "compute", "ffma-static", 0), energy(fits, "compute", "ffma-indexed", 0)
    if not (math.isnan(static) or math.isnan(indexed)):
        report.check(indexed > static, "runtime index costs more than static", f"{indexed:.1f} vs {static:.1f} pJ")


# --- counters: does every probe exercise its component? ----------------------------------------------------------------

def check_counters(counter_rows, report):
    # NaN if missing; aliases of other chips count. A counter that the table does not have (no run collects it) is NaN.
    get = lambda r, name: counter_table.value_of(r, name) if name in counter_table.ENTRIES else math.nan
    rows = full_rate(counter_rows)
    l1_rows, l2_rows, dram_rows, bad = 0, 0, 0, []
    instructions = {}

    for (level, operation, width), r in rows.items():
        if r["probe"] not in ("global", "texture") or width == 0:
            continue
        requested = number(r["accesses_per_launch"]) * width / 32
        l2 = sum(get(r, n) for n in ("lts__t_sectors_op_read.sum", "lts__t_sectors_op_write.sum",
                                     "lts__t_sectors_srcunit_tex_op_atom.sum", "lts__t_sectors_srcunit_tex_op_red.sum")
                 if not math.isnan(get(r, n))) / requested
        dram = (get(r, "dram__sectors_read.sum") + get(r, "dram__sectors_write.sum")) / requested
        atomic = operation in ("red", "atom")
        if level == "L1":
            l1_rows += 1
            if get(r, "l1tex__t_sector_hit_rate.pct") < 90 or l2 > 0.1:
                bad.append(f"{operation} L1 w{width}: L1 hit {get(r, 'l1tex__t_sector_hit_rate.pct'):.0f} %, L2 {l2:.2f}")
        elif level == "L2":
            l2_rows += 1
            if not 0.9 <= l2 <= 1.2 or dram > 0.05:
                bad.append(f"{operation} L2 w{width}: L2 {l2:.2f}, DRAM {dram:.2f}")
        elif level == "DRAM":
            dram_rows += 1
            expected = 2.0 if atomic else 1.0
            if not 0.9 * expected <= dram <= 1.2 * expected:
                bad.append(f"{operation} DRAM w{width}: DRAM {dram:.2f} (expected {expected:.0f})")
        if operation == "ld" and r["probe"] == "global":
            instructions.setdefault(width, []).append((level, get(r, "smsp__inst_executed.sum") / (number(r["accesses_per_launch"]) / 32)))
    report.check(not bad, "probes hit their level (counters)", f"{l1_rows} L1, {l2_rows} L2, {dram_rows} DRAM rows"
                 + ("; off: " + "; ".join(bad[:4]) if bad else ""))

    # The locality, write and mixed probes walk the footprints in their own ways (probeScatter by a counter over the grid,
    # probeWrite two streams or in place, probeMixed as the global probe): the DRAM share of their L2 sectors must be that
    # of their level, and an L2 row must not be served by L1 (on a GPU whose SM count has a large power of two, the
    # locality probe's L2 control fits in L1: the program warns when it prints the setup). On an RTX 3090 the L2 rows of
    # the locality probe have 2-11 % L1 hits, its DRAM rows 97-100 % of their L2 sectors from DRAM. The sub-sector probe
    # hits L1 by design (the later words of a sector) and is left out.
    off, checked = [], 0
    for (level, operation, width), r in rows.items():
        if r["probe"] not in ("locality", "write", "mixed") or width == 0 or operation == "local":
            continue
        l2 = sum(get(r, n) for n in ("lts__t_sectors_srcunit_tex_op_read.sum", "lts__t_sectors_srcunit_tex_op_write.sum")
                 if not math.isnan(get(r, n)))
        dram = sum(get(r, n) for n in ("dram__sectors_read.sum", "dram__sectors_write.sum") if not math.isnan(get(r, n)))
        l1_hits = get(r, "l1tex__t_sector_hit_rate.pct")
        requested = number(r["accesses_per_launch"]) * width / 32
        checked += 1
        if level == "L1":
            wrong = l2 > 0.1 * requested
        elif level == "L2":
            wrong = l2 < 0.7 * requested or dram > 0.05 * l2 or (r["probe"] != "write" and l1_hits > 30)
        else:
            wrong = l2 < 0.7 * requested or dram < 0.85 * l2
        if wrong:
            off.append(f"{operation} {level} w{width}: L1 hit {l1_hits:.0f} %, {l2 / requested:.2f} L2 sectors per requested "
                       f"sector, {dram / l2 if l2 else math.nan:.2f} of them from DRAM")
    if checked:
        report.check(not off, "locality, write, mixed hit their level", f"{checked} rows" + ("; off: " + "; ".join(off[:4])
                                                                                               if off else ""))

    for width, values in sorted(instructions.items()):
        low, high = min(v for _, v in values), max(v for _, v in values)
        report.check(high <= low * 1.03, f"ld w{width}: same instructions per level", ", ".join(f"{lv} {v:.2f}" for lv, v in values))

    # Shared memory: wavefronts per warp instruction as designed (probeShared 16 bytes: 4; red.shared: 1; the conflict
    # probe: ways x width / 4), and the conflict counter = the extra ones.
    off, checked = [], 0
    for (level, operation, width), r in rows.items():
        if level != "shared":
            continue
        kind = "atom" if operation.startswith("red") else ("st" if operation.startswith("st") else "ld")
        instructions_per_launch = number(r["accesses_per_launch"]) / 32
        wavefronts = get(r, f"l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{kind}.sum") / instructions_per_launch
        ways = int(operation.split("-")[1][:-3]) if "way" in operation else 1
        expected = ways * width / 4
        conflicts = get(r, f"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{kind}.sum") / instructions_per_launch
        checked += 1
        if abs(wavefronts / expected - 1) > 0.05 or (not math.isnan(conflicts) and abs(conflicts - (expected - width / 4)) > 0.1):
            off.append(f"{operation} w{width}: {wavefronts:.2f} wavefronts (expected {expected:.0f}), {conflicts:.2f} conflicts")
    if checked:
        report.check(not off, "shared wavefronts as designed", f"{checked} shared rows" + ("; off: " + "; ".join(off[:4]) if off else ""))

    for (level, operation, width), r in rows.items():
        if operation != "ld.const":
            continue
        requests = get(r, "idc__requests.sum")
        per_access = requests / (number(r["accesses_per_launch"]) / 32)
        to_l15 = get(r, "idc__requests_lookup_miss_tag_miss.sum") / requests
        expected = level == "const-L1"
        report.check(abs(per_access - 1) <= 0.05 and (to_l15 <= 0.05 if expected else to_l15 >= 0.8),
                     f"constant probe {level}", f"{per_access:.3f} requests per warp access, {100 * to_l15:.1f} % to L1.5")

    # Latency probes: every step of a chain is one coalesced line (4 sectors per L2 read request), and the lines hit their
    # level. Random lines are sometimes still in the cache above by chance: on an RTX 3090, 5-12 % of the L2 probes' loads
    # hit L1 (most with 32 working warps per SM) and 2-8 % of the DRAM probes' lines are in L2. The model counts the
    # actual events, so this only shifts a probe's mix, but a probe with more than 15 % fails.
    off, checked = [], 0
    for (level, operation, width), r in rows.items():
        if not operation.startswith("chase"):
            continue
        warp_accesses = number(r["accesses_per_launch"]) / 32
        requests = get(r, "lts__t_requests_srcunit_tex_op_read.sum") / warp_accesses
        sectors = get(r, "lts__t_sectors_srcunit_tex_op_read.sum") / warp_accesses
        dram = get(r, "dram__sectors_read.sum") / warp_accesses
        checked += 1
        if not 0.85 <= requests <= 1.02 or abs(sectors / requests - 4) > 0.2 or (dram < 0.85 * sectors if level == "DRAM"
                                                                                  else dram > 0.05 * sectors):
            off.append(f"{level} {operation}: {requests:.2f} requests, {sectors:.2f} L2 and {dram:.2f} DRAM sectors per "
                       "warp access")
    if checked:
        report.check(not off, "latency probes: one line per step, at their level",
                     f"{checked} rows" + ("; off: " + "; ".join(off[:4]) if off else ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="--output prefix of a probe run")
    parser.add_argument("--counters", help="--output prefix of a run with --profile")
    arguments = parser.parse_args()

    rows = read(arguments.run + ".csv")
    fits = read(arguments.run + "_fits.csv")
    report = Report()
    noise = baseline_noise(rows)
    print(f"== {arguments.run}: measurement (power noise {noise:.2f} W: the sleeps' sd around their temperature line)")
    check_measurement(rows, report, noise)
    check_fits(rows, fits, report, noise)
    print(f"== {arguments.run}: relations")
    check_relations(rows, fits, report, noise)
    if arguments.counters:
        print(f"== {arguments.counters}: counters")
        check_counters(read(arguments.counters + ".csv"), report)
    print(f"{report.counts['PASS']} passed, {report.counts['WARN']} warnings, {report.counts['FAIL']} failed")
    sys.exit(1 if report.counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
