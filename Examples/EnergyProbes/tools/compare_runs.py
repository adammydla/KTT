#!/usr/bin/env python3
"""Repeatability of EnergyProbes runs.

Reads <prefix>_energy_table.json and <prefix>.csv of two or more runs made with the same options and prints, for every
energy, its value in each run, the mean, the coefficient of variation (sample standard deviation / mean) and the range
(max - min) / mean. The SM sweep is recomputed from the CSV, with the sleep baseline on the same SMs moved to the probe's
temperature, so runs made before that correction existed compare the same way.

    python3 tools/compare_runs.py Study Study2 Study3 [--csv repeatability.csv]
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_table  # noqa: E402


def number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return math.nan
    return value


def load(prefix):
    with open(prefix + "_energy_table.json") as file:
        table = json.load(file)
    return table, counter_table.read_rows(prefix + ".csv")


def fit_line(points):
    n = len(points)
    if n < 2:
        return math.nan, math.nan
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    if sxx == 0.0:
        return math.nan, math.nan
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sxx
    return mean_y - slope * mean_x, slope


def metrics(table, rows):
    """All compared values of one run as {(section, name): value}."""
    values = {}

    def put(section, name, value):
        values[(section, name)] = number(value)

    put("power", "constant (idle, no context) [W]", table.get("constant_w"))
    put("power", "sleep baseline [W]", table["baseline"]["mean_w"])
    put("power", "static (baseline - constant) [W]", table.get("static_w"))
    put("power", "baseline slope [W/C]", table["baseline"]["slope_w_per_c"])
    put("power", "baseline temperature [C]", table["baseline"]["mean_temperature_c"])

    for fit in table["fits"]:
        width = fit["width_bytes"]
        label = f'{fit["level"]} {fit["operation"]}' + (f" w{width}" if width else "")
        if fit["level"] == "compute":
            put(f'fit pJ/op, {fit["payload"]}', label, fit["pj_per_access"])
        elif not width:  # constant memory: a warp reads one word, so energies are per access
            put(f'fit pJ/access, {fit["payload"]}', label, fit["pj_per_access"])
        else:
            put(f'fit pJ/B, {fit["payload"]}', label, fit["pj_per_byte"])

    for entry in table.get("compute", []):
        put("compute pJ/FLOP", entry["operation"], entry["pj_per_flop"])
    put("compute pJ/FLOP", "loop overhead [pJ/op]", table.get("compute_loop_pj_per_op"))

    for entry in table.get("atomics", []):
        put("atomics pJ/access", entry["probe"], entry["pj_per_access"])
        if entry["reference"]:
            put("atomics pJ/access", f'{entry["probe"]} above {entry["reference"]}', entry["extra_pj_per_access"])

    for entry in table.get("constant", []):
        put("constant pJ/warp request", entry["level"], entry["pj_per_warp_instruction"])
    if "constant_l1_miss_pj_per_request" in table:
        put("constant pJ/warp request", "L1 miss (L1.5 - L1)", table["constant_l1_miss_pj_per_request"])

    for entry in table.get("control_datapath", []):
        total = number(entry["total"])
        share = number(entry["datapath"]) / total if total else math.nan
        put("datapath share", entry["probe"], share)

    for entry in table.get("width_model", []):
        put("width model", f'{entry["level"]} pJ/request', entry["pj_per_request"])
        put("width model", f'{entry["level"]} pJ/B', entry["pj_per_byte"])

    # Duty-100 rows on all SMs (random data, 16-byte accesses for memory probes): the measured energies the fits are
    # anchored on, without any fitting.
    for row in rows:
        if row["active_sms"] != "all" or row["duty_pct"] != "100" or row["payload"] != "random":
            continue
        if row["level"] == "compute":
            put("duty 100 pJ/op", row["operation"], row["pj_per_access"])
        elif row["width_bytes"] == "16":
            put("duty 100 pJ/B", f'{row["level"]} {row["operation"]}', row["pj_per_byte"])

    # NVML energy counter against KTT power on the sleep baselines on all SMs.
    ratios = [number(r["counter_power_w"]) / number(r["power_w"]) for r in rows
              if r["level"] == "baseline" and r["active_sms"] == "all" and number(r["power_w"]) > 0.0]
    ratios = [r for r in ratios if not math.isnan(r)]
    if ratios:
        put("power", "energy counter / KTT power at baseline", statistics.mean(ratios))

    # SM sweep: probe minus the sleep baseline on the same SMs, moved to the probe's temperature.
    slope = number(table["baseline"]["slope_w_per_c"])
    sleeps = {r["active_sms"]: r for r in rows if r["level"] == "baseline" and r["active_sms"] != "all"}
    lines = {}
    for row in rows:
        count = row["active_sms"]
        if row["level"] == "baseline" or count == "all" or count not in sleeps:
            continue
        sleep = sleeps[count]
        baseline = number(sleep["power_w"]) + slope * (number(row["temperature_c"]) - number(sleep["temperature_c"]))
        dynamic = number(row["power_w"]) - baseline
        accesses = number(row["accesses_per_launch"])
        label = f'{row["level"]} {row["operation"]}'
        if accesses > 0.0:
            put("SM sweep pJ/access", f"{label}, {int(count):2d} SMs", dynamic * number(row["duration_s"]) / accesses * 1e12)
        lines.setdefault(label, []).append((int(count), dynamic))
    for label, points in lines.items():
        intercept, per_sm = fit_line(points)
        put("SM sweep line", f"{label} W per SM", per_sm)
        put("SM sweep line", f"{label} intercept W", intercept)
    intercept, per_sm = fit_line([(int(c), number(r["power_w"])) for c, r in sleeps.items()])
    put("SM sweep line", "sleep W per SM", per_sm)

    return values


LEVELS = ["shared", "L1", "L2", "DRAM", "none", "compute", "const-L1", "const-L1.5"]


def order(key):
    """Sections in a fixed order, entries by memory level, then as they came (SM counts from most to fewest)."""
    section, name = key
    words = name.replace(",", " ").split()
    level = LEVELS.index(words[0]) if words and words[0] in LEVELS else len(LEVELS)
    count = -int(words[-2]) if len(words) > 2 and words[-1] == "SMs" else 0
    rest = " ".join(words[1:-2]) if count else ""
    return (SECTIONS.index(section) if section in SECTIONS else len(SECTIONS), section, level if count or section.startswith("duty") else 0, rest, count)


SECTIONS = ["power", "fit pJ/B, random", "fit pJ/op, random", "fit pJ/access, random", "duty 100 pJ/B", "duty 100 pJ/op",
            "compute pJ/FLOP", "atomics pJ/access", "constant pJ/warp request", "datapath share", "width model",
            "fit pJ/B, zero", "fit pJ/op, zero", "SM sweep line", "SM sweep pJ/access"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefixes", nargs="+", help="output prefixes of the runs (--output)")
    parser.add_argument("--csv", help="also write the table to this CSV file")
    arguments = parser.parse_args()

    runs = [metrics(*load(prefix)) for prefix in arguments.prefixes]
    keys = []
    for run in runs:
        keys += [key for key in run if key not in keys]
    keys.sort(key=order)  # stable: entries of equal rank keep the order they came in

    output = []
    section = None
    for key in keys:
        values = [run.get(key, math.nan) for run in runs]
        present = [v for v in values if not math.isnan(v)]
        mean = statistics.mean(present) if present else math.nan
        sd = statistics.stdev(present) if len(present) > 1 else math.nan
        cv = sd / abs(mean) if present and mean != 0.0 else math.nan
        spread = (max(present) - min(present)) / abs(mean) if len(present) > 1 and mean != 0.0 else math.nan
        output.append((key, values, mean, cv, spread))

        if key[0] != section:
            section = key[0]
            print(f"\n{section}")
            print(f'  {"":34}' + "".join(f"{p:>10}" for p in arguments.prefixes) + f'{"mean":>10}{"CV %":>8}{"range %":>9}')
        print(f"  {key[1]:34}" + "".join(f"{v:10.4g}" for v in values) + f"{mean:10.4g}{100 * cv:8.2f}{100 * spread:9.2f}")

    if arguments.csv:
        with open(arguments.csv, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["section", "metric"] + arguments.prefixes + ["mean", "cv", "range"])
            for (section, name), values, mean, cv, spread in output:
                writer.writerow([section, name.strip()] + values + [mean, cv, spread])


if __name__ == "__main__":
    main()
