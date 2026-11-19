#!/usr/bin/env python3
"""Put the counters of a --recount run into a copy of the --validate run it counted again (2026-11-17).

A --validate run measures power and counters; a counter that it did not collect (a candidate added later) would need the
whole run again, power included. --recount runs only the counter pass over the same configurations:

    ./EnergyProbesCuda $GPU --recount Run_validation.csv --output RunRecount
    python3 tools/merge_counters.py Run_validation.csv RunRecount_recount.csv Run2_validation.csv

The output has every row and column of the original, with each counter of the recount in place of the original's (and
counters the original lacks added). Counters are a property of the code, so the two runs must agree on the counters
they share: per kernel the median relative difference of the instruction and DRAM sector counts is printed, and above
TOLERANCE the merge refuses (another build of the kernels, other data or another GPU).
"""

import argparse
import collections
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_table  # noqa: E402

COMPARED = ("smsp__inst_executed.sum", "dram__sectors_read.sum", "lts__t_sectors_srcunit_tex_op_read.sum")
TOLERANCE = 0.02
NOT_COUNTERS = ("kernel", "configuration", "kernel_s", "ktt_kernel_s", "note")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("original", help="PREFIX_validation.csv of the --validate run")
    parser.add_argument("recount", help="PREFIX_recount.csv of the --recount run")
    parser.add_argument("output", help="the merged PREFIX_validation.csv to write")
    arguments = parser.parse_args()
    original = counter_table.read_rows(arguments.original)
    recount = {(r["kernel"], r["configuration"]): r for r in counter_table.read_rows(arguments.recount)}
    with open(arguments.original) as file:
        header = next(csv.reader(file))
    with open(arguments.recount) as file:
        added = [name for name in next(csv.reader(file)) if name not in NOT_COUNTERS]
    fields = header + [name for name in added if name not in header]

    differences = collections.defaultdict(list)
    missing = 0
    for row in original:
        if row["valid"] != "1":
            continue
        new = recount.get((row["kernel"], row["configuration"]))
        if new is None or not new.get("smsp__inst_executed.sum"):
            missing += 1
            continue
        for name in COMPARED:
            old, value = counter_table.number(row.get(name, "")), counter_table.number(new.get(name, ""))
            if old > 0 and value == value:
                differences[row["kernel"]].append(abs(value / old - 1))
        for name in added:
            row[name] = new[name]
    worst = 0.0
    for kernel, values in differences.items():
        median = statistics.median(values)
        worst = max(worst, median)
        print(f"  {kernel:16} counts differ from the original by {100 * median:.2f} % (median of {', '.join(COMPARED)})")
    if missing:
        print(f"ERROR: {missing} valid configurations of {arguments.original} are not counted in {arguments.recount}")
        return 1
    if worst > TOLERANCE:
        print(f"ERROR: the counts differ by more than {100 * TOLERANCE:.0f} %: not the same kernels, data or GPU")
        return 1
    with open(arguments.output, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(original)
    print(f"Wrote {arguments.output}: {len(original)} rows, {len(added)} counters from the recount")
    return 0


if __name__ == "__main__":
    sys.exit(main())
