#!/usr/bin/env python3
"""Which counters of tools/counters.json a GPU has, before the first run on it (NEWER_ARCHITECTURES.md section 4.2).

Nsight Compute lists the metrics of every chip it supports without a GPU (`ncu --query-metrics --chip gh100`). This
compares the counter table with such a list: per counter whether the chip has it (under its name, an alias or its
alternative), and which model events would lose a counter. A missing counter counts as 0, so an event that loses all its
counters silently costs nothing on that GPU.

    python3 tools/check_chip_metrics.py --chip gh100 [--chip gb202 ...]   # runs ncu (--ncu PATH, or found on PATH)
    python3 tools/check_chip_metrics.py --metrics gh100.txt                # a saved `ncu --query-metrics` output
    ncu --list-chips                                                        # the chip names

Exit status 1 if a counter that feeds an event is missing on a chip (aliases and alternatives count as present; counters
marked "optional" in the table, which only some chips have, do not count).
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_table  # noqa: E402


# Compute capability of the chips `ncu --list-chips` names (by prefix; the longest prefix wins), for the table's min_cc.
CHIP_CC = {"gv100": 7.0, "gv11b": 7.2, "tu1": 7.5, "ga100": 8.0, "ga10b": 8.7, "ga10": 8.6, "ad10": 8.9, "gh100": 9.0,
           "gb100": 10.0, "gb102": 10.0, "gb10b": 12.1, "gb20b": 11.0, "gb20": 12.0}


def chip_cc(label):
    """The compute capability of a chip name or of a file named after one (gh100.txt); None if unknown."""
    name = os.path.basename(label).lower()
    matches = [prefix for prefix in CHIP_CC if name.startswith(prefix)]
    return CHIP_CC[max(matches, key=len)] if matches else None


def base_name(counter):
    """The metric a counter name is a roll-up of: 'dram__sectors_read.sum' -> 'dram__sectors_read'."""
    return counter.split(".", 1)[0]


def read_metrics(text):
    """The base metric names of an `ncu --query-metrics` listing (the first word of every line that names a metric)."""
    names = set()
    for line in text.splitlines():
        word = line.split(maxsplit=1)[0] if line.strip() else ""
        if "__" in word:
            names.add(word)
    return names


def find_ncu(explicit):
    if explicit:
        return explicit
    found = shutil.which("ncu")
    if found:
        return found
    installed = sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"))
    if installed:
        return installed[-1]
    raise SystemExit("ncu not found: pass --ncu PATH or --metrics FILE")


def query(ncu, chip):
    result = subprocess.run([ncu, "--query-metrics", "--chip", chip], capture_output=True, text=True)
    if result.returncode != 0 or "__" not in result.stdout:
        raise SystemExit(f"{ncu} --query-metrics --chip {chip} failed: {(result.stderr or result.stdout).strip()[:300]}")
    return result.stdout


def report(label, metrics, cc=None):
    """Prints the comparison for one chip; returns the counters that feed events and are missing. With the chip's compute
    capability, counters whose min_cc is above it are not expected."""
    entries = {name: entry for name, entry in counter_table.ENTRIES.items()
               if cc is None or entry.get("min_cc", 0) <= cc + 1e-9}
    # Names that stand for one another: a counter and its aliases, a counter and its alternative.
    stands_for = {name: {name} for name in entries}
    for name, entry in entries.items():
        for other in (entry.get("alias_of"), entry.get("alternative")):
            if other:
                stands_for[name].add(other)
                stands_for.setdefault(other, {other}).add(name)
    present = {name for name in entries if base_name(name) in metrics}
    covered = {name for name in entries if stands_for[name] & present}
    feeding = [name for name, entry in entries.items() if "feeds" in entry and "alias_of" not in entry]
    missing_feeding = [name for name in feeding if name not in covered and not entries[name].get("optional")]
    checks = [name for name, entry in entries.items() if "use" in entry]
    print(f"{label} (compute capability {cc if cc else 'unknown'}): {len(metrics)} metrics; of the {len(entries)} counters "
          f"expected there, {len(present)} exist under their "
          f"name, {len(covered)} with aliases and alternatives")
    for name in feeding:
        if name not in present:
            via = sorted((stands_for[name] & present) - {name})
            state = f"via {', '.join(via)}" if via else "not here (optional)" if entries[name].get("optional") else "MISSING"
            print(f"  {state:46} {name}  (feeds {entries[name]['feeds']})")
    absent_checks = [name for name in checks if name not in covered]
    if absent_checks:
        print(f"  checks the chip does not have ({len(absent_checks)}): " + ", ".join(absent_checks))
    events = sorted({entries[name]["feeds"] for name in missing_feeding})
    if events:
        print("  events that lose a counter (it counts as 0 there): " + "; ".join(events))
    return missing_feeding


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chip", action="append", default=[], help="a chip name of `ncu --list-chips` (repeatable)")
    parser.add_argument("--metrics", action="append", default=[], help="a saved `ncu --query-metrics` listing (repeatable)")
    parser.add_argument("--cc", type=float, help="the compute capability (default: from the chip or file name)")
    parser.add_argument("--ncu", help="the ncu executable (default: on PATH, else the newest /opt/nvidia/nsight-compute)")
    arguments = parser.parse_args()
    if not arguments.chip and not arguments.metrics:
        parser.error("give --chip or --metrics")
    failed = False
    for path in arguments.metrics:
        with open(path) as file:
            failed |= bool(report(os.path.basename(path), read_metrics(file.read()), arguments.cc or chip_cc(path)))
    if arguments.chip:
        ncu = find_ncu(arguments.ncu)
        for chip in arguments.chip:
            failed |= bool(report(chip, read_metrics(query(ncu, chip)), arguments.cc or chip_cc(chip)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
