#!/usr/bin/env python3
"""Which CUPTI counters of the energy model a GPU lacks, and what that does to the model.

The counters come from the counter table, tools/counters.json (ModelCounters() in ProbeKernels.cpp collects the same);
the GPU's metrics from Nsight Compute
(`ncu --query-metrics --chip <chip>`, no GPU needed) or from a profiled run's CSV (columns that are missing or empty).
KTT skips counters a GPU lacks with only a warning, and tools/energy_model.py counts a missing counter as 0, so a
missing counter silently removes its events from the model. The output lists each missing counter with the model events
it feeds and, where ncu knows one, a metric of a similar name to map it to.

    python3 tools/check_counters.py --chip ad102            # before building or measuring (chip names: ncu --list-chips)
    python3 tools/check_counters.py --csv ProfileFull.csv   # after a profiled run (--profile)
    python3 tools/check_counters.py --csv ProfileFull.csv --cc 7.5   # runs from before 2026-11-06 do not record it

A counter that the GPU lacks is covered (an alias or an alternative of it is there), not applicable (it cannot exist below
its compute capability, as LDGSTS before sm_80), optional (only an option uses it), without an event (a check), or
missing; only a missing one fails.
"""

import argparse
import csv
import difflib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import counter_table  # noqa: E402

# Compute capability of the chips Nsight Compute names (ncu --list-chips), for counters that a GPU cannot have.
CHIP_CC = {"gv100": 7.0, "tu102": 7.5, "tu104": 7.5, "tu106": 7.5, "tu116": 7.5, "tu117": 7.5, "ga100": 8.0,
           "ga102": 8.6, "ga103": 8.6, "ga104": 8.6, "ga106": 8.6, "ga107": 8.6, "ga10b": 8.7, "ad102": 8.9,
           "ad103": 8.9, "ad104": 8.9, "ad106": 8.9, "ad107": 8.9, "gh100": 9.0, "gb100": 10.0, "gb102": 10.0,
           "gb202": 12.0, "gb203": 12.0, "gb205": 12.0, "gb206": 12.0, "gb207": 12.0}


def base(name):
    """A counter's name without its .sum/.avg/.pct suffix (as ncu --query-metrics lists it)."""
    return name.rsplit(".", 1)[0]


def model_counters():
    """The counters of ModelCounters() (tools/counters.json), without their suffix."""
    return [base(name) for name in counter_table.NAMES]


def chip_metrics(chip):
    result = subprocess.run(["ncu", "--query-metrics", "--chip", chip], capture_output=True, text=True)
    names = {line.split()[0] for line in result.stdout.splitlines() if line.strip() and re.match(r"^[a-z]", line)}
    if not names:
        raise SystemExit(f"ncu knows no metrics for chip {chip!r} ({result.stderr.strip()}); see `ncu --list-chips`")
    return names


def csv_metrics(path):
    """Counters with a value in some row of a profiled run (columns of counters KTT skipped are missing or empty)."""
    rows = counter_table.read_rows(path)
    return {name.rsplit(".", 1)[0] for name in rows[0] if any(row.get(name) not in ("", None) for row in rows)}


def csv_compute_capability(path):
    """The compute capability of the GPU of a run: recorded in its energy table (runs since 2026-11-06), else guessed from
    the GPU's name there (counter_table.DEVICE_CC); None if neither works."""
    table = re.sub(r"(_validation)?\.csv$", "", path) + "_energy_table.json"
    try:
        with open(table) as file:
            data = json.load(file)
    except (OSError, ValueError):
        return None
    if data.get("compute_capability"):
        return float(data["compute_capability"])
    return counter_table.device_compute_capability(data.get("device") or "")


def classify(name, available, compute_capability):
    """Why a counter that the GPU lacks does or does not matter: (kind, text). Kinds: covered, not applicable, option,
    no event, missing."""
    entry = counter_table.ENTRIES[name]
    if "alias_of" in entry and base(entry["alias_of"]) in available:
        return "covered", f"the same counter as {base(entry['alias_of'])}, which is available"
    aliases = [base(a) for a in counter_table.ALIASES.get(name, ())]
    if any(a in available for a in aliases):
        return "covered", f"available as {', '.join(a for a in aliases if a in available)} (an alias)"
    alternative = counter_table.ALTERNATIVES.get(name)
    if alternative and base(alternative) in available:
        return "covered", f"this GPU has {base(alternative)} instead (this one exists on {entry.get('exists', '?')})"
    if not counter_table.applicable(name, compute_capability):
        return "not applicable", f"exists from compute capability {entry['min_cc']}: {entry.get('exists', '')}"
    if entry.get("study"):
        return "option", f"screening counter, only --study-counters collects it ({entry.get('use', '')})"
    if "only_with" in entry:
        return "option", f"only {entry['only_with']} uses it ({entry.get('exists', '')})"
    if "feeds" not in entry:
        return "no event", entry.get("use", "no event")
    return "missing", entry["feeds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--chip", help="Nsight Compute chip name, e.g. tu102, ga100, ga102, ad102, gh100, gb100, gb202")
    source.add_argument("--csv", help="CSV of a run with --profile (or a _validation.csv)")
    parser.add_argument("--cc", type=float, help="compute capability, e.g. 7.5 (from --chip or the run's energy table "
                        "when not given): counters that cannot exist on it are 'not applicable', not missing")
    arguments = parser.parse_args()

    counters = counter_table.NAMES
    available = chip_metrics(arguments.chip) if arguments.chip else csv_metrics(arguments.csv)
    compute_capability = arguments.cc or (CHIP_CC.get(arguments.chip) if arguments.chip
                                          else csv_compute_capability(arguments.csv))
    absent = [name for name in counters if base(name) not in available]
    kinds = {name: classify(name, available, compute_capability) for name in absent}
    missing = [name for name in absent if kinds[name][0] != "covered"]
    for name in absent:
        if kinds[name][0] == "covered":
            print(f"  {base(name)}: covered, {kinds[name][1]}")
    print(f"{len(counters) - len(missing)} of {len(counters)} counters of ModelCounters() available "
          f"({arguments.chip or arguments.csv}" + (f", compute capability {compute_capability}" if compute_capability
                                                   else "") + ")")
    for name in missing:
        kind, text = kinds[name]
        label = {"missing": "missing", "not applicable": "not applicable", "option": "missing, optional",
                 "no event": "missing, no event"}[kind]
        similar = difflib.get_close_matches(base(name), sorted(available), n=2, cutoff=0.85) if arguments.chip else []
        print(f"  {label} {base(name)}\n      " + ("feeds: " if kind == "missing" else "") + text
              + (f"\n      similar: {', '.join(similar)}" if similar and kind == "missing" else ""))
    harmful = [name for name in missing if kinds[name][0] == "missing"]
    if compute_capability is None and any(counter_table.ENTRIES[name].get("min_cc") for name in harmful):
        print("\nThe compute capability is unknown: pass --cc, so that counters this GPU cannot have are not missing.")
    if harmful:
        print("\nEvents fed by a missing counter read 0 in tools/energy_model.py: map the counter (an alias in "
              "tools/counters.json) before trusting the model on this GPU.")
    sys.exit(1 if harmful else 0)


if __name__ == "__main__":
    main()
