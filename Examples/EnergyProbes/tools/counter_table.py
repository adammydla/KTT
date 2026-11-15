"""The CUPTI counter table of EnergyProbes (tools/counters.json), for the Python tools.

The table is the one place for counter names: the program collects every name in it (CounterTable.cpp), and the tools
take the names, the events they feed and their aliases (the same counter under another chip's name) from here. A counter
that is not in the table cannot be read (value_of raises), so a new counter has to be added to the table first.
"""

import csv
import json
import math
import os
import re

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "counters.json")

with open(PATH) as _file:
    _TABLE = json.load(_file)

# Every counter, in the order ModelCounters() collects them, and its entry.
NAMES = [entry["name"] for entry in _TABLE["counters"]]
ENTRIES = {entry["name"]: entry for entry in _TABLE["counters"]}
# Optional counter groups and the counter that shows whether a run has the group.
GROUPS = dict(_TABLE["groups"])

# The canonical name of each alias, and the aliases of each canonical name.
CANONICAL = {name: entry["alias_of"] for name, entry in ENTRIES.items() if "alias_of" in entry}
ALIASES = {}
for _alias, _name in CANONICAL.items():
    ALIASES[_name] = ALIASES.get(_name, ()) + (_alias,)

def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def read_rows(path):
    """The rows of a run's CSV (PREFIX.csv, PREFIX_validation.csv) as dicts, like csv.DictReader. Runs from before
    2026-11-11 wrote notes with a comma unquoted: such a row has more cells than the header, and every counter after the
    note would be read one column off. The extra cells are put back into the note."""
    with open(path, newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header is None:
            return []
        note = header.index("note") if "note" in header else None
        rows = []
        for fields in reader:
            if not fields:
                continue
            extra = len(fields) - len(header)
            if extra > 0 and note is not None:
                fields = fields[:note] + [",".join(fields[note:note + extra + 1])] + fields[note + extra + 1:]
            row = dict.fromkeys(header)
            row.update(zip(header, fields))
            rows.append(row)
    return rows


def value_of(row, name):
    """A counter's value in a row, from its canonical name or an alias; NaN if the row has none. Raises for a name that
    is not in the table."""
    if name not in ENTRIES:
        raise KeyError(f"counter {name} is not in {PATH}; add it there first")
    for candidate in (name,) + ALIASES.get(name, ()):
        value = number(row.get(candidate))
        if not math.isnan(value):
            return value
    return math.nan


def feeding(event):
    """The counters that feed exactly `event`, in table order."""
    return [name for name in NAMES if ENTRIES[name].get("feeds") == event]


# Compute capability from a GPU's name, for runs that did not record it: the first pattern found.
DEVICE_CC = [(r"RTX 20\d\d|TITAN RTX|Quadro RTX|\bT4\b|GTX 16\d\d", 7.5), (r"V100|TITAN V\b", 7.0),
             (r"A100|A30\b|A800", 8.0), (r"RTX 30\d\d|RTX A\d+|\bA10G?\b|\bA40\b|A16\b|A2\b", 8.6),
             (r"RTX 40\d\d|RTX \d+ Ada|L4\b|L40", 8.9), (r"H100|H200|H800|GH200", 9.0), (r"B100|B200|GB200", 10.0),
             (r"RTX 50\d\d|RTX PRO", 12.0)]


def device_compute_capability(device):
    """The compute capability of a GPU from its name (DEVICE_CC), for runs that did not record it; None if unknown."""
    return next((cc for pattern, cc in DEVICE_CC if re.search(pattern, device)), None)
