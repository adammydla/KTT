#!/usr/bin/env python3
"""Compare the probes and the energy model of two GPUs (for a port: the new GPU against a reference).

    python3 tools/compare_gpus.py --ref-power R/Full,R/Ffma3,R/Writes,R/Instr --ref-counters R/ProfileFull4 \
        --power Full --counters ProfileFull [--labels "RTX 3090,RTX 2080 Ti"]

The --power and --counters prefixes are those of energy_model.py (comma-separated runs are averaged). Printed:
1. the GPUs and their sleep baselines;
2. the energies of the plain probes (pJ per access or per operation of one thread), which the probes measure directly;
3. combined costs of the fit (energy_model.COMBINED: a line read from DRAM, a line written back, a conflict-free shared
   load, an FFMA), which the probes determine well;
4. the energy of every event. Events that always occur together trade off in the fit, so a single event's energy is not
   comparable between GPUs (on an RTX 2080 Ti "L2 read miss request" came out 14265 and "DRAM read sector" 394 pJ, on the
   RTX 3090 6724 and 2419 pJ, for nearly the same line read). Compare the GPUs on sections 2 and 3.
"""

import argparse
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import energy_model as m  # noqa: E402

# Plain probes that every GPU has (labels of energy_model.probe_table).
PLAIN = ["L1 ld w4", "L1 ld w16", "L2 ld w4", "L2 ld w16", "DRAM ld w4", "DRAM ld w16", "L2 st w16", "DRAM st w16",
         "shared ld.shared w16", "shared st.shared w16", "compute loop", "compute ffma", "compute ffma3", "compute imad",
         "compute hfma2", "compute dfma", "compute hmma", "compute mufu", "compute shfl"]


def model(power, counters):
    """Probes, energy table, energies per event and the events no probe exercises."""
    with contextlib.redirect_stdout(io.StringIO()):
        probes, table = m.probe_table(power.split(","), counters.split(","))
    all_probes = {p.label: p for p in probes}
    if any(p.label == "compute ffma3" for p in probes):
        probes = [p for p in probes if p.label not in m.TWO_REGISTER_FFMA]
    probes = [p for p in probes if p.label not in m.CHECK_ONLY]
    energies, _, unused = m.fit(probes, m.EVENTS, False)
    return all_probes, table, energies, unused


def ratio(a, b):
    return f"{b / a:6.2f}" if a and b and a > 0 else "     -"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-power", required=True, help="power runs of the reference GPU")
    parser.add_argument("--ref-counters", required=True, help="counter runs of the reference GPU")
    parser.add_argument("--power", required=True, help="power runs of the new GPU")
    parser.add_argument("--counters", required=True, help="counter runs of the new GPU")
    parser.add_argument("--labels", help="names of the two GPUs, comma-separated (default: from the energy tables)")
    arguments = parser.parse_args()

    ref = model(arguments.ref_power, arguments.ref_counters)
    new = model(arguments.power, arguments.counters)
    labels = arguments.labels.split(",") if arguments.labels else [ref[1].get("device", "reference"),
                                                                   new[1].get("device", "new")]
    width = max(len(l) for l in labels) + 2

    print("GPUs:")
    for label, (_, table, _, _) in zip(labels, (ref, new)):
        baseline = table.get("baseline", {})
        cc = table.get("compute_capability") or m.counter_table.device_compute_capability(table.get("device") or "")
        static = table.get("static_w")
        print(f"  {label:{width}} compute capability {cc or '?'}, sleep baseline "
              f"{baseline.get('mean_w', float('nan')):.1f} W at {baseline.get('mean_temperature_c', float('nan')):.1f} C "
              f"({baseline.get('slope_w_per_c', float('nan')):.2f} W/C), static "
              + (f"{static:.1f} W" if isinstance(static, (int, float)) else "-"))

    print(f"\nPlain probes (pJ per access, or per operation of one thread):\n  {'probe':24} {labels[0]:>12} "
          f"{labels[1]:>12}  ratio")
    for label in PLAIN:
        a = ref[0].get(label)
        b = new[0].get(label)
        if a or b:
            print(f"  {label:24} {a.energy if a else float('nan'):12.2f} {b.energy if b else float('nan'):12.2f}  "
                  f"{ratio(a.energy if a else None, b.energy if b else None)}")

    print(f"\nCombined costs of the fit (compare GPUs on these):")
    for (label, a, unit), (_, b, _) in zip(m.combined_costs(ref[2]), m.combined_costs(new[2])):
        print(f"  {label:76} {a:9.2f} {b:9.2f} {unit}  {ratio(a, b)}")

    print(f"\nEnergy per event (pJ; single events trade off in the fit, so they are not comparable between GPUs):")
    for name in m.EVENTS:
        if name == "inst fma lite (fp32)":
            continue
        flags = "".join(f"  no probe on the {label}" for label, (_, _, _, unused) in zip(labels, (ref, new))
                        if name in unused)
        print(f"  {name:30} {ref[2][name]:10.1f} {new[2][name]:10.1f}{flags}")


if __name__ == "__main__":
    main()
