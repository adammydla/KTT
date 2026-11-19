#!/usr/bin/env python3
"""Energies of single SASS opcodes against the model's price of their pipe (the optional opcode probes).

The model prices instructions per pipe: every warp instruction of the fma pipe costs the same, whether FFMA, FADD, FMUL
or IMAD, and so on. Wattchmen (Tran et al., ICS'26) prices every SASS opcode instead. The opcode probes (EnergyProbes
--only opcodes, COMPUTE_OP 17) run one opcode per chain and step, with helpers where the opcode alone cannot keep its data
changing. For each probe this tool takes the model's prediction (the pipe prices times the counted events, helpers and
loop included) and puts what is left of the measured energy on the target opcode:

    energy of one warp instruction of the opcode = pipe price + (measured - predicted) / target instructions per access

The probes of MUFU, conversions and POPC need helpers (the chain that feeds the opcode, and LOP3, IADD3, FMUL or DMUL
that keep the results). A second pass prices the helpers that have probes of their own (LOP3, IADD3, FMUL, DMUL by a
constant) with their measured energies instead of their pipe's price; FFMA and IMAD helpers keep the pipe price, which
was fitted on them.

The target instructions per access come from the SASS of the probe's loop (tools/check_sass.py compiles it the way the
program does): 64 target instructions per loop iteration are 64 counted operations, except where ptxas folds two steps
into one three-input instruction (IADD3, LOP3). The probes with almost constant operands (-const) give the same with
operands that do not change.

    python3 tools/opcode_energy.py --power Opcodes1 --counters OpcodesProfile1 --model models/rtx3090_1200mhz.model.json \
        [--out models/rtx3090_1200mhz.opcodes.json]

The operand reuse probes (EnergyProbes --only reuse, COMPUTE_OPCODE 20-22, 2026-11-18) are priced the same way and
printed apart, not put in the table: FFMA with the same operand values from 0, 1 or 2 shared registers, so that the
difference is the operand reuse cache (operands marked .reuse in the SASS are read from it instead of the register file
banks). With --per-run every power run is priced on its own and the spread of the runs printed:

    python3 tools/opcode_energy.py --power Reuse1,Reuse2,Reuse3 --counters ReuseProfile1 --model ... --per-run

The output file maps SASS opcodes (as Nsight Compute names them with all modifiers: FADD, MUFU.EX2, I2FP.F32.S32, ...)
to their energy per warp instruction and their pipe; tools/app_energy.py score --opcodes applies it to kernels whose
per-opcode counts Nsight Compute collected. Needs NumPy (energy_model) and, for the SASS, a CUDA toolkit (no GPU).
"""

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_sass  # noqa: E402
import energy_model  # noqa: E402
import energy_predictor  # noqa: E402

# COMPUTE_OPCODE -> (probe name, SASS opcode with modifiers, the model's pipe event of that opcode).
FMA = "inst fma (fp32, imad)"
OPCODES = {
    1: ("fadd", "FADD", FMA), 2: ("fmul", "FMUL", FMA), 3: ("ffma-rr", "FFMA", FMA),
    4: ("dadd", "DADD", "inst fp64"), 5: ("dmul", "DMUL", "inst fp64"), 6: ("dfma-rr", "DFMA", "inst fp64"),
    7: ("iadd3", "IADD3", "inst alu"), 8: ("lop3", "LOP3.LUT", "inst alu"), 9: ("shf", "SHF.L.W.U32.HI", "inst alu"),
    10: ("mufu-ex2", "MUFU.EX2", "inst xu (mufu)"), 11: ("mufu-lg2", "MUFU.LG2", "inst xu (mufu)"),
    12: ("mufu-sin", "MUFU.SIN", "inst xu (mufu)"), 13: ("mufu-rcp", "MUFU.RCP", "inst xu (mufu)"),
    14: ("mufu-sqrt", "MUFU.SQRT", "inst xu (mufu)"), 15: ("mufu-tanh", "MUFU.TANH", "inst xu (mufu)"),
    16: ("i2f", "I2FP.F32.S32", "inst alu"), 17: ("f2i", "F2I.NTZ", "inst xu (mufu)"), 18: ("popc", "POPC", "inst xu (mufu)"),
    19: ("f2f", "F2F", "inst fp64"),
}
# The operand reuse probes: FFMA with the same operand values in all chains, from 0, 1 or 2 shared registers.
REUSE_OPCODES = {20: ("ffma-same", "FFMA", FMA), 21: ("ffma-reuse1", "FFMA", FMA), 22: ("ffma-reuse2", "FFMA", FMA)}
# Opcodes whose target is several SASS opcodes (the F2F probe converts to double and back: two per step).
TARGET_PARTS = {"F2F": ("F2F.F64.F32", "F2F.F32.F64")}
OPERATIONS_PER_ITERATION = 64  # UNROLL x COMPUTE_CHAINS
# Helpers priced with their own probe in the second pass, and how many instructions of each the loop has of its own.
# The F2F probe's DMUL multiplies by a constant: priced with the DMUL probe's almost constant operands ("DMUL const").
HELPERS = {"LOP3.LUT": 1, "IADD3": 1, "FMUL": 0, "FMUL.RZ": 0, "DMUL": 0}
HELPER_PROBES = {"LOP3.LUT": "LOP3.LUT", "IADD3": "IADD3", "FMUL": "FMUL", "FMUL.RZ": "FMUL", "DMUL": "DMUL const"}


def target_per_iteration(code, constant, arch, cuda):
    """Target instructions in one iteration of the probe's loop (from its SASS)."""
    nvrtc = check_sass.Nvrtc(cuda)
    with tempfile.TemporaryDirectory() as work:
        source = check_sass.probe_source("Compute.cu", {"COMPUTE_OP": 17, "COMPUTE_OPCODE": code,
                                                        "COMPUTE_CONSTANT": constant})
        addressed, _ = check_sass.sass(nvrtc, cuda, source, arch, "probeCompute", work, addresses=True)
    loop = check_sass.main_loop(addressed)
    target = {**OPCODES, **REUSE_OPCODES}[code][1]
    count = sum(loop.get(part, 0) for part in TARGET_PARTS.get(target, (target,)))
    # The loop's own IADD3 or LOP3 (one each) is not a target.
    own = {"IADD3": 1, "LOP3.LUT": 1}.get(target, 0) if code in (7, 8) else 0
    loop["_reads"] = check_sass.register_reads(check_sass.loop_lines(addressed), target)
    return max(count - own, 0), loop


def reuse_summary(rows):
    """The operand reuse cache from the FFMA probes with the same operand values (opcodes 20-22): energy per FFMA, reused
    operands per FFMA, and the saving per reused operand against the probe without reuse (ffma-same)."""
    by_name = {row["probe"].split("opcode-", 1)[1]: row for row in rows}
    base = by_name.get("ffma-same")
    result = {}
    for name in ("ffma-rr", "ffma-same", "ffma-reuse1", "ffma-reuse2"):
        row = by_name.get(name)
        if row is None:
            continue
        reused, same_bank = row["loop"]["_reads"]
        entry = {"pj_per_ffma": row["opcode_pj"], "reused_per_ffma": reused, "same_bank_pairs_per_ffma": same_bank}
        if base is not None and name.startswith("ffma-reuse") and reused > base["loop"]["_reads"][0]:
            saving = (base["opcode_pj"] - row["opcode_pj"]) / (reused - base["loop"]["_reads"][0])
            entry["saving_pj_per_reused_operand"] = saving
            entry["saving_share_of_ffma"] = saving / base["opcode_pj"]
        result[name] = entry
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--power", required=True, type=lambda s: s.split(","), help="power runs (prefixes)")
    parser.add_argument("--counters", required=True, type=lambda s: s.split(","), help="counter runs (prefixes)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--arch", default="86")
    parser.add_argument("--out", help="write the opcode table here")
    parser.add_argument("--per-run", action="store_true",
                        help="price every power run on its own and print the spread (repeats)")
    args = parser.parse_args()
    cuda = os.environ.get("CUDA_PATH", "/usr/local/cuda")
    model = energy_predictor.Model.load(args.model)
    with contextlib.redirect_stdout(io.StringIO()):
        checks, _ = energy_model.probe_table(args.power, args.counters,
                                             busy_scale=float(model.options.get("busy_scale_gbs", 95.0)), checks=True)
    if args.per_run and len(args.power) > 1:
        runs = []
        for power in args.power:
            with contextlib.redirect_stdout(io.StringIO()):
                checks, _ = energy_model.probe_table([power], args.counters,
                                                     busy_scale=float(model.options.get("busy_scale_gbs", 95.0)),
                                                     checks=True)
            probes = {p.label: p for p in checks if p.label.startswith(energy_model.OPCODE_PREFIX)}
            runs.append((power, opcode_rows(probes, model, args.arch, cuda)))
        print_per_run(runs)
        return
    probes = {p.label: p for p in checks if p.label.startswith(energy_model.OPCODE_PREFIX)}
    if not probes:
        raise SystemExit("no opcode probes in these runs (EnergyProbes --only opcodes or --only reuse)")
    first, rows, table = opcode_rows(probes, model, args.arch, cuda, verbose=True)
    reuse = reuse_summary([row for row in rows if row["probe"].split("opcode-", 1)[1].startswith("ffma-")])
    if reuse:
        print_reuse(reuse)
    if args.out:
        with open(args.out, "w") as file:
            json.dump({"format": "energyprobes-opcodes", "version": 1, "model": os.path.basename(args.model),
                       "model_events_sha256": model.data["tools"].get("events_sha256"),
                       "power_runs": args.power, "counter_runs": args.counters, "opcodes": table, "probes": rows,
                       **({"operand_reuse": reuse} if reuse else {})},
                      file, indent=1)
        print(f"wrote {args.out}")


def print_reuse(reuse):
    print("\noperand reuse cache (FFMA; ffma-same, ffma-reuse1/2 have the same operand values, only their registers differ):")
    print(f"  {'probe':<14}{'pJ/FFMA':>9}{'reused/FFMA':>13}{'same-bank pairs':>17}{'saving/reused pJ':>18}{'share':>8}")
    for name, entry in reuse.items():
        saving = entry.get("saving_pj_per_reused_operand")
        print(f"  {name:<14}{entry['pj_per_ffma']:>9.1f}{entry['reused_per_ffma']:>13.2f}"
              f"{entry['same_bank_pairs_per_ffma']:>17.2f}" + (f"{saving:>18.1f}{entry['saving_share_of_ffma']:>8.1%}"
                                                               if saving is not None else ""))


def print_per_run(runs):
    """Every power run priced on its own: the reuse probes (if any) and every opcode's energy, with the spread."""
    import statistics
    names = {}
    for power, (_, rows, _) in runs:
        for row in rows:
            names.setdefault(row["probe"], []).append(row["opcode_pj"])
    print(f"{'probe':<30}" + "".join(f"{power:>12}" for power, _ in runs) + f"{'mean':>10}{'spread':>9}")
    for probe, values in names.items():
        if len(values) != len(runs):
            continue
        mean = statistics.fmean(values)
        print(f"{probe:<30}" + "".join(f"{v:>12.1f}" for v in values) + f"{mean:>10.1f}"
              f"{(max(values) - min(values)) / 2 / mean:>9.1%}")
    reuses = [reuse_summary([r for r in rows if r["probe"].split("opcode-", 1)[1].startswith("ffma-")])
              for _, (_, rows, _) in runs]
    for name in ("ffma-reuse1", "ffma-reuse2"):
        savings = [r[name]["saving_pj_per_reused_operand"] for r in reuses
                   if name in r and "saving_pj_per_reused_operand" in r[name]]
        if savings:
            base = statistics.fmean(r["ffma-same"]["pj_per_ffma"] for r in reuses if "ffma-same" in r)
            print(f"{name}: saving per reused operand " + ", ".join(f"{v:.1f}" for v in savings) + " pJ; mean "
                  f"{statistics.fmean(savings):.1f} pJ = {statistics.fmean(savings) / base:.1%} of the FFMA without reuse")


def opcode_rows(probes, model, arch, cuda, verbose=False):
    """(first-pass prices, rows of every opcode probe, the opcode table) of the opcode probes in `probes`."""
    table = {}
    rows = []

    def price(code, constant, helpers):
        name, sass_opcode, pipe = {**OPCODES, **REUSE_OPCODES}[code]
        label = f"{energy_model.OPCODE_PREFIX}{name}" + ("-const" if constant else "")
        probe = probes.get(label)
        if probe is None:
            return None
        targets, loop = target_per_iteration(code, constant, arch, cuda)
        per_access = targets / OPERATIONS_PER_ITERATION / 32  # warp instructions of the target per counted operation
        predicted = sum(probe.counts[e] * model.pj.get(e, 0.0) for e in probe.counts)
        # Second pass: helpers with probes of their own at their measured energy.
        for helper, own in HELPERS.items():
            if helper == sass_opcode or helper not in loop or HELPER_PROBES[helper] not in helpers:
                continue
            entry = helpers[HELPER_PROBES[helper]]
            count = max(loop[helper] - own, 0)
            predicted += (entry["pj"] - entry["pipe_pj"]) * count / OPERATIONS_PER_ITERATION / 32
        pipe_pj = model.pj[pipe]
        opcode_pj = pipe_pj + (probe.energy - predicted) / per_access if per_access > 0 else float("nan")
        return {"probe": label, "opcode": sass_opcode, "pipe": pipe, "constant_operands": bool(constant),
                "targets_per_iteration": targets, "measured_pj_per_access": probe.energy,
                "predicted_pj_per_access": predicted, "pipe_pj": pipe_pj, "opcode_pj": opcode_pj,
                "ratio": opcode_pj / pipe_pj, "loop": loop,
                "pipe_counts_per_access": {e: probe.counts[e] for e in probe.counts if e.startswith("inst")}}

    first = {}
    for code in OPCODES:
        for constant in (0, 1):
            row = price(code, constant, {})
            if row:
                first[row["opcode"] + (" const" if constant else "")] = {"pj": row["opcode_pj"], "pipe_pj": row["pipe_pj"]}
    helpers = {op: entry for op, entry in first.items() if op in HELPER_PROBES.values()}
    if verbose:
        print(f"{'probe':<30}{'pipe':<24}{'targets/iter':>13}{'measured':>10}{'predicted':>10}{'pipe pJ':>9}"
              f"{'opcode pJ':>10}{'ratio':>7}{'1st pass':>9}")
    for code in list(OPCODES) + list(REUSE_OPCODES):
        for constant in (0, 1):
            row = price(code, constant, helpers)
            if row is None:
                continue
            row["first_pass_pj"] = price(code, constant, {})["opcode_pj"]
            rows.append(row)
            if verbose:
                print(f"{row['probe']:<30}{row['pipe']:<24}{row['targets_per_iteration']:>13}"
                      f"{row['measured_pj_per_access']:>10.2f}{row['predicted_pj_per_access']:>10.2f}"
                      f"{row['pipe_pj']:>9.0f}{row['opcode_pj']:>10.0f}{row['ratio']:>7.2f}{row['first_pass_pj']:>9.0f}")
            if code in REUSE_OPCODES:
                continue
            if not constant:
                table[row["opcode"]] = {"pj": row["opcode_pj"], "pipe": row["pipe"], "pipe_pj": row["pipe_pj"],
                                        "probe": row["probe"]}
            elif row["opcode"] in table:
                table[row["opcode"]]["constant_operands_pj"] = row["opcode_pj"]
    if verbose and any("constant_operands_pj" in entry for entry in table.values()):
        print("\nwith almost constant operands / random operands:")
        for opcode, entry in table.items():
            if "constant_operands_pj" in entry:
                print(f"  {opcode:<10}{entry['constant_operands_pj'] / entry['pj']:.2f}")
    return first, rows, table


if __name__ == "__main__":
    main()
