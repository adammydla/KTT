#!/usr/bin/env python3
"""Turns AccelWattch's microbenchmarks into one KTT kernel file, for --validate accelwattch.

AccelWattch (Kandiah et al., MICRO 2021) calibrated its power model with about 100 CUDA microbenchmarks, published in
Accel-Sim's gpu-app-collection (branch release-accelwattch, src/cuda/accelwattch-ubench). They were written by other
people, for power modelling, so they are a held-out set that nobody here designed: EnergyProbes' model, unchanged,
predicts their energy from their counters, and --validate measures it.

This script reads the sources (they are not part of this repository; download them yourself) and writes, into OUTDIR:

    AccelWattch.cu     every usable benchmark's kernel under `#if AW_BENCH == i`, renamed awKernel, with its macros
                       and local headers inlined; scalar parameters become constants (iterations: AW_ITER,
                       read at run time from a __device__ variable)
    accelwattch.tsv    one line per benchmark: index, name, grid (blocks), block (threads), iterations, pointers,
                       bytes (the largest allocation of the program; EnergyProbes sizes its buffers to the largest)

A benchmark is used when its program launches exactly one kernel (an init kernel <<<1,1>>> aside is not allowed either),
uses no texture references (removed in CUDA 12), has no template kernel, and every scalar argument and the launch
dimensions can be computed from its macros (`iterations` is set here; other command-line values with --set, e.g.
`--set blocks=82 --set div=1,8,32`: a list makes one variant per value). Skipped benchmarks are listed with the reason.
The pointer arguments get buffers of random floats in [1, 2) (EnergyProbes sets them up) of the largest allocation of
the benchmarks written (at least 16 MiB); programs that allocate more than --max-bytes are skipped. __constant__ arrays
stay unfilled (zeros).

    python3 tools/accelwattch_import.py SRC OUTDIR [--set NAME=V1[,V2..]] [--iterations-from RUN_validation.csv]

SRC is the accelwattch-ubench folder. With --iterations-from, the iterations of each benchmark are scaled from a
previous --validate accelwattch run so that one launch takes about --target-ms (the first run uses --iterations).
"""
import argparse
import csv
import os
import re
import sys

FAMILIES = ["functional_benchmarks", "memories_benchmarks", "mix_benchmarks", "static_power_modeling"]
POINTERS = 4               # awKernel always takes this many pointers (unused ones padded)
MIN_BYTES = 16 << 20       # the smallest buffer EnergyProbes allocates per pointer (ValidationKernels.cpp AddAccelWattch)
SCALAR_TYPES = r"(?:const\s+)?(?:unsigned\s+long\s+long|unsigned\s+long|unsigned\s+int|unsigned|long\s+long|long|int|" \
               r"uint64_t|uint32_t|int64_t|int32_t|size_t|float|double)"


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def matching_brace(text, start):
    """Index after the brace that closes the one at `start`."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced braces")


def defines(text):
    result = {}
    for name, value in re.findall(r"^\s*#define\s+(\w+)[ \t]+([^\n]+)", text, flags=re.M):
        result[name] = value.strip()
    return result


def evaluate(expression, macros, variables):
    """Value of a simple integer C expression over macros and known variables; None if it has anything else."""
    text = expression
    for _ in range(10):
        changed = False
        for name in re.findall(r"[A-Za-z_]\w*", text):
            if name in variables:
                text = re.sub(rf"\b{name}\b", f"({variables[name]})", text)
                changed = True
            elif name in macros:
                text = re.sub(rf"\b{name}\b", f"({macros[name]})", text)
                changed = True
        if not changed:
            break
    text = re.sub(r"sizeof\s*\(\s*(?:unsigned\s+)?(?:int|float|unsigned|uint32_t)\s*\)", "4", text)
    text = re.sub(r"sizeof\s*\(\s*(?:unsigned\s+long\s+long|unsigned\s+long|long|double|uint64_t)\s*\)", "8", text)
    text = re.sub(r"\b(\d+)[uUlL]+\b", r"\1", text)
    if not re.fullmatch(r"[\d\s+\-*/%()]+", text):
        return None
    try:
        return int(eval(text.replace("/", "//"), {"__builtins__": {}}))
    except Exception:
        return None


def inline_includes(text, folders, seen=None):
    """Replaces local #include lines with the files' text (found in `folders`); drops system headers."""
    seen = seen if seen is not None else set()

    def replace(match):
        name = match.group(1).strip()
        for folder in folders:
            path = os.path.normpath(os.path.join(folder, name))
            if os.path.isfile(path):
                if path in seen:
                    return ""
                seen.add(path)
                with open(path, errors="replace") as file:
                    return inline_includes(file.read(), [os.path.dirname(path)] + folders, seen)
        return ""

    return re.sub(r'^\s*#include\s*[<"]([^>"]+)[>"][^\n]*$', replace, text, flags=re.M)


def device_part(text):
    """Everything a kernel may need from the file: macros, __device__ functions and __constant__/__device__ variables."""
    parts = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if re.match(r"\s*#\s*(?:define|undef)\b", lines[i]):
            macro = [lines[i]]
            while macro[-1].rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
                macro.append(lines[i])
            parts.append("\n".join(macro))
        i += 1
    for match in re.finditer(r"(?:__device__|__constant__)[^;{(]*(?:\([^)]*\))?\s*[;{]", text):
        start = match.start()
        if text[match.end() - 1] == "{":
            parts.append(text[start:matching_brace(text, match.end() - 1)])
        else:
            parts.append(text[start:match.end()])
    return "\n".join(parts)


def convert(path, root, settings, max_bytes):
    """([entries], None) for a usable benchmark (one entry per variant of --set values it uses), (None, reason)
    otherwise. entry: name, kernel text, grid, block, pointers, bytes."""
    with open(path, errors="replace") as file:
        raw = file.read()
    folder = os.path.dirname(path)
    text = strip_comments(inline_includes(raw, [folder, os.path.join(folder, "include"), os.path.join(folder, "..", "include"),
                                                 os.path.join(root, "include")]))
    if re.search(r"\btexture\s*<|\btex1D|\btex2D|cudaBindTexture", text):
        return None, "texture references (removed in CUDA 12)"
    main_at = text.find("int main")
    if main_at < 0:
        return None, "no main"
    main = text[main_at:]
    launches = re.findall(r"(\w+)\s*<<<([^>]*)>>>\s*\(([^;]*)\)\s*;", main)
    kernels = sorted({name for name, _, _ in launches})
    if len(kernels) != 1:
        return None, f"launches {len(kernels)} kernels ({', '.join(kernels)})"
    name, dims, call = launches[0]
    match = re.search(rf"(template\s*<[^>]*>\s*)?__global__\s+(?:static\s+)?void\s+{name}\s*\(([^)]*)\)\s*\{{", text)
    if not match:
        return None, f"kernel {name} not found"
    if match.group(1):
        return None, "template kernel"
    macros = defines(text)
    # Command-line values (argv) the program reads into variables, other than iterations: from --set.
    used = [name for name in settings if re.search(rf"\b{name}\s*=\s*ato\w*\s*\(\s*argv", main)]
    variants = [{}]
    for name in used:
        variants = [dict(v, **{name: value}) for v in variants for value in settings[name]]
    entries = []
    for variant in variants:
        entry, reason = convert_variant(text, main, name, dims, call, match, macros, variant, max_bytes)
        if reason:
            return None, reason
        suffix = "".join(f" {k}={v}" for k, v in variant.items())
        family = os.path.relpath(path, root).split(os.sep)[0].replace("_benchmarks", "").replace("_modeling", "")
        entry["name"] = f"{family}/{os.path.basename(os.path.dirname(path))}{suffix}"
        entries.append(entry)
    return entries, None


def convert_variant(text, main, name, dims, call, match, macros, variant, max_bytes):
    variables = dict(variant)
    for var, expression in re.findall(r"\b(?:int|unsigned|size_t|long)\s+(\w+)\s*=\s*([^;]+);", main):
        if var != "iterations" and var not in variant:
            value = evaluate(expression, macros, variables)
            if value is not None:
                variables[var] = value
    # Launch dimensions: dim3 NAME(x, y) or an integer variable.
    grid_name, block_name = [d.strip() for d in dims.split(",")[:2]]

    def dimension(token):
        found = re.search(rf"dim3\s+{re.escape(token)}\s*\(([^)]*)\)", main)
        if found:
            values = [evaluate(v, macros, variables) for v in found.group(1).split(",")]
            if any(v is None for v in values):
                return None
            product = 1
            for v in values:
                product *= v
            return product
        return evaluate(token, macros, variables)

    grid, block = dimension(grid_name), dimension(block_name)
    if not grid or not block:
        return None, f"launch dimensions not computable ({grid_name}, {block_name})"
    if block > 1024:
        return None, f"block of {block} threads"
    # Parameters: pointers stay; scalars become constants from the call's arguments.
    params = [p.strip() for p in match.group(2).split(",") if p.strip()]
    args = [a.strip() for a in call.split(",")]
    if len(args) != len(params):
        return None, "call and parameter lists differ"
    pointers, constants = [], []
    for param, arg in zip(params, args):
        if "*" in param:
            pointers.append(param)
            continue
        scalar = re.fullmatch(rf"({SCALAR_TYPES})\s+(\w+)", param)
        if not scalar:
            return None, f"parameter {param!r}"
        if arg == "iterations":
            # A runtime value, as the program's argument was: from a __device__ variable the compiler cannot
            # assume (with a compile-time constant it folded whole loops, e.g. integer adds into one multiply).
            constants.append(f"const {scalar.group(1).replace('const ', '')} {scalar.group(2)} = awIterations;")
        else:
            value = evaluate(arg, macros, variables)
            if value is None:
                return None, f"scalar argument {arg!r} not computable"
            constants.append(f"const {scalar.group(1).replace('const ', '')} {scalar.group(2)} = {value};")
    if len(pointers) > POINTERS:
        return None, f"{len(pointers)} pointers"
    # The largest allocation sets the buffers' size.
    largest = MIN_BYTES
    for size in re.findall(r"cudaMalloc\s*\(\s*\([^)]*\)\s*&\s*\w+\s*,\s*([^)]+)\)", main):
        value = evaluate(size, macros, variables)
        if value is not None:
            if value > max_bytes:
                return None, f"allocates {value} bytes (more than --max-bytes)"
            largest = max(largest, value)
    body_start = match.end() - 1
    body_end = matching_brace(text, body_start)
    body = text[body_start + 1:body_end - 1]
    # NVRTC (KTT compiles with it) knows asm and __asm__, not the __asm that nvcc also takes.
    body = re.sub(r"\b__asm\b(?!__)", "asm", body)
    while len(pointers) < POINTERS:
        pointers.append(f"void* awUnused{len(pointers)}")
    kernel = "\n".join([
        device_part(text[:match.start()]),
        'extern "C" __global__ void awKernel(' + ", ".join(pointers) + ")",
        "{",
        "    " + "\n    ".join(constants),
        body,
        "}",
    ])
    return {"kernel": kernel, "grid": grid, "block": block, "pointers": len([p for p in params if "*" in p]),
            "bytes": largest}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="the accelwattch-ubench folder of gpu-app-collection (branch release-accelwattch)")
    parser.add_argument("outdir")
    parser.add_argument("--iterations", type=int, default=100, help="iterations of every benchmark (default 100)")
    parser.add_argument("--iterations-from", help="a --validate accelwattch run's CSV: scale each benchmark's iterations "
                        "so that one launch takes about --target-ms")
    parser.add_argument("--target-ms", type=float, default=20.0)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=V1[,V2..]",
                        help="a value the programs read from the command line (e.g. blocks=82); a list makes variants")
    parser.add_argument("--max-bytes", type=float, default=2 ** 31, help="skip programs that allocate more (default 2 GiB)")
    parser.add_argument("--exclude", action="append", default=[], metavar="FOLDER",
                        help="skip this benchmark (its folder name), e.g. one that faults: a fault ends the whole run")
    args = parser.parse_args()
    settings = {}
    for item in args.set:
        key, _, values = item.partition("=")
        settings[key.strip()] = [int(v) for v in values.split(",") if v.strip()]

    previous = {}
    if args.iterations_from:
        with open(args.iterations_from) as file:
            for row in csv.DictReader(file):
                if row.get("valid") == "1" and row.get("repeat") in (None, "", "0"):
                    found = re.search(r"AW_BENCH=(\d+)", row["configuration"])
                    if found:
                        previous[int(found.group(1))] = float(row["kernel_s"])

    entries, skipped = [], []
    for family in FAMILIES:
        base = os.path.join(args.source, family)
        if not os.path.isdir(base):
            continue
        for folder in sorted(os.listdir(base)):
            files = sorted(f for f in os.listdir(os.path.join(base, folder)) if f.endswith(".cu")) \
                if os.path.isdir(os.path.join(base, folder)) else []
            if folder == "include":
                continue
            if folder in args.exclude:
                skipped.append((f"{family}/{folder}", "--exclude"))
                continue
            if len(files) != 1:
                skipped.append((f"{family}/{folder}", f"{len(files)} .cu files"))
                continue
            found, reason = convert(os.path.join(base, folder, files[0]), args.source, settings, args.max_bytes)
            if found:
                entries.extend(found)
            else:
                skipped.append((f"{family}/{folder}", reason))

    old_iterations = {}
    tsv = os.path.join(args.outdir, "accelwattch.tsv")
    if previous and os.path.isfile(tsv):
        with open(tsv) as file:
            for line in file:
                if line.strip() and not line.startswith("#"):
                    fields = line.split("\t")
                    old_iterations[fields[1]] = (int(fields[0]), int(fields[4]))

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "AccelWattch.cu"), "w") as cu, open(tsv, "w") as table:
        cu.write("// Generated by tools/accelwattch_import.py from AccelWattch's microbenchmarks (Accel-Sim gpu-app-collection,\n"
                 "// release-accelwattch, src/cuda/accelwattch-ubench; Copyright (c) 2018-2021 the AccelWattch authors, BSD-3).\n"
                 "// One benchmark per AW_BENCH value; do not edit, rerun the script.\n"
                 "#if defined(__CUDACC_RTC__)\ntypedef unsigned int uint32_t;\ntypedef int int32_t;\n"
                 "typedef unsigned long uint64_t;\ntypedef long int64_t;\n#else\n#include <stdint.h>\n#endif\n")
        table.write("# index\tname\tgrid\tblock\titerations\tpointers\tbytes\n")
        for index, entry in enumerate(entries):
            iterations = args.iterations
            if entry["name"] in old_iterations:
                old_index, old = old_iterations[entry["name"]]
                if old_index in previous and previous[old_index] > 0:
                    iterations = max(1, int(round(old * args.target_ms * 1e-3 / previous[old_index])))
                else:
                    iterations = old
            cu.write(f"\n#if AW_BENCH == {index} // {entry['name']}\n#define AW_ITER {iterations}\n"
                     f"__device__ unsigned long long awIterations = AW_ITER;\n{entry['kernel']}\n#endif\n")
            table.write(f"{index}\t{entry['name']}\t{entry['grid']}\t{entry['block']}\t{iterations}\t{entry['pointers']}"
                        f"\t{entry['bytes']}\n")
    print(f"{len(entries)} benchmarks written to {args.outdir}; {len(skipped)} skipped:")
    for name, reason in skipped:
        print(f"  {name}: {reason}")


if __name__ == "__main__":
    sys.exit(main())
