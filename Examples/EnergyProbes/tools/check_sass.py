#!/usr/bin/env python3
"""SASS checks of the probe kernels, compiled the way the program compiles them.

Every probe variant (the parameter combinations the host uses) is compiled with NVRTC, with the parameters as #define
lines in front of kernels/Common.cuh and the probe's file, as KTT does, then assembled with ptxas. Checks:

1. every variant compiles without spilling registers; a variant whose registers let fewer than BLOCKS_PER_SM blocks
   fit on an SM at once is reported with its occupancy (it runs, but the blocks of an SM then share stripes; the program
   notes it in the rows, see ResidentBlocks in Hardware.cpp);
2. the footprints of a probe differ in at most one immediate (or not at all where the footprint is an argument), so
   that differences between footprints come from the memory hierarchy;
3. the optional tensor probes compile to their tensor instruction (IMMA, DMMA, HMMA.SP, QMMA), on the architectures the
   planner runs them on; the opcode probes have 64 of their opcode per loop iteration, and the operand reuse probes
   their number of .reuse operands per FFMA (printed with the same-bank register pairs);
4. with --against OLD: every variant has the same SASS as the same kernel in OLD, for checking that a change of the
   sources leaves the measured code unchanged. OLD is an older kernels/ directory (a copy taken before the change), or
   one .cu with all kernels (the layout before the probes were split into files). Kernels OLD does not have, variants
   that set a parameter OLD does not use, and variants OLD does not compile (a new value of a parameter) are new code and
   are skipped.

    python3 tools/check_sass.py [--arch 70,80,86,90] [--against old_kernels/] [--sms 132]

Needs libnvrtc, ptxas and cuobjdump of a CUDA toolkit (CUDA_PATH, default /usr/local/cuda); no GPU.
"""

import argparse
import ctypes
import itertools
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.join(HERE, "..", "kernels")
COMMON = {"BLOCK_SIZE": 256, "BLOCKS_PER_SM": 4, "UNROLL": 16, "BURST_ACCESSES": 64, "GRID_STRIPES": 512}
# GRID_STRIPES is the power of two at or above the number of blocks (SMs x BLOCKS_PER_SM, see ChooseFootprints): 512
# for 65-128 SMs at 4 blocks per SM; --sms sets it for another GPU (an H100 with 132 SMs needs 1024).
FOOTPRINTS = [32768, 2097152, 268435456]
# The footprint sweeps: name -> (parameter, values it is compiled with). Local memory (probeWrite, WRITE_OP 2) is per SM
# and at most LOCAL_FRAME_BYTES per thread.
SWEEPS = {"FOOTPRINT": ("FOOTPRINT", FOOTPRINTS), "CONST_BYTES": ("CONST_BYTES", [1024, 32768]),
          "LOCAL": ("FOOTPRINT", [32768, 1048576])}


# The optional "tensor" probes (COMPUTE_OP 13-16): the SASS opcode each must compile to, and the architectures the planner
# runs it on (ProbePlan.cpp). FP8 mma.sync compiles on sm_90 and sm_100 too, but into conversions and HMMA (emulated).
TENSOR_OPCODES = {13: "IMMA", 14: "DMMA", 15: "HMMA.SP", 16: "QMMA"}
# The opcode probes (COMPUTE_OP 17, Common.h OpcodeCount, OpcodeLastWithConstant) and the SASS opcode each must contain 64
# times per loop iteration (16 steps of 4 chains), except IADD3 and LOP3: ptxas folds two steps into one three-input
# instruction (32 per iteration, plus one of the loop's own). IADD3 with constant operands is not checked.
OPCODE_COUNT = 19
OPCODE_LAST_WITH_CONSTANT = 7
OPCODE_TARGETS = {1: "FADD", 2: "FMUL", 3: "FFMA", 4: "DADD", 5: "DMUL", 6: "DFMA", 7: "IADD3", 8: "LOP3.LUT",
                  9: "SHF.L.W.U32.HI", 10: "MUFU.EX2", 11: "MUFU.LG2", 12: "MUFU.SIN", 13: "MUFU.RCP", 14: "MUFU.SQRT",
                  15: "MUFU.TANH", 16: "I2FP.F32.S32", 17: "F2I.NTZ", 18: "POPC",
                  19: "F2F.F64.F32"}
# The optional "reuse" probes (COMPUTE_OP 17, COMPUTE_OPCODE 20-22, Common.h OpcodeReuseFirst/Last): 64 FFMA per loop
# iteration with the same operand values, and the range of operands marked .reuse per FFMA each must have (ptxas on sm_86
# and sm_89/90: 0, 0.75, 1.5; on sm_70/75/80 it reorders some FFMAs: 0.64-0.66 and 1.31-1.34).
OPCODE_REUSE = {20: (0.0, 0.0), 21: (0.6, 0.8), 22: (1.25, 1.55)}


def planned(op, arch):
    """Whether the planner runs COMPUTE_OP `op` on sm_`arch` (HMMA from sm_70, IMMA/DMMA/sparse from sm_80, FP8 on sm_89
    and sm_12x only)."""
    arch = int(arch)
    if op == 5:
        return arch >= 70
    if op in (13, 14, 15):
        return arch >= 80
    if op == 16:
        return arch == 89 or arch // 10 == 12
    return True


def resident_blocks(registers, block_size=None, blocks_per_sm=None):
    """Blocks per SM that fit at once by the register file (as ResidentBlocks in Hardware.cpp and cuda_occupancy.h for
    sm_70 and newer: 64 K registers in 4 partitions, allocated per warp in units of 256)."""
    block_size = block_size or COMMON["BLOCK_SIZE"]
    per_warp = (registers * 32 + 255) // 256 * 256
    return (65536 // 4) // per_warp * 4 // ((block_size + 31) // 32)


def variants():
    """(file, function, parameters, footprint sweep of SWEEPS or None) of every variant the host compiles (ProbePlan.cpp).
    Arguments that only name a configuration (duty, offsets, conflict degree, runs) do not change the code."""
    result = [("Baseline.cu", "probeSleep", {}, None), ("Baseline.cu", "probeSpin", {}, None)]
    for op, width in itertools.product([0, 2, 5], [4, 8, 16]):
        result.append(("Global.cu", "probeGlobal", {"LOAD_OP": op, "WIDTH": width}, "FOOTPRINT"))
    for op in [6, 7]:
        result.append(("Global.cu", "probeGlobal", {"LOAD_OP": op, "WIDTH": 4}, "FOOTPRINT"))
    # The optional "mix" and "saturate" groups: work added to every load.
    for width in [4, 16]:
        for extra in ({"MIX_FFMA": 4}, {"MIX_FFMA": 16}, {"MIX_L1": 2}, {"MIX_L1": 8}, {"MIX_SHARED": 2}, {"MIX_SHARED": 8}):
            result.append(("Global.cu", "probeGlobal", dict({"LOAD_OP": 0, "WIDTH": width}, **extra), "FOOTPRINT"))
    for op in [0, 5]:
        result.append(("Scatter.cu", "probeScatter", {"LOAD_OP": op, "WIDTH": 16}, "FOOTPRINT"))
        for width in [4, 8]:
            result.append(("Partial.cu", "probePartial", {"LOAD_OP": op, "WIDTH": width}, "FOOTPRINT"))
        result.append(("Partial.cu", "probePartial", {"LOAD_OP": op, "WIDTH": 4, "PARTIAL_WORDS": 1}, "FOOTPRINT"))
    for op in [0, 1, 2]:
        result.append(("Shared.cu", "probeShared", {"SHARED_OP": op, "SMEM_BYTES": 8192}, None))
    for op, width in itertools.product([0, 1], [4, 16]):
        result.append(("Shared.cu", "probeSharedBanks", {"SHARED_OP": op, "SMEM_BYTES": 8192, "SHARED_WIDTH": width}, None))
    # The optional "broadcast" group: 2, 16 or 32 lanes per address.
    for width, group in itertools.product([4, 16], [2, 16, 32]):
        result.append(("Shared.cu", "probeSharedBanks", {"SHARED_OP": 0, "SMEM_BYTES": 8192, "SHARED_WIDTH": width,
                                                         "SHARED_GROUP": group}, None))
    result.append(("Constant.cu", "probeConstant", {}, "CONST_BYTES"))
    for op in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12] + list(TENSOR_OPCODES):
        result.append(("Compute.cu", "probeCompute", {"COMPUTE_OP": op}, None))
    for op, lanes in itertools.product([1, 3], [1, 8]):
        result.append(("Compute.cu", "probeCompute", {"COMPUTE_OP": op, "COMPUTE_LANES": lanes}, None))
    # The optional "opcodes" group (COMPUTE_OP 17): every opcode, the first seven also with almost constant operands.
    for opcode in range(1, OPCODE_COUNT + 1):
        for constant in ([0, 1] if opcode <= OPCODE_LAST_WITH_CONSTANT else [0]):
            result.append(("Compute.cu", "probeCompute", {"COMPUTE_OP": 17, "COMPUTE_OPCODE": opcode,
                                                          "COMPUTE_CONSTANT": constant}, None))
    # The optional "reuse" group (opcode 3 is in the list above).
    for opcode in OPCODE_REUSE:
        result.append(("Compute.cu", "probeCompute", {"COMPUTE_OP": 17, "COMPUTE_OPCODE": opcode}, None))
    for op, width in itertools.product(range(3), [4, 16]):
        result.append(("Write.cu", "probeWrite", {"WRITE_OP": op, "WIDTH": width}, "LOCAL" if op == 2 else "FOOTPRINT"))
    for chains in [1, 4, 16]:
        result.append(("Latency.cu", "probeLatency", {"CHAINS": chains, "WIDTH": 4}, "FOOTPRINT"))
    result.append(("Latency.cu", "probeLatency", {"CHAINS": 1, "ROWS": 1, "WIDTH": 4}, "FOOTPRINT"))
    return result


class Nvrtc:
    def __init__(self, cuda):
        self.lib = ctypes.CDLL(os.path.join(cuda, "lib64", "libnvrtc.so"))

    def ptx(self, source, arch):
        program = ctypes.c_void_p()
        self.lib.nvrtcCreateProgram(ctypes.byref(program), source.encode(), b"probe.cu", 0, None, None)
        options = (ctypes.c_char_p * 1)(f"--gpu-architecture=compute_{arch}".encode())
        status = self.lib.nvrtcCompileProgram(program, 1, options)
        if status != 0:
            size = ctypes.c_size_t()
            self.lib.nvrtcGetProgramLogSize(program, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            self.lib.nvrtcGetProgramLog(program, log)
            self.lib.nvrtcDestroyProgram(ctypes.byref(program))
            raise RuntimeError(log.value.decode())
        size = ctypes.c_size_t()
        self.lib.nvrtcGetPTXSize(program, ctypes.byref(size))
        ptx = ctypes.create_string_buffer(size.value)
        self.lib.nvrtcGetPTX(program, ptx)
        self.lib.nvrtcDestroyProgram(ctypes.byref(program))
        return ptx.value.decode()


def defines(parameters):
    return "".join(f"#define {name} {value}\n" for name, value in {**COMMON, **parameters}.items())


def probe_source(file, parameters, directory=KERNELS):
    """The program the host builds for a probe (KernelSource in ProbeKernels.cpp) with KTT's #defines in front."""
    with open(os.path.join(directory, "Common.cuh")) as common, open(os.path.join(directory, file)) as probe:
        return (defines(parameters) + '#line 1 "Common.cuh"\n' + common.read() + f'\n#line 1 "{file}"\n'
                + probe.read())


def sass(nvrtc, cuda, source, arch, function, work, addresses=False):
    """SASS lines and registers per thread of `function` (with addresses: (address, line) pairs)."""
    ptx_file = os.path.join(work, "probe.ptx")
    cubin = os.path.join(work, "probe.cubin")
    with open(ptx_file, "w") as file:
        file.write(nvrtc.ptx(source, arch))
    result = subprocess.run([os.path.join(cuda, "bin", "ptxas"), "-v", f"-arch=sm_{arch}", "-o", cubin, ptx_file],
                            check=True, capture_output=True, text=True)
    registers, spills, current = {}, {}, None
    for line in result.stderr.splitlines():
        match = re.search(r"Compiling entry function '(\w+)'", line)
        current = match.group(1) if match else current
        match = re.search(r"Used (\d+) registers", line)
        if match and current:
            registers[current] = int(match.group(1))
        match = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if match and current:
            spills[current] = int(match.group(1)) + int(match.group(2))
    dump = subprocess.run([os.path.join(cuda, "bin", "cuobjdump"), "-sass", "-fun", function, cubin], check=True,
                          capture_output=True, text=True).stdout
    matches = [m for m in (re.match(r"\s*/\*([0-9a-f]{4,})\*/\s*(.*?)\s*;", l) for l in dump.splitlines()) if m]
    if spills.get(function):
        raise RuntimeError(f"{function} spills {spills[function]} bytes of registers")
    if addresses:
        return [(int(m.group(1), 16), m.group(2)) for m in matches], registers.get(function)
    return [m.group(2) for m in matches], registers.get(function)


def opcode_of(line):
    """The opcode of a SASS line with its modifiers (FFMA, MUFU.EX2, LOP3.LUT), without a guard predicate."""
    parts = line.split()
    return parts[1] if parts and parts[0].startswith("@") else (parts[0] if parts else "")


def loop_lines(addressed):
    """The SASS lines of one iteration of a probe's main loop: the instructions between the target and the source of the
    widest backward branch, without those of backward branches nested in it (the idle loop of the duty window, which
    does not run at duty 100)."""
    branches = []
    for address, line in addressed:
        match = re.search(r"\bBRA\s+(?:`?\(?\.?L?_?x?_?)?0x([0-9a-f]+)", line)
        if match and int(match.group(1), 16) <= address:
            branches.append((int(match.group(1), 16), address))
    if not branches:
        return []
    start, end = max(branches, key=lambda b: b[1] - b[0])
    nested = [(a, b) for a, b in branches if start < a and b < end]
    return [line for address, line in addressed
            if start <= address <= end and not any(a <= address <= b for a, b in nested)]


def main_loop(addressed):
    """Opcode counts of one iteration of a probe's main loop (loop_lines)."""
    counts = {}
    for line in loop_lines(addressed):
        opcode = opcode_of(line)
        counts[opcode] = counts.get(opcode, 0) + 1
    return counts


def register_reads(lines, opcode="FFMA"):
    """Per instruction `opcode` in `lines`: (operands marked .reuse, pairs of other source registers in the same bank).
    Banks: register number modulo 2 (two banks per sub-core, Huerta et al. 2025, arXiv 2503.20481); RZ and constants are
    no register reads. A reused operand is read from the operand reuse cache, not from its bank."""
    reused, same_bank, count = 0, 0, 0
    for line in lines:
        if opcode_of(line) != opcode:
            continue
        operands = [o.strip() for o in line.split(None, 2 if line.startswith("@") else 1)[-1].split(",")][1:]
        count += 1
        reused += sum(1 for o in operands if ".reuse" in o)
        banks = [int(m.group(1)) % 2 for o in operands if ".reuse" not in o
                 for m in [re.match(r"[-|!]*R(\d+)", o)] if m]
        same_bank += sum(1 for i in range(len(banks)) for j in range(i + 1, len(banks)) if banks[i] == banks[j])
    return (reused / count, same_bank / count) if count else (0.0, 0.0)


def immediates_only(a, b):
    # Integer, hexadecimal and floating-point immediates (ptxas on sm_120 builds a 32-bit constant with HFMA2 and two
    # half-precision immediates, e.g. "HFMA2 R5, -RZ, RZ, 0, 0.0001220703125").
    def normalize(line):
        line = re.sub(r"(?<![\w.])-?\d+\.\d+(?:e[-+]?\d+)?\b", "IMM", line)
        return re.sub(r"(?<![\w.])-?\d+\b", "IMM", re.sub(r"\b0x[0-9a-f]+\b", "IMM", line))
    return len(a) == len(b) and all(normalize(x) == normalize(y) for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default="70,80,86,90", help="architectures, e.g. 86 or 70,80,86,90")
    parser.add_argument("--against", help="older kernels/ directory, or one .cu with all kernels; SASS must be identical")
    parser.add_argument("--sms", type=int, default=82, help="SMs of the GPU, for GRID_STRIPES (default 82, an RTX 3090)")
    arguments = parser.parse_args()
    blocks = arguments.sms * COMMON["BLOCKS_PER_SM"]
    COMMON["GRID_STRIPES"] = 1 << max(blocks - 1, 0).bit_length()
    cuda = os.environ.get("CUDA_PATH", "/usr/local/cuda")
    nvrtc = Nvrtc(cuda)
    old_directory = arguments.against if arguments.against and os.path.isdir(arguments.against) else None
    old_source = open(arguments.against).read() if arguments.against and old_directory is None else None

    def old_program(file, function, parameters, full):
        """The program of a variant from OLD, or None if OLD has no such kernel (or file) or does not use one of its
        parameters."""
        if old_directory and not os.path.exists(os.path.join(old_directory, file)):
            return None
        body = probe_source(file, {}, old_directory) if old_directory else old_source
        if not re.search(rf"\b{function}\s*\(", body) or any(not re.search(rf"\b{name}\b", body) for name in parameters):
            return None
        return probe_source(file, full, old_directory) if old_directory else defines(full) + old_source

    problems = 0
    occupancy = 0
    reuse_report = []

    with tempfile.TemporaryDirectory() as work:
        for arch in (a.strip().lower().replace("sm_", "") for a in arguments.arch.split(",")):  # 86 or sm_86
            checked = 0
            for file, function, parameters, footprint in variants():
                if function == "probeCompute" and not planned(parameters["COMPUTE_OP"], arch):
                    continue
                name, values = SWEEPS[footprint] if footprint else (None, [None])
                compiled = []
                for value in values:
                    full = {**parameters, **({name: value} if footprint else {})}
                    label = f"sm_{arch} {function} " + " ".join(f"{k}={v}" for k, v in full.items())
                    try:
                        lines, registers = sass(nvrtc, cuda, probe_source(file, full), arch, function, work)
                    except (RuntimeError, subprocess.CalledProcessError) as error:
                        print(f"  {label}: does not compile: {str(error).strip()[:300]}")
                        problems += 1
                        continue
                    compiled.append(lines)
                    if function == "probeCompute" and parameters.get("COMPUTE_OP") == 17 \
                            and not (parameters["COMPUTE_OPCODE"] == 7 and parameters["COMPUTE_CONSTANT"]):
                        addressed, _ = sass(nvrtc, cuda, probe_source(file, full), arch, function, work, addresses=True)
                        target = OPCODE_TARGETS.get(parameters["COMPUTE_OPCODE"], "FFMA")
                        found = main_loop(addressed).get(target, 0)
                        wanted = 33 if parameters["COMPUTE_OPCODE"] in (7, 8) else 64
                        if found != wanted:
                            print(f"  {label}: {found} {target} per loop iteration, not {wanted}")
                            problems += 1
                        if parameters["COMPUTE_OPCODE"] in OPCODE_REUSE \
                                or (parameters["COMPUTE_OPCODE"] == 3 and not parameters["COMPUTE_CONSTANT"]):
                            low, high = OPCODE_REUSE.get(parameters["COMPUTE_OPCODE"], (0.0, 0.0))
                            reused, same_bank = register_reads(loop_lines(addressed))
                            reuse_report.append(f"sm_{arch} opcode {parameters['COMPUTE_OPCODE']}: {reused:.2f} reused "
                                                f"operands, {same_bank:.2f} same-bank register pairs per FFMA")
                            if not low - 1e-9 <= reused <= high + 1e-9:
                                print(f"  {label}: {reused:.2f} reused operands per FFMA, not {low}-{high}")
                                problems += 1
                    expected = TENSOR_OPCODES.get(parameters.get("COMPUTE_OP")) if function == "probeCompute" else None
                    if expected and not any(re.search(rf"\b{re.escape(expected)}\b", line) for line in lines):
                        print(f"  {label}: no {expected} instruction in the SASS (emulated or removed)")
                        problems += 1
                    fit = resident_blocks(registers) if registers else 0
                    if registers is None:
                        print(f"  {label}: no register count from ptxas")
                        problems += 1
                    elif fit < COMMON["BLOCKS_PER_SM"]:
                        print(f"  {label}: {registers} registers per thread, {fit} of {COMMON['BLOCKS_PER_SM']} blocks of "
                              f"{COMMON['BLOCK_SIZE']} threads fit per SM")
                        occupancy += 1
                    old = old_program(file, function, parameters, full) if arguments.against else None
                    if old is not None:
                        try:
                            old, old_registers = sass(nvrtc, cuda, old, arch, function, work)
                        except RuntimeError:
                            old = None  # OLD does not compile this variant: new code
                    if old is not None:
                        if old != lines or old_registers != registers:
                            print(f"  {label}: SASS differs from {arguments.against} ({len(old)} vs {len(lines)} "
                                  f"instructions, {old_registers} vs {registers} registers)")
                            problems += 1
                    checked += 1
                if not compiled:
                    continue
                differing = sum(1 for i in range(len(compiled[0]))
                                if len({len(c) for c in compiled}) == 1 and len({c[i] for c in compiled}) > 1)
                if not all(immediates_only(compiled[0], c) for c in compiled[1:]) or differing > 1:
                    print(f"  sm_{arch} {function} {parameters}: footprints differ in {differing} lines")
                    problems += 1
            print(f"sm_{arch}: {checked} variants compiled" + (", compared" if arguments.against else ""))

    if reuse_report:
        print("operand reuse probes (FFMA, main loop):\n  " + "\n  ".join(reuse_report))
    print((f"{occupancy} variants with fewer than {COMMON['BLOCKS_PER_SM']} blocks per SM (they run; the rows note it)\n"
           if occupancy else "") + ("all checks passed" if problems == 0 else f"{problems} problems"))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
