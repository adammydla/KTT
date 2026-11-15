#!/usr/bin/env python3
"""SASS checks of the probe kernels, compiled the way the program compiles them.

Every probe variant (the parameter combinations the host uses) is compiled with NVRTC, with the parameters as #define
lines in front of kernels/Common.cuh and the probe's file, as KTT does, then assembled with ptxas. Checks:

1. every variant compiles without spilling registers; a variant whose registers let fewer than BLOCKS_PER_SM blocks
   fit on an SM at once is reported with its occupancy (it runs, but the blocks of an SM then share stripes; the program
   notes it in the rows, see ResidentBlocks in Hardware.cpp);
2. the footprints of a probe differ in at most one immediate (or not at all where the footprint is an argument), so
   that differences between footprints come from the memory hierarchy;
3. with --against OLD: every variant has the same SASS as the same kernel in OLD, for checking that a change of the
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
    for op in [0, 5]:
        result.append(("Scatter.cu", "probeScatter", {"LOAD_OP": op, "WIDTH": 16}, "FOOTPRINT"))
        for width in [4, 8]:
            result.append(("Partial.cu", "probePartial", {"LOAD_OP": op, "WIDTH": width}, "FOOTPRINT"))
        result.append(("Partial.cu", "probePartial", {"LOAD_OP": op, "WIDTH": 4, "PARTIAL_WORDS": 1}, "FOOTPRINT"))
    for op in [0, 1, 2]:
        result.append(("Shared.cu", "probeShared", {"SHARED_OP": op, "SMEM_BYTES": 8192}, None))
    for op, width in itertools.product([0, 1], [4, 16]):
        result.append(("Shared.cu", "probeSharedBanks", {"SHARED_OP": op, "SMEM_BYTES": 8192, "SHARED_WIDTH": width}, None))
    result.append(("Constant.cu", "probeConstant", {}, "CONST_BYTES"))
    for op in [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]:
        result.append(("Compute.cu", "probeCompute", {"COMPUTE_OP": op}, None))
    for op, width in itertools.product(range(3), [4, 16]):
        result.append(("Write.cu", "probeWrite", {"WRITE_OP": op, "WIDTH": width}, "LOCAL" if op == 2 else "FOOTPRINT"))
    for chains in [1, 16]:
        result.append(("Latency.cu", "probeLatency", {"CHAINS": chains, "WIDTH": 4}, "FOOTPRINT"))
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


def sass(nvrtc, cuda, source, arch, function, work):
    """SASS lines and registers per thread of `function`."""
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
    lines = [m.group(1) for m in (re.match(r"\s*/\*[0-9a-f]{4}\*/\s*(.*?)\s*;", l) for l in dump.splitlines()) if m]
    if spills.get(function):
        raise RuntimeError(f"{function} spills {spills[function]} bytes of registers")
    return lines, registers.get(function)


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

    with tempfile.TemporaryDirectory() as work:
        for arch in (a.strip().lower().replace("sm_", "") for a in arguments.arch.split(",")):  # 86 or sm_86
            checked = 0
            for file, function, parameters, footprint in variants():
                if function == "probeCompute" and parameters["COMPUTE_OP"] == 5 and int(arch) < 70:
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

    print((f"{occupancy} variants with fewer than {COMMON['BLOCKS_PER_SM']} blocks per SM (they run; the rows note it)\n"
           if occupancy else "") + ("all checks passed" if problems == 0 else f"{problems} problems"))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
