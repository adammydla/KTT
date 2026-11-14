"""The events of the energy model, computed from CUPTI counters (standard library only).

An event is something the model prices: a warp instruction of a pipe, a sector served by a cache level or by DRAM, a
shared-memory wavefront, a nanosecond of DRAM kept awake. events() turns the counters of one row (a probe, or one launch
of a kernel) into event counts; add_busy() adds the one event that needs the duration. energy_model.py fits an energy to
every event, energy_predictor.py prices a kernel with a model file. Both must count events the same way, so the
definitions live here and nowhere else (EnergyModel.h, the C++ predictor, repeats them; tests/ compares the two).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_table  # noqa: E402

def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


# The counter names, their aliases (the same counter under another chip's name; GB202, the RTX 5090, counts DRAM sectors
# as dram__sectors_op_read/_op_write) and the events they feed come from the counter table, tools/counters.json.
ALIASES = counter_table.ALIASES
value_of = counter_table.value_of
read_rows = counter_table.read_rows  # the rows of a run's CSV (repairs rows with a comma in the note)


def counter(row, name):
    """A counter's value; 0 if the row has none. Negative values count as 0: CUPTI reports the two L2 reduction counters
    negative now and then in kernels without reductions (down to -1.8 million sectors in a probe row), which would make
    an event negative."""
    value = value_of(row, name)
    return 0.0 if math.isnan(value) or value < 0 else value


# The lsu pipe's instructions of shared memory and of the global path (smsp__inst_executed_op_<op>.sum).
SHARED_INSTRUCTIONS = counter_table.feeding("inst lsu shared")
GLOBAL_INSTRUCTIONS = counter_table.feeding("inst lsu global")

# Events of the model, computed from the CUPTI counters of one row (warp instructions per pipe, sectors per level). The fp16
# pipe is part of the FMA pipe on GA10x; "other" is everything else (branches, uniform datapath, address unit, ...). Global
# atomics are L2 operations of their own (neither read nor write sectors); a constant load (LDC) is one request to the
# constant cache (IDC), and a miss there is served by the constant L1.5. L2 requests (one per line an instruction touches)
# carry what a transfer between L1 and L2 costs beyond its sectors: coalesced 16-byte accesses send 4 sectors per request,
# scattered sectors one; the locality probes with 32-byte runs separate the two. L2 read miss requests do the same for
# DRAM reads (each one an access to DRAM, of 1 to 4 sectors); for writes see events(). Counters a run did not collect
# count as 0,
# so the validation run must have collected the same counters as the probes.
FMA_HALVES = False  # --fma-halves (a model file carries its own setting)


def events(row, fma_halves=None):
    """Event counts of one row of counters. `fma_halves`: None = the module's FMA_HALVES."""
    if fma_halves is None:
        fma_halves = FMA_HALVES
    total = counter(row, "smsp__inst_executed.sum")
    alu = counter(row, "smsp__inst_executed_pipe_alu.sum")
    fma = counter(row, "smsp__inst_executed_pipe_fma.sum")
    # fp16 runs on the fma pipe on GA10x and newer (counted as its fp16 type), on a pipe of its own on Turing and GA100.
    separate_fp16 = not math.isnan(value_of(row, "smsp__inst_executed_pipe_fp16.sum"))
    fp16 = counter(row, "smsp__inst_executed_pipe_fp16.sum" if separate_fp16 else "smsp__inst_executed_pipe_fma_type_fp16.sum")
    fp64 = counter(row, "smsp__inst_executed_pipe_fp64.sum")
    tensor = counter(row, "smsp__inst_executed_pipe_tensor.sum")
    xu = counter(row, "smsp__inst_executed_pipe_xu.sum")
    lsu = counter(row, "smsp__inst_executed_pipe_lsu.sum")
    tex = counter(row, "smsp__inst_executed_pipe_tex.sum")
    # Branch instructions (BRA) issue to none of the pipes above (on GA102 not to cbu either) and count as "inst other". An
    # event of their own (smsp__inst_executed_op_branch, collected since ProfileFull4) is not identifiable from the probes:
    # every probe loop has one branch per iteration in a fixed proportion to its other instructions, the fit priced it 0
    # and the ffma-loop probe (one BRA per 4 FFMAs) came out 12 % low instead of 6 %.
    # The alu and fma pipes count in active lanes / 32: a lane that is inactive or whose guard predicate is false does no
    # work (the guarded moves of a dynamically indexed register array cost almost nothing when the guard is false in every
    # lane). For fully active warps this is the warp instruction count; runs without the thread counters use that.
    def lanes(name, warp_count):
        value = value_of(row, name)
        return value / 32 if not math.isnan(value) else warp_count
    alu_lanes = lanes("sm__thread_inst_executed_pipe_alu_pred_on.sum", alu)
    fma_lanes = lanes("sm__thread_inst_executed_pipe_fma_pred_on.sum", fma)
    fp16_lanes = lanes("sm__thread_inst_executed_pipe_fp16_pred_on.sum" if separate_fp16
                       else "sm__thread_inst_executed_pipe_fma_type_fp16_pred_on.sum", fp16)
    # A bank conflict replays a shared wavefront for the lanes whose bank another lane took; the conflict counters count
    # exactly these extra wavefronts. A replayed wavefront moves fewer bytes than one of a conflict-free access, so
    # conflicts are events of their own (runs without the conflict counters count none).
    ld_conflicts = counter(row, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
    st_conflicts = counter(row, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum")
    # An atomic that misses L2 reads its line from DRAM and later writes it back: one read and one write miss request.
    # Writes to DRAM are priced by their sectors: L2 writes back every dirty line, whether the store that dirtied it missed
    # L2 or hit a line that a load had brought in (an update in place, a spill). A write miss request priced as the DRAM
    # access, as before, left those write-backs without one, and the write probes that load and store the same lines came
    # out 10-19 % too low. Write miss requests beyond one per four written-back sectors are partial writes (1 to 3
    # sectors per request), which cost a request of their own.
    atomic_misses = counter(row, "lts__t_requests_srcunit_tex_op_atom_lookup_miss.sum") \
        + counter(row, "lts__t_requests_srcunit_tex_op_red_lookup_miss.sum")
    # The lsu pipe's instructions by memory space: shared memory (LDS, STS, ATOMS, LDSM), the global path (LDG, STG, RED,
    # ATOM, generic and local accesses, LDGSTS) and the rest of the pipe. In the shared probes every instruction comes with
    # its wavefronts, so with one lsu event the fit traded the energy of a shared instruction for that of its wavefronts and
    # priced the global instructions too low: in the example kernels the lsu instructions cost about twice the fitted value
    # and the shared wavefronts about half. Runs without these counters count all lsu instructions as "inst lsu other".
    shared_inst = sum(counter(row, name) for name in SHARED_INSTRUCTIONS)
    global_inst = sum(counter(row, name) for name in GLOBAL_INSTRUCTIONS)
    # The fma pipe without fp16 is one event, or with --fma-halves (FMA_HALVES) two: its halves on GA10x (active lanes / 32,
    # as above). fmaheavy executes FP32 and the integer multiply-add (IMAD, which also carries moves and shifts), fmalite
    # FP32 only. In the probes, IMAD runs on the heavy half alone, throughput-bound FP32 (the FFMA probes, the FFMAs of the
    # mixed probes) splits exactly half and half, and the FFMAs of a dependent chain (the MUFU probe) run on the lite half;
    # an fp16 instruction counts in both halves. With one event, the fit prices an FFMA between the FFMA probe and the
    # IMAD probe (8 % under ffma3). The split fixes that and lowers the bias on the example kernels, but it widens their
    # scatter (see docs/FINDINGS.md), so it is not the default. Runs without the half counters have only the one event.
    fma_pipe = fma_lanes if separate_fp16 else max(fma_lanes - fp16_lanes, 0.0)
    lite = value_of(row, "sm__thread_inst_executed_pipe_fmalite_pred_on.sum")
    lite = min(max(lite / 32 - fp16_lanes, 0.0), fma_pipe) if fma_halves and not math.isnan(lite) else 0.0
    return {
        "inst alu": alu_lanes,
        "inst fma (fp32, imad)": fma_pipe - lite,
        "inst fma lite (fp32)": lite,
        "inst fp16": fp16_lanes,
        "inst fp64": fp64,
        "inst tensor": tensor,
        "inst xu (mufu)": xu,
        "inst lsu shared": shared_inst,
        "inst lsu global": global_inst,
        "inst lsu other": max(lsu - shared_inst - global_inst, 0.0),
        "inst tex": tex,
        "inst other": max(total - alu - fma - fp64 - tensor - xu - lsu - tex - (fp16 if separate_fp16 else 0.0), 0.0),
        "L1 sector": counter(row, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
        + counter(row, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")
        + counter(row, "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum")
        + counter(row, "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"),
        "L1 tex sector": counter(row, "l1tex__t_sectors_pipe_tex_mem_texture.sum"),
        "L2 read sector": counter(row, "lts__t_sectors_srcunit_tex_op_read.sum"),
        "L2 write sector": counter(row, "lts__t_sectors_srcunit_tex_op_write.sum"),
        "L2 read request": counter(row, "lts__t_requests_srcunit_tex_op_read.sum"),
        "L2 write request": counter(row, "lts__t_requests_srcunit_tex_op_write.sum"),
        "L2 read miss request": counter(row, "lts__t_requests_srcunit_tex_op_read_lookup_miss.sum") + atomic_misses,
        "L2 partial write miss request": max(counter(row, "lts__t_requests_srcunit_tex_op_write_lookup_miss.sum")
                                             + atomic_misses - counter(row, "dram__sectors_write.sum") / 4, 0.0),
        "DRAM read sector": counter(row, "dram__sectors_read.sum"),
        "DRAM write sector": counter(row, "dram__sectors_write.sum"),
        "L2 atom sector": counter(row, "lts__t_sectors_srcunit_tex_op_atom.sum"),
        "L2 red sector": counter(row, "lts__t_sectors_srcunit_tex_op_red.sum"),
        "shared ld wavefront": max(counter(row, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum") - ld_conflicts, 0.0),
        "shared st wavefront": max(counter(row, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum") - st_conflicts, 0.0),
        "shared ld conflict": ld_conflicts,
        "shared st conflict": st_conflicts,
        "shared atom wavefront": counter(row, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum"),
        "const request": counter(row, "idc__requests.sum"),
        "const L1 miss": counter(row, "idc__requests_lookup_miss_tag_miss.sum"),
        # The time that steady traffic keeps DRAM awake (add_busy(), which needs the duration: 0 until then).
        "DRAM busy ns": 0.0,
    }


# Steady DRAM traffic costs a power of its own while it lasts. The dependent-load probes draw, above the sleep baseline, a
# cost per byte plus a fixed power that grows with the DRAM rate and saturates: on an RTX 3090 (LowRate and Instr, 32 rows
# from 1.3 to 900 GB/s on 1 to 82 SMs) 172 pJ/B + 13.4 W x (1 - exp(-rate / 95 GB/s)), rms 2.4 W; a fixed power that does
# not depend on the rate fits worse (rms 3.8 W), and so does one per working SM (2.9 W). That is the shape of DRAM power-
# down: with requests at random times the DRAM rests whenever a gap outlasts its idle timer, and such gaps get
# exponentially rarer as the rate rises. So a byte at 113 GB/s costs 45 % more than at full bandwidth. The plain probes,
# whose lower duties are bursts at the full rate with idle time between, show no fixed power, because DRAM rests between
# the bursts; their energies per access include it at the full rate. L2 shows no fixed power (49-51 pJ/B, the fixed term
# within its error), so L2 has no such event.
# The event is the nanoseconds DRAM is kept awake: a probe's time per access at duty 100 (from the power run) or a
# kernel's time, times 1 - exp(-rate / BUSY_SCALE_GBS) of its DRAM rate (sectors read and written per ns). The fit prices
# it; only the shape comes from the sweep (tools/fit_busy.py measures it; --busy-scale sets it for another GPU). A kernel is taken at its average rate over its whole time, so one with a DRAM
# phase and a compute phase is counted as if its traffic were spread evenly. BUSY (--no-busy: off) adds it.
BUSY = True
BUSY_EVENT = "DRAM busy ns"
BUSY_SCALE_GBS = 95.0


def add_busy(counts, ns, busy=None, scale_gbs=None):
    """Set the DRAM busy event of `counts` (events of one access or one launch) for a duration of `ns` nanoseconds.
    `busy` and `scale_gbs`: None = the module's BUSY and BUSY_SCALE_GBS."""
    busy = BUSY if busy is None else busy
    scale_gbs = BUSY_SCALE_GBS if scale_gbs is None else scale_gbs
    counts[BUSY_EVENT] = 0.0
    if busy and ns > 0:
        rate = (counts["DRAM read sector"] + counts["DRAM write sector"]) * 32 / ns
        counts[BUSY_EVENT] = ns * -math.expm1(-rate / scale_gbs)
    return counts


EVENTS = list(events({}).keys())
