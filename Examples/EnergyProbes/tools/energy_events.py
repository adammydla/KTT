"""The events of the energy model, computed from CUPTI counters (standard library only).

An event is something the model prices: a warp instruction of a pipe, a sector served by a cache level or by DRAM, a
shared-memory wavefront, a nanosecond of DRAM kept awake. events() turns the counters of one row (a probe, or one launch
of a kernel) into event counts; add_busy() adds the one event that needs the duration. energy_model.py fits an energy to
every event, energy_predictor.py prices a kernel with a model file. Both must count events the same way, so the
definitions live here and nowhere else (EnergyModel.cpp, the C++ predictor, repeats them; tests/ compares the two).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import counter_table  # noqa: E402

number = counter_table.number
value_of = counter_table.value_of
read_rows = counter_table.read_rows

# The lsu pipe's instructions of shared memory and of the global path (smsp__inst_executed_op_<op>.sum).
SHARED_INSTRUCTIONS = counter_table.feeding("inst lsu shared")
GLOBAL_INSTRUCTIONS = counter_table.feeding("inst lsu global")


def counter(row, name):
    """A counter's value; 0 if the row has none. Negative values count as 0: CUPTI reports the two L2 reduction counters
    negative now and then in kernels without reductions, which would make an event negative."""
    value = value_of(row, name)
    return 0.0 if math.isnan(value) or value < 0 else value


def events(row):
    """Event counts of one row of counters. Counters a run did not collect count as 0, so the kernels to predict must be
    profiled with the same counters as the probes."""
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

    # The alu, fma and fp16 pipes count in active lanes / 32: a lane that is inactive or whose guard predicate is false
    # does no work (the guarded moves of a dynamically indexed register array cost almost nothing when the guard is false
    # in every lane). For fully active warps this is the warp instruction count; runs without the thread counters use it.
    def lanes(name, warp_count):
        value = value_of(row, name)
        return value / 32 if not math.isnan(value) else warp_count
    alu_lanes = lanes("sm__thread_inst_executed_pipe_alu_pred_on.sum", alu)
    fma_lanes = lanes("sm__thread_inst_executed_pipe_fma_pred_on.sum", fma)
    fp16_lanes = lanes("sm__thread_inst_executed_pipe_fp16_pred_on.sum" if separate_fp16
                       else "sm__thread_inst_executed_pipe_fma_type_fp16_pred_on.sum", fp16)
    # A bank conflict replays a shared wavefront for the lanes whose bank another lane took. A replayed wavefront moves
    # fewer bytes than one of a conflict-free access, so conflicts are events of their own.
    ld_conflicts = counter(row, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
    st_conflicts = counter(row, "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum")
    # An atomic that misses L2 reads its line from DRAM and later writes it back: one read and one write miss request.
    atomic_misses = counter(row, "lts__t_requests_srcunit_tex_op_atom_lookup_miss.sum") \
        + counter(row, "lts__t_requests_srcunit_tex_op_red_lookup_miss.sum")
    # The lsu pipe's instructions by memory space. With one lsu event the fit traded the energy of a shared instruction
    # for that of its wavefronts and priced the global instructions too low. Runs without these counters count all lsu
    # instructions as "inst lsu other".
    shared_inst = sum(counter(row, name) for name in SHARED_INSTRUCTIONS)
    global_inst = sum(counter(row, name) for name in GLOBAL_INSTRUCTIONS)
    return {
        "inst alu": alu_lanes,
        "inst fma (fp32, imad)": fma_lanes if separate_fp16 else max(fma_lanes - fp16_lanes, 0.0),
        "inst fp16": fp16_lanes,
        "inst fp64": fp64,
        "inst tensor": tensor,
        "inst xu (mufu)": xu,
        "inst lsu shared": shared_inst,
        "inst lsu global": global_inst,
        "inst lsu other": max(lsu - shared_inst - global_inst, 0.0),
        "inst tex": tex,  # no probe prices it (KTT has no texture arguments): a kernel that counts it gets a warning
        # Branches and the instructions of the control, address and uniform units.
        "inst other": max(total - alu - fma - fp64 - tensor - xu - lsu - tex - (fp16 if separate_fp16 else 0.0), 0.0),
        "L1 sector": counter(row, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
        + counter(row, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")
        + counter(row, "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum")
        + counter(row, "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum"),
        "L1 tex sector": counter(row, "l1tex__t_sectors_pipe_tex_mem_texture.sum"),
        "L2 read sector": counter(row, "lts__t_sectors_srcunit_tex_op_read.sum"),
        "L2 write sector": counter(row, "lts__t_sectors_srcunit_tex_op_write.sum"),
        # An L2 request (one per line an instruction touches) carries what a transfer between L1 and L2 costs beyond its
        # sectors: coalesced 16-byte accesses send 4 sectors per request, scattered sectors one. A read miss request
        # does the same for DRAM reads.
        "L2 read request": counter(row, "lts__t_requests_srcunit_tex_op_read.sum"),
        "L2 write request": counter(row, "lts__t_requests_srcunit_tex_op_write.sum"),
        "L2 read miss request": counter(row, "lts__t_requests_srcunit_tex_op_read_lookup_miss.sum") + atomic_misses,
        # Writes to DRAM are priced by their sectors: L2 writes back every dirty line, whether the store that dirtied
        # it missed L2 or hit a line a load had brought in. Write miss requests beyond one per four written-back
        # sectors are partial writes (1 to 3 sectors per request), which cost a request of their own.
        "L2 partial write miss request": max(counter(row, "lts__t_requests_srcunit_tex_op_write_lookup_miss.sum")
                                             + atomic_misses - counter(row, "dram__sectors_write.sum") / 4, 0.0),
        "DRAM read sector": counter(row, "dram__sectors_read.sum"),
        "DRAM write sector": counter(row, "dram__sectors_write.sum"),
        # Global atomics are L2 operations of their own (neither read nor write sectors).
        "L2 atom sector": counter(row, "lts__t_sectors_srcunit_tex_op_atom.sum"),
        "L2 red sector": counter(row, "lts__t_sectors_srcunit_tex_op_red.sum"),
        "shared ld wavefront": max(counter(row, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum") - ld_conflicts, 0.0),
        "shared st wavefront": max(counter(row, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum") - st_conflicts, 0.0),
        "shared ld conflict": ld_conflicts,
        "shared st conflict": st_conflicts,
        "shared atom wavefront": counter(row, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum"),
        # A constant load (LDC) is one request to the constant cache; a miss there is served by the constant L1.5.
        "const request": counter(row, "idc__requests.sum"),
        "const L1 miss": counter(row, "idc__requests_lookup_miss_tag_miss.sum"),
        "DRAM busy ns": 0.0,  # add_busy(), which needs the duration
    }


# Steady DRAM traffic costs a power of its own while it lasts. The dependent-load probes draw, above the sleep baseline, a
# cost per byte plus a fixed power that grows with the DRAM rate and saturates: on an RTX 3090 (rows from 1.3 to 900
# GB/s) 172 pJ/B + 13.4 W x (1 - exp(-rate / 95 GB/s)). That is the shape of DRAM power-down: the DRAM rests whenever a
# gap between requests outlasts its idle timer, and such gaps get exponentially rarer as the rate rises. The event is
# the nanoseconds DRAM is kept awake: a probe's time per access at duty 100, or a kernel's time, times
# 1 - exp(-rate / BUSY_SCALE_GBS) of its DRAM rate. The fit prices it; only the shape's scale is a setting (a model file
# carries its own). A kernel is taken at its average rate over its whole time.
BUSY_EVENT = "DRAM busy ns"
BUSY_SCALE_GBS = 95.0


def add_busy(counts, ns, scale_gbs=None):
    """Set the DRAM busy event of `counts` (events of one access or one launch) for a duration of `ns` nanoseconds."""
    counts[BUSY_EVENT] = 0.0
    if ns > 0:
        rate = (counts["DRAM read sector"] + counts["DRAM write sector"]) * 32 / ns
        counts[BUSY_EVENT] = ns * -math.expm1(-rate / (scale_gbs or BUSY_SCALE_GBS))
    return counts


EVENTS = list(events({}).keys())
