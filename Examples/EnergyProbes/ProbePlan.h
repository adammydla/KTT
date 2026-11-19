// What a probe run measures: the groups of probes, each a KTT kernel whose tuning parameters are the probe's variants.
// The run, the dry run and the number of configurations all come from this one table.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "Hardware.h"
#include "Options.h"

// The probe kernels (kernels/*.cu).
enum class ProbeKernel
{
    Global,   // probeGlobal: loads, stores, atomics and the no-load loop at a footprint
    Compute,  // probeCompute: arithmetic and other instruction classes
    Shared,   // probeShared: conflict-free shared loads, stores and atomics
    Banks,    // probeSharedBanks: shared loads and stores with bank conflicts
    Constant, // probeConstant: constant-memory loads
    Scatter,  // probeScatter: streams and scattered runs of sectors (request size)
    Partial,  // probePartial: every thread walks a sector of its own (partial sectors)
    Write,    // probeWrite: loads and stores together, local memory
    Latency   // probeLatency: chains of dependent loads
};

struct ProbeGroup
{
    std::string name;   // for --only, the results file and the KTT kernel
    std::string probe;  // Row::probe
    ProbeKernel kernel;
    std::vector<std::pair<std::string, std::vector<uint64_t>>> parameters; // the variants, besides DUTY
    std::function<bool(const std::vector<uint64_t>&)> allowed; // over the values in the order of `parameters`; empty: all
    std::vector<uint64_t> duties; // all duties of the run (energy = slope of the fit), or 100 only (energy of that row)
    bool placement;               // the kernel claims SM slots, which the launcher checks
    std::string description;
    bool optional = false;        // runs only when --only names it (not part of the calibration run)
};

// probeScatter and probePartial (the locality, sub-sector and write-path groups) number the sectors of the whole grid, so
// the data one SM walks at the L2 footprint depends on the block count. A thread's counter visits one residue class of
// sectors (modulo the gcd of the grid's step and the footprint), and the blocks of an SM (placed round-robin: block b on
// SM b mod SMs, as the RTX 3090's counters imply) cover the classes they start in: half of the footprint with 82 SMs, a
// quarter with 68, but footprint / SMs with 64, 80 or 144, where it fits in L1. Bytes of `footprint` one SM walks:
uint64_t GridWalkPerSm(uint64_t footprint, uint64_t multiprocessors, uint64_t blocksPerSm, uint64_t sectorsPerBlock);

// Whether those groups' L2 rows measure L2 on this GPU: every SM walks at least 4x the L1 in both kernels. Where they do
// not, the L2 rows are left out of the plan (a warning says so). A step of an odd number of blocks, which makes every SM
// walk the whole footprint, was tried and measured on 2026-11-16: the blocks of an SM then share one cycle and hit each
// other's lines in L1 (70 % of the stream's sectors on an RTX 3090), so it is not used.
bool GridWalkReachesL2(const Options& options, const Hardware& hardware, const Footprints& footprints);

// The groups of the run in the order they run, restricted to --only.
std::vector<ProbeGroup> ProbeGroups(const Options& options, const Hardware& hardware, const Footprints& footprints);

// The names --only accepts.
const std::vector<std::string>& ProbeGroupNames();

// Configurations of a group: the allowed combinations of its parameters times its duties.
size_t Configurations(const ProbeGroup& group);

// Prints the GPU, the footprints, warnings about them and the plan of the run.
void DescribeSetup(const Hardware& hardware, const Options& options, const Footprints& footprints,
    const std::vector<ProbeGroup>& groups);
