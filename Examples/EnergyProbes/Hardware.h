// The GPU as the probes see it: sizes from KTT and from an MT4G result file, the footprint of every memory
// level, and the level a probe is expected to hit.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include "Common.h"
#include "Options.h"

struct Hardware
{
    std::string name;
    uint32_t multiprocessors = 0;
    uint64_t globalMemory = 0;
    uint64_t sharedPerBlock = 0;
    std::optional<uint64_t> l1Size;
    std::optional<uint64_t> l2Size;
    std::optional<uint64_t> l2SegmentSize;
    std::optional<uint64_t> maxThreadsPerMultiprocessor;
    std::optional<uint64_t> sharedPerMultiprocessor;
    std::optional<uint64_t> constantL1Size; // MT4G's constant L1 (per SM)
    std::optional<uint64_t> constantL15Size;
    uint32_t computeCapabilityMajor = 0;
    uint32_t computeCapabilityMinor = 0;
};

struct Footprints
{
    uint64_t minimum = 0;
    uint64_t l1 = 0;
    uint64_t l2 = 0;
    uint64_t dram = 0;
    uint64_t gridStripes = 1;  // power of two >= number of blocks
    uint64_t disjointFrom = 0; // footprints of at least this size are split over all blocks of the grid
    uint64_t constantL1 = 0;   // constant memory read by the constant probe at each level
    uint64_t constantL15 = 0;
};

void ApplyOverrides(Hardware& hardware, const Options& options);

Footprints ChooseFootprints(const Hardware& hardware, const Options& options);

// Blocks of `blockSize` threads that fit on an SM at once with `registers` per thread, by the register file and the
// thread limit only (as cuda_occupancy.h for sm_70 and newer: 64 K registers per SM in 4 partitions, allocated per warp in
// units of 256). 0 if the registers are unknown. On sm_75 the 16-byte write probes take 80-82 registers: 3 or 2 of 4
// blocks of 256 threads fit, and the blocks of an SM then share stripes.
uint64_t ResidentBlocks(const uint64_t registers, const uint64_t blockSize, const Hardware& hardware);

std::string ExpectedLevel(const uint64_t operation, const uint64_t footprint, const Hardware& hardware,
    const Footprints& footprints);

// Level of the constant probe: the constant L1 for footprints up to half its size, the L1.5 from eight times its size.
std::string ExpectedConstantLevel(const uint64_t footprint, const Hardware& hardware, const Footprints& footprints);
