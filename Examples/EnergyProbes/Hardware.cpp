#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <json.hpp>

#include "Hardware.h"

namespace
{

std::optional<double> JsonNumber(const nlohmann::json& root, const std::vector<std::string>& path)
{
    const nlohmann::json* node = &root;

    for (const auto& key : path)
    {
        if (!node->is_object() || !node->contains(key))
        {
            return std::nullopt;
        }

        node = &node->at(key);
    }

    if (!node->is_number())
    {
        return std::nullopt;
    }

    return node->get<double>();
}

// MT4G marks unreliable measurements with a low confidence; it uses 0.5 as its own validity threshold.
std::optional<uint64_t> ConfidentSize(const nlohmann::json& root, const std::vector<std::string>& path)
{
    std::vector<std::string> confidencePath = path;
    confidencePath.back() = "confidence";
    const auto confidence = JsonNumber(root, confidencePath);
    const auto size = JsonNumber(root, path);

    if (!size.has_value() || (confidence.has_value() && confidence.value() <= 0.5))
    {
        return std::nullopt;
    }

    return static_cast<uint64_t>(size.value());
}

// MT4G measures cache sizes with a pointer chase, which can come out a little low: on an RTX 2080 Ti it reported 64768 B
// (63.25 KiB) for a 64 KiB L1. The L1 footprint was then above half of it, and every L1 row was labelled "mixed". A
// measured size at most 3 % below a power of two or a multiple of 8 KiB is taken as that size. (The RTX 3090's 108288 B
// is 5.6 % below 112 KiB and stays as measured.)
std::optional<uint64_t> RoundMeasuredSize(const std::optional<uint64_t> size, const std::string& what)
{
    if (!size.has_value() || size.value() == 0)
    {
        return size;
    }

    const uint64_t measured = size.value();
    std::vector<uint64_t> candidates = {CeilPowerOfTwo(measured)};

    if (measured >= 8 * KiB)
    {
        candidates.push_back((measured + 8 * KiB - 1) / (8 * KiB) * (8 * KiB));
    }

    uint64_t rounded = measured;

    for (const uint64_t candidate : candidates)
    {
        if (candidate > measured && (candidate - measured) * 100 <= 3 * candidate
            && (rounded == measured || candidate < rounded))
        {
            rounded = candidate;
        }
    }

    if (rounded != measured)
    {
        std::cout << "MT4G: " << what << " measured as " << measured << " B, taken as " << FormatBytes(rounded)
            << " (at most 3 % below; --l1-kib overrides the L1)\n";
    }

    return rounded;
}

void ApplyMt4gResults(Hardware& hardware, const std::string& file)
{
    std::ifstream input(file);

    if (!input)
    {
        throw std::runtime_error("Unable to open MT4G result file " + file);
    }

    nlohmann::json root;
    input >> root;

    if (root.contains("general") && root["general"].contains("name") && root["general"]["name"].is_string())
    {
        const std::string mt4gName = root["general"]["name"].get<std::string>();

        if (hardware.name.empty())
        {
            hardware.name = mt4gName;
        }
        else if (mt4gName != hardware.name)
        {
            std::cout << "WARNING: MT4G file describes '" << mt4gName << "' but the selected device is '" << hardware.name
                << "'\n";
        }
    }

    // Device properties normally come from KTT; they are only filled from MT4G for --dry-run.
    if (hardware.multiprocessors == 0)
    {
        hardware.multiprocessors = static_cast<uint32_t>(JsonNumber(root, {"compute", "multiProcessorCount"}).value_or(0));
    }

    if (hardware.globalMemory == 0)
    {
        hardware.globalMemory = static_cast<uint64_t>(JsonNumber(root, {"memory", "main", "totalGlobalMem", "value"})
            .value_or(0));
    }

    if (hardware.sharedPerBlock == 0)
    {
        hardware.sharedPerBlock = static_cast<uint64_t>(JsonNumber(root, {"memory", "shared", "sharedMemPerBlock", "value"})
            .value_or(48 * KiB));
    }

    if (hardware.computeCapabilityMajor == 0)
    {
        hardware.computeCapabilityMajor = static_cast<uint32_t>(JsonNumber(root, {"general", "computeCapability", "major"})
            .value_or(0));
        hardware.computeCapabilityMinor = static_cast<uint32_t>(JsonNumber(root, {"general", "computeCapability", "minor"})
            .value_or(0));
    }

    if (const auto l1 = ConfidentSize(root, {"memory", "l1", "size", "size"}))
    {
        hardware.l1Size = RoundMeasuredSize(l1, "L1");
    }

    if (const auto l2 = JsonNumber(root, {"memory", "l2", "size", "value"}))
    {
        hardware.l2Size = static_cast<uint64_t>(l2.value());
    }

    if (const auto segment = ConfidentSize(root, {"memory", "l2", "segmentSize", "size"}))
    {
        hardware.l2SegmentSize = RoundMeasuredSize(segment, "L2 segment");
    }

    if (const auto threads = JsonNumber(root, {"compute", "maxThreadsPerMultiProcessor"}))
    {
        hardware.maxThreadsPerMultiprocessor = static_cast<uint64_t>(threads.value());
    }

    if (const auto sharedPerSm = JsonNumber(root, {"memory", "shared", "sharedMemPerMultiProcessor", "value"}))
    {
        hardware.sharedPerMultiprocessor = static_cast<uint64_t>(sharedPerSm.value());
    }

    if (const auto constantL1 = ConfidentSize(root, {"memory", "constant", "l1", "size", "size"}))
    {
        hardware.constantL1Size = RoundMeasuredSize(constantL1, "constant L1");
    }

    // MT4G reports 64 KiB + 1 with confidence 0 when the L1.5 holds all of constant memory.
    if (const auto constantL15 = ConfidentSize(root, {"memory", "constant", "l1.5", "size", "size"}))
    {
        hardware.constantL15Size = RoundMeasuredSize(constantL15, "constant L1.5");
    }
}

// Footprints from this size on are expected to be served by DRAM.
uint64_t DramLimit(const Hardware& hardware, const Footprints& footprints)
{
    return hardware.l2Size.has_value() ? 4 * hardware.l2Size.value() : footprints.dram;
}

} // namespace

void ApplyOverrides(Hardware& hardware, const Options& options)
{
    if (!options.mt4gFile.empty())
    {
        ApplyMt4gResults(hardware, options.mt4gFile);
    }

    if (options.l1Override.has_value())
    {
        hardware.l1Size = options.l1Override;
    }

    if (options.l2Override.has_value())
    {
        hardware.l2Size = options.l2Override;
        hardware.l2SegmentSize.reset();
    }
}

Footprints ChooseFootprints(const Hardware& hardware, const Options& options)
{
    Footprints result;

    // The blocks of one SM read disjoint stripes of the footprint (see probeGlobal). A stripe needs at least two rows of
    // one 16-byte load per thread, so that consecutive loads of a thread return different data.
    const uint64_t row = DefaultWidth * options.blockSize;
    result.minimum = 2 * row * CeilPowerOfTwo(options.blocksPerSm);

    // L1: a quarter of the cache keeps the working set resident even with a smaller L1/shared carve-out.
    result.l1 = hardware.l1Size.has_value() ? FloorPowerOfTwo(hardware.l1Size.value() / 4) : 16 * KiB;
    result.l1 = std::max(result.l1, result.minimum);

    // L2: half of the L2 (or of one L2 partition, as reported by MT4G's segment size) to stay in the near partition.
    if (hardware.l2Size.has_value())
    {
        uint64_t usable = hardware.l2Size.value();

        if (hardware.l2SegmentSize.has_value())
        {
            usable = std::min(usable, hardware.l2SegmentSize.value());
        }

        result.l2 = FloorPowerOfTwo(usable / 2);
    }
    else
    {
        result.l2 = 2 * MiB;
    }

    result.l2 = std::max(result.l2, 2 * result.l1);

    // DRAM: far larger than L2 so that almost every access misses.
    result.dram = hardware.l2Size.has_value() ? CeilPowerOfTwo(8 * hardware.l2Size.value()) : GiB;
    result.dram = std::max(result.dram, 256 * MiB);
    result.dram = std::min({result.dram, 2 * GiB, FloorPowerOfTwo(hardware.globalMemory / 4)});

    // DRAM footprints are split over all blocks, so that blocks on different SMs cannot convoy in L2. Only `blocks` of the
    // stripes are read, and that part must still be a DRAM footprint. A stripe needs two rows, and the data of one SM must
    // still be twice its L1.
    const uint64_t blocks = std::max<uint64_t>(static_cast<uint64_t>(hardware.multiprocessors) * options.blocksPerSm, 1);
    const uint64_t l1 = hardware.l1Size.value_or(4 * result.l1);
    result.gridStripes = CeilPowerOfTwo(blocks);
    result.disjointFrom = CeilPowerOfTwo(std::max({DramLimit(hardware, result) * result.gridStripes / blocks,
        2 * row * result.gridStripes, 2 * l1 * result.gridStripes / options.blocksPerSm}));

    // Constant memory: half of the constant L1, and half of the L1.5 (which also holds instructions) or of all constant
    // memory if the L1.5 holds it all. Every warp of an SM walks a stripe of its own, of at least two words.
    const uint64_t warpSlots = CeilPowerOfTwo(options.blocksPerSm * std::max<uint64_t>(options.blockSize / 32, 1));
    result.constantL1 = std::max(hardware.constantL1Size.has_value() ? FloorPowerOfTwo(hardware.constantL1Size.value() / 2)
        : 1 * KiB, 8 * warpSlots);
    result.constantL15 = std::max(FloorPowerOfTwo(std::min(hardware.constantL15Size.value_or(ConstantWords * 4),
        ConstantWords * 4) / 2), 2 * result.constantL1);
    return result;
}

uint64_t ResidentBlocks(const uint64_t registers, const uint64_t blockSize, const Hardware& hardware)
{
    if (registers == 0 || blockSize == 0)
    {
        return 0;
    }

    const uint64_t partitions = 4;
    const uint64_t registersPerPartition = 65536 / partitions;
    const uint64_t perWarp = (registers * 32 + 255) / 256 * 256;
    const uint64_t warpsPerBlock = (blockSize + 31) / 32;
    const uint64_t byRegisters = registersPerPartition / perWarp * partitions / warpsPerBlock;
    const uint64_t byThreads = hardware.maxThreadsPerMultiprocessor.value_or(2048) / blockSize;
    return std::min(byRegisters, byThreads);
}

std::string ExpectedLevel(const uint64_t operation, const uint64_t footprint, const Hardware& hardware,
    const Footprints& footprints)
{
    if (operation == LoadNone)
    {
        return "none";
    }

    const uint64_t l1Limit = hardware.l1Size.value_or(4 * footprints.l1) / 2;
    const uint64_t l2Limit = hardware.l2Size.has_value() ? hardware.l2Size.value() / 2 : footprints.l2;
    const uint64_t dramLimit = DramLimit(hardware, footprints);

    // L1 is write-through for global stores: they always go to L2 (and from there to DRAM if the footprint is large).
    // Global atomics are performed in L2.
    if (operation == StoreDefault || operation == AtomicRed || operation == AtomicAdd)
    {
        return footprint <= l2Limit ? "L2" : (footprint >= dramLimit ? "DRAM" : "mixed");
    }

    if (operation != LoadCg && footprint <= l1Limit)
    {
        return "L1";
    }

    if (footprint <= l2Limit && (operation == LoadCg || footprint >= 4 * hardware.l1Size.value_or(footprints.l1)))
    {
        return "L2";
    }

    if (footprint >= dramLimit)
    {
        return "DRAM";
    }

    return "mixed";
}

std::string ExpectedConstantLevel(const uint64_t footprint, const Hardware& hardware, const Footprints& footprints)
{
    const uint64_t l1 = hardware.constantL1Size.value_or(2 * footprints.constantL1);
    return footprint <= l1 / 2 ? "const-L1" : (footprint >= 8 * l1 ? "const-L1.5" : "const-mixed");
}
