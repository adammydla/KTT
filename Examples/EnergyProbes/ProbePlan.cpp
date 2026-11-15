#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <tuple>

#include "ProbePlan.h"

const std::vector<std::string>& ProbeGroupNames()
{
    static const std::vector<std::string> names = {"global", "compute", "shared", "latency", "banks", "constant",
        "locality", "sub-sector", "write-path", "misaligned", "writes"};
    return names;
}

// The probe set is what the model's scores need, and no more (README, "Which probes run"): of the bank-conflict,
// latency, locality and write-path grids only the ends, since the model predicts the points between them as well
// as it fits them. The dependent loads run right after the shared probes, before the groups that load DRAM hardest:
// after them, their 16-chain DRAM row hit a thermal limit of the memory on an RTX 3090.
std::vector<ProbeGroup> ProbeGroups(const Options& options, const Hardware& hardware, const Footprints& footprints)
{
    const uint64_t l1 = footprints.l1;
    const uint64_t l2 = footprints.l2;
    const uint64_t dram = footprints.dram;
    const uint64_t localDram = WriteLocalFrameBytes * options.blockSize * options.blocksPerSm;
    const std::vector<uint64_t> fullRate = {100};
    const std::pair<std::string, std::vector<uint64_t>> stripes = {"GRID_STRIPES", {footprints.gridStripes}};
    const std::pair<std::string, std::vector<uint64_t>> shared = {"SMEM_BYTES", {options.sharedBytes}};
    using Values = const std::vector<uint64_t>&;

    // Plain loads at the three levels and the no-load loop for every width; stores and atomics start at L2 (L1 is
    // write-through, atomics are performed in L2).
    std::set<std::tuple<uint64_t, uint64_t, uint64_t>> global;
    std::set<std::tuple<uint64_t, uint64_t, uint64_t>> streams;

    for (const uint64_t width : {4, 8, 16})
    {
        global.insert({{LoadDefault, l1, width}, {LoadDefault, l2, width}, {LoadDefault, dram, width},
            {LoadNone, l1, width}});
    }

    for (const uint64_t footprint : {l2, dram})
    {
        global.insert({{StoreDefault, footprint, 4}, {StoreDefault, footprint, 16}, {AtomicRed, footprint, AtomicWidth},
            {AtomicAdd, footprint, AtomicWidth}});
    }

    for (const uint64_t width : {4, 16})
    {
        streams.insert({{LoadDefault, l1, width}, {LoadDefault, l2, width}, {LoadDefault, dram, width},
            {StoreDefault, l2, width}, {StoreDefault, dram, width}});
    }

    const auto in = [](const std::set<std::tuple<uint64_t, uint64_t, uint64_t>>& plan)
    {
        return [plan](Values v) { return plan.count({v[0], v[1], v[2]}) > 0; };
    };

    std::vector<uint64_t> compute = {ComputeLoop, ComputeFfma, ComputeDfma, ComputeImad, ComputeHfma2, ComputeMufu,
        ComputeFfma3, ComputeShfl, ComputeBar, ComputeFfmaLoop, ComputeDynamicIndex};

    // HMMA in the shape the GPU has (kernels/Compute.cu): m8n8k4 on sm_70, m16n8k8 on sm_75, m16n8k16 from sm_80.
    if (hardware.computeCapabilityMajor >= 7 || hardware.computeCapabilityMajor == 0)
    {
        compute.push_back(ComputeHmma);
    }

    const std::vector<ProbeGroup> all = {
        {"global", "global", ProbeKernel::Global,
            {{"LOAD_OP", {LoadDefault, LoadNone, StoreDefault, AtomicRed, AtomicAdd}}, {"FOOTPRINT", {l1, l2, dram}},
                {"WIDTH", {4, 8, 16}}, stripes},
            in(global), options.duties, true,
            "ld of 4, 8, 16 B at L1, L2, DRAM; the no-load loop; st of 4, 16 B and 32-bit red, atom at L2, DRAM"},
        {"compute", "compute", ProbeKernel::Compute, {{"COMPUTE_OP", compute}}, {}, options.duties, false,
            "the empty loop, FFMA (x * x + c and three registers), DFMA, IMAD, HFMA2, HMMA, MUFU, shuffle, barrier, the "
            "FFMA loop with branches, FFMA accumulators with a runtime index"},
        {"shared", "shared", ProbeKernel::Shared, {shared, {"SHARED_OP", {SharedLoad, SharedStore, SharedRed}}}, {},
            options.duties, false, "conflict-free ld.shared, st.shared (16 B) and red.shared"},
        {"latency", "latency", ProbeKernel::Latency,
            {{"FOOTPRINT", {l2, dram}}, {"WIDTH", {4}}, {"CHAINS", {1, 16}}, {"ACTIVE_WARPS", {1, 8}}}, {}, fullRate, false,
            "chains of dependent 4-byte loads at L2, DRAM: 1 or 16 chains per warp, 1 or 8 working warps per block"},
        {"banks", "banks", ProbeKernel::Banks,
            {shared, {"SHARED_OP", {SharedLoad, SharedStore}}, {"SHARED_WIDTH", {4, 16}}, {"BANK_WAYS", {1, 8}}},
            [](Values v) { return v[2] * v[3] <= WavefrontBytes; }, options.duties, false,
            "ld.shared, st.shared of 4, 16 B with 1 and 8 times the wavefronts of the conflict-free access"},
        {"constant", "constant", ProbeKernel::Constant, {{"CONST_BYTES", {footprints.constantL1, footprints.constantL15}}},
            {}, options.duties, true, "ld.const (one word per warp access) at the constant L1 and L1.5"},
        {"locality", "locality", ProbeKernel::Scatter,
            {{"LOAD_OP", {LoadDefault, StoreDefault}}, {"FOOTPRINT", {l2, dram}}, {"WIDTH", {DefaultWidth}},
                {"RUN_BYTES", {0, 32, 128}}},
            [dram](Values v) { return v[1] == dram || v[3] <= SectorBytes; }, options.duties, false,
            "16-byte ld, st in one stream or in runs of 32 B (one sector per L2 request) and 128 B placed apart, at DRAM; "
            "stream and 32 B at L2"},
        {"sub-sector", "sub-sector", ProbeKernel::Partial,
            {{"LOAD_OP", {LoadDefault, StoreDefault}}, {"FOOTPRINT", {l2, dram}}, {"WIDTH", {4, 8}}, {"SPREAD", {0, 1}}},
            [dram](Values v) { return v[1] == dram || v[3] != 0; }, options.duties, false,
            "ld, st of 4, 8 B, every thread walking a sector of its own: the sectors of a warp adjacent or scattered at DRAM, "
            "scattered at L2"},
        {"write-path", "write path", ProbeKernel::Partial,
            {{"LOAD_OP", {LoadDefault, StoreDefault}}, {"FOOTPRINT", {l2, dram}}, {"WIDTH", {4}}, {"SPREAD", {0, 2, 1}},
                {"PARTIAL_WORDS", {1, 8}}},
            {}, fullRate, false, "4-byte ld, st of 1 or 8 words of each sector, with 4, 2 or 1 sectors per L2 request, at L2, DRAM"},
        {"misaligned", "misaligned", ProbeKernel::Global,
            {{"LOAD_OP", {LoadDefault, StoreDefault}}, {"FOOTPRINT", {l1, l2, dram}}, {"WIDTH", {4, 16}}, stripes,
                {"OFFSET_BYTES", {16, 32, 64, 96}}},
            in(streams), fullRate, true,
            "ld of 4, 16 B at L1, L2, DRAM and st at L2, DRAM, started 16, 32, 64, 96 B off the 128-byte line"},
        {"writes", "write", ProbeKernel::Write,
            {{"WRITE_OP", {WriteCopy, WriteUpdate, WriteLocal}}, {"FOOTPRINT", {l1, l2, dram, localDram}}, {"WIDTH", {4, 16}},
                stripes},
            [=](Values v) { return v[0] == WriteLocal ? v[1] == l1 || v[1] == localDram : v[1] == l2 || v[1] == dram; },
            options.duties, true,
            "loads and stores together, 4 and 16 B: copy and update in place at L2, DRAM; local memory at L1 and with the "
            "whole frame of every thread"},
    };

    std::vector<ProbeGroup> groups;
    std::copy_if(all.cbegin(), all.cend(), std::back_inserter(groups), [&options](const ProbeGroup& group)
    {
        return options.only.empty() || std::find(options.only.cbegin(), options.only.cend(), group.name) != options.only.cend();
    });
    return groups;
}

size_t Configurations(const ProbeGroup& group)
{
    std::vector<uint64_t> values(group.parameters.size());
    size_t count = 0;

    // Every combination of the parameters' values, like an odometer.
    const std::function<void(size_t)> visit = [&](const size_t index)
    {
        if (index == group.parameters.size())
        {
            count += !group.allowed || group.allowed(values) ? 1 : 0;
            return;
        }

        for (const uint64_t value : group.parameters[index].second)
        {
            values[index] = value;
            visit(index + 1);
        }
    };

    visit(0);
    return count * group.duties.size();
}

void DescribeSetup(const Hardware& hardware, const Options& options, const Footprints& footprints,
    const std::vector<ProbeGroup>& groups)
{
    const uint64_t blocks = static_cast<uint64_t>(hardware.multiprocessors) * options.blocksPerSm;
    const auto size = [](const std::optional<uint64_t>& bytes) { return bytes ? FormatBytes(bytes.value()) : "unknown"; };
    const auto warn = [](const std::string& text) { std::cout << "WARNING: " << text << "\n"; };

    std::cout << "Device: " << hardware.name << ", " << hardware.multiprocessors << " SMs, " << blocks << " blocks x "
        << options.blockSize << " threads\n"
        << "L1 size: " << size(hardware.l1Size) << ", L2 size: " << size(hardware.l2Size)
        << (hardware.l2SegmentSize ? ", L2 segment: " + FormatBytes(hardware.l2SegmentSize.value()) : "") << "\n"
        << "Footprints: L1 " << FormatBytes(footprints.l1) << ", L2 " << FormatBytes(footprints.l2) << ", DRAM "
        << FormatBytes(footprints.dram) << "; constant L1 " << FormatBytes(footprints.constantL1) << ", L1.5 "
        << FormatBytes(footprints.constantL15) << "\n"
        << "Stripes: the " << options.blocksPerSm << " blocks of an SM access disjoint stripes; footprints from "
        << FormatBytes(footprints.disjointFrom) << " on are split over all " << blocks << " blocks (" << footprints.gridStripes
        << " stripes, " << FormatBytes(footprints.dram / footprints.gridStripes * blocks) << " of the DRAM footprint accessed)\n";

    if (footprints.dram < footprints.disjointFrom)
    {
        warn("the DRAM footprint is too small to split over all blocks, blocks on different SMs may share lines in L2.");
    }

    if (options.mt4gFile.empty() && !options.l1Override && !options.l2Override)
    {
        warn("no MT4G file given, cache sizes are guesses. Pass --mt4g <GPU>.json for meaningful footprints.");
    }

    // The L1 footprint is a quarter of the L1, but at least two rows of 16-byte loads per block stripe.
    if (hardware.l1Size.has_value() && ExpectedLevel(LoadDefault, footprints.l1, hardware, footprints) != "L1")
    {
        warn("the L1 footprint (" + FormatBytes(footprints.l1) + ") is not below half of the L1 (" + size(hardware.l1Size)
            + "): its rows are labelled " + ExpectedLevel(LoadDefault, footprints.l1, hardware, footprints)
            + ". Check --l1-kib, or lower --block-size or --blocks-per-sm (the footprint is at least "
            + FormatBytes(footprints.minimum) + ").");
    }
    else if (hardware.l1Size.has_value() && 3 * footprints.l1 > hardware.l1Size.value())
    {
        warn("the L1 footprint (" + FormatBytes(footprints.l1) + ") is more than a third of the L1 (" + size(hardware.l1Size)
            + "); with a smaller L1/shared carve-out it may not stay in L1 (the hit rates of a --profile run show it).");
    }

    if (hardware.l1Size.has_value() && footprints.l2 < 4 * hardware.l1Size.value())
    {
        warn("the L2 footprint is less than 4x the L1 size, the L2 probes may still hit L1.");
    }

    if (hardware.computeCapabilityMajor != 0 && hardware.computeCapabilityMajor < 7)
    {
        warn("before Volta, plain global loads are not cached in L1 by default and the sleep baseline cannot sleep.");
    }

    if (hardware.maxThreadsPerMultiprocessor.has_value()
        && options.blocksPerSm * options.blockSize > hardware.maxThreadsPerMultiprocessor.value())
    {
        warn("more threads per SM than can be resident, blocks will run in several waves.");
    }

    if (options.sharedBytes > std::min<uint64_t>(48 * KiB, hardware.sharedPerBlock)
        || (hardware.sharedPerMultiprocessor.has_value()
            && options.sharedBytes * options.blocksPerSm > hardware.sharedPerMultiprocessor.value()))
    {
        warn("the shared probes do not fit " + std::to_string(options.blocksPerSm) + " blocks per SM.");
    }

    // probeScatter and probePartial number the sectors of the whole grid and advance every thread by the sectors of the
    // grid per step, so at the L2 footprint the data of one SM is the footprint times the blocks per SM, divided by the
    // largest power of two in the number of blocks: half of it with 82 SMs, but 1/16 with 80 and 1/64 with 64, where it
    // fits in L1. The counters of a --profile run show it, and the model prices the events that happened.
    for (const auto& [name, sectorsPerBlock] : {std::pair<std::string, uint64_t>{"locality", options.blockSize * DefaultWidth
        / SectorBytes}, std::pair<std::string, uint64_t>{"sub-sector", options.blockSize}})
    {
        const uint64_t sectors = footprints.l2 / SectorBytes;
        const uint64_t perSm = blocks == 0 ? footprints.l2 : std::min(footprints.l2, sectors
            / std::gcd(blocks * sectorsPerBlock, sectors) * sectorsPerBlock * SectorBytes * options.blocksPerSm);

        if (perSm < 4 * hardware.l1Size.value_or(4 * footprints.l1))
        {
            warn("with " + std::to_string(blocks) + " blocks the " + name + " probes walk only " + FormatBytes(perSm)
                + " per SM at the L2 footprint (less than 4x the L1): their L2 rows will mostly hit L1 (check them with "
                "tools/check_run.py --counters).");
        }
    }

    std::cout << "Plan" << (options.randomOrder ? " (the probes of a group in random order)" : "") << ": idle power for "
        << options.idleS << " s (without a CUDA context, before the warm-up and at the end), warm-up " << options.warmupS
        << " s, a sleep baseline, then every group followed by a sleep baseline:\n";
    size_t configurations = 1 + groups.size();

    for (const ProbeGroup& group : groups)
    {
        const size_t count = Configurations(group);
        configurations += count;
        std::cout << "  " << std::left << std::setw(11) << group.name << std::right << std::setw(4) << count << " at duty "
            << std::left << std::setw(7) << JoinList(group.duties, "/") << " " << group.description << "\n" << std::right;
    }

    // A configuration whose power does not settle runs to --max-time-ms, so a queue timeout must allow the upper bound.
    const uint64_t fixedS = options.warmupS + 3 * options.idleS;
    std::cout << configurations << " configurations (" << 1 + groups.size() << " of them baselines), about "
        << (configurations * options.minTimeMs / 1000 + fixedS) / 60 << " min of measurement if every power reading settles "
        << "at once, at most " << (configurations * options.maxTimeMs / 1000 + fixedS) / 60 << " min\n";
}
