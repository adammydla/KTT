#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ProbePlan.h"

Plan BuildPlan(const Options& options, const Footprints& footprints)
{
    Plan plan;

    if (options.sweep)
    {
        for (uint64_t footprint = footprints.minimum; footprint <= footprints.dram; footprint *= 2)
        {
            plan.insert({LoadDefault, footprint, DefaultWidth});
        }
    }

    // Every width: plain loads at the three levels, and the no-load loop of the same width as its overhead reference.
    for (const uint64_t width : options.widths)
    {
        plan.insert({LoadDefault, footprints.l1, width});
        plan.insert({LoadDefault, footprints.l2, width});
        plan.insert({LoadDefault, footprints.dram, width});
        plan.insert({LoadNone, footprints.l1, width});
    }

    // ld.cg skips L1. At the L2 footprint every SM streams over many L2 slices; a tiny footprint would concentrate all
    // traffic on a few slices and measure that bottleneck instead.
    plan.insert({LoadCa, footprints.l1, DefaultWidth});
    plan.insert({LoadCg, footprints.l2, DefaultWidth});
    plan.insert({LoadCg, footprints.dram, DefaultWidth});
    // The read-only data path (MT4G's read-only cache, LDG.CONSTANT) and stores. Stores skip L1, so they start at the L2
    // footprint.
    plan.insert({LoadNc, footprints.l1, DefaultWidth});
    plan.insert({LoadNc, footprints.l2, DefaultWidth});
    plan.insert({StoreDefault, footprints.l2, DefaultWidth});
    plan.insert({StoreDefault, footprints.dram, DefaultWidth});

    // 32-bit atomics in L2 (and with a DRAM footprint, the misses they cause), with 4-byte loads, stores and the no-load
    // loop as references: the same code apart from the memory instruction (red vs st, atom vs ld).
    if (options.atomics)
    {
        for (const uint64_t footprint : {footprints.l2, footprints.dram})
        {
            for (const uint64_t operation : {AtomicRed, AtomicAdd, StoreDefault, LoadDefault})
            {
                plan.insert({operation, footprint, AtomicWidth});
            }
        }

        plan.insert({LoadNone, footprints.l1, AtomicWidth});
    }

    return plan;
}

std::vector<uint64_t> SharedOps(const Options& options)
{
    return options.atomics ? std::vector<uint64_t>{SharedLoad, SharedStore, SharedRed}
        : std::vector<uint64_t>{SharedLoad, SharedStore};
}

std::vector<uint64_t> ComputeOps(const Hardware& hardware)
{
    std::vector<uint64_t> operations = {ComputeLoop, ComputeFfma, ComputeDfma, ComputeImad, ComputeHfma2};

    // HMMA in the shape the GPU has (kernels/Compute.cu): m8n8k4 on sm_70, m16n8k8 on sm_75, m16n8k16 from sm_80.
    if (hardware.computeCapabilityMajor >= 7)
    {
        operations.push_back(ComputeHmma);
    }

    operations.push_back(ComputeMufu);
    operations.push_back(ComputeFfma3);

    for (const uint64_t operation : {ComputeShfl, ComputeBar, ComputeFfmaLoop, ComputeFmulFadd, ComputeMufuSin})
    {
        operations.push_back(operation);
    }

    return operations;
}

std::vector<uint64_t> SweepCounts(const Hardware& hardware, const Options& options)
{
    std::vector<uint64_t> counts;
    const uint64_t multiprocessors = std::max<uint64_t>(hardware.multiprocessors, 1);

    if (options.smSweepAuto)
    {
        for (uint64_t count = multiprocessors; count >= 1; count /= 2)
        {
            counts.push_back(count);
        }

        if (counts.back() != 1)
        {
            counts.push_back(1);
        }
    }

    for (const uint64_t count : options.smSweep)
    {
        if (count <= multiprocessors && std::find(counts.cbegin(), counts.cend(), count) == counts.cend())
        {
            counts.push_back(count);
        }
    }

    std::sort(counts.begin(), counts.end(), std::greater<uint64_t>());
    return counts;
}

void DescribeSetup(const Hardware& hardware, const Options& options, const Footprints& footprints, const Plan& plan)
{
    const uint64_t blocks = static_cast<uint64_t>(hardware.multiprocessors) * options.blocksPerSm;

    std::cout << "Device: " << hardware.name << ", " << hardware.multiprocessors << " SMs, " << blocks << " blocks x "
        << options.blockSize << " threads\n"
        << "L1 size: " << (hardware.l1Size ? FormatBytes(hardware.l1Size.value()) : "unknown (default footprint)")
        << ", L2 size: " << (hardware.l2Size ? FormatBytes(hardware.l2Size.value()) : "unknown (default footprint)")
        << (hardware.l2SegmentSize ? ", L2 segment: " + FormatBytes(hardware.l2SegmentSize.value()) : "") << "\n"
        << "Footprints: L1 " << FormatBytes(footprints.l1) << ", L2 " << FormatBytes(footprints.l2) << ", DRAM "
        << FormatBytes(footprints.dram) << "\n"
        << "Stripes: the " << options.blocksPerSm << " blocks of an SM access disjoint stripes; footprints from "
        << FormatBytes(footprints.disjointFrom) << " on are split over all " << blocks << " blocks (" << footprints.gridStripes
        << " stripes)\n";

    if (footprints.dram < footprints.disjointFrom)
    {
        std::cout << "WARNING: the DRAM footprint is too small to split over all blocks, blocks on different SMs may share "
            << "lines in L2.\n";
    }

    if (options.mt4gFile.empty() && !options.l1Override && !options.l2Override)
    {
        std::cout << "WARNING: no MT4G file given, cache sizes are guesses. Pass --mt4g <GPU>.json for meaningful footprints.\n";
    }

    // The L1 footprint is a quarter of the L1, but at least two rows of 16-byte loads per block stripe (the minimum
    // footprint, 32 KiB with 4 blocks of 256 threads). On a GPU with 128 KiB of L1 or less the minimum wins; at half of the
    // L1 or more the L1 rows are labelled "mixed" and stay out of the L1 fits.
    if (hardware.l1Size.has_value() && ExpectedLevel(LoadDefault, footprints.l1, hardware, footprints) != "L1")
    {
        std::cout << "WARNING: the L1 footprint (" << FormatBytes(footprints.l1) << ") is not below half of the L1 ("
            << FormatBytes(hardware.l1Size.value()) << "): the L1 rows would be labelled "
            << ExpectedLevel(LoadDefault, footprints.l1, hardware, footprints) << " and left out of the L1 fits. Check the "
            << "L1 size (--l1-kib), or lower --block-size or --blocks-per-sm (the footprint is at least "
            << FormatBytes(footprints.minimum) << ").\n";
    }
    else if (hardware.l1Size.has_value() && 3 * footprints.l1 > hardware.l1Size.value())
    {
        // On an RTX 3090 the 32 KiB footprint is 30 % of the L1 and hits L1 in every row; on an RTX 2080 Ti it is half of it.
        std::cout << "WARNING: the L1 footprint (" << FormatBytes(footprints.l1) << ") is more than a third of the L1 ("
            << FormatBytes(hardware.l1Size.value()) << ", the minimum of two rows per stripe is "
            << FormatBytes(footprints.minimum) << "); with a smaller L1/shared carve-out it may not stay in L1 (the hit "
            << "rates of a run with --profile show it).\n";
    }

    if (hardware.l1Size.has_value() && footprints.l2 < 4 * hardware.l1Size.value())
    {
        std::cout << "WARNING: the L2 footprint is less than 4x the L1 size, the L2 probe may still hit L1.\n";
    }

    if (hardware.computeCapabilityMajor != 0 && hardware.computeCapabilityMajor < 7)
    {
        std::cout << "WARNING: before Volta, plain global loads are not cached in L1 by default (treat the 'ld.ca' row as the "
            << "L1 number) and the sleep baseline cannot sleep (it polls like the spin kernel).\n";
    }

    if (hardware.maxThreadsPerMultiprocessor.has_value()
        && options.blocksPerSm * options.blockSize > hardware.maxThreadsPerMultiprocessor.value())
    {
        std::cout << "WARNING: more threads per SM than can be resident, blocks will run in several waves.\n";
    }

    if (options.sharedBytes > std::min<uint64_t>(48 * KiB, hardware.sharedPerBlock)
        || (hardware.sharedPerMultiprocessor.has_value()
            && options.sharedBytes * options.blocksPerSm > hardware.sharedPerMultiprocessor.value()))
    {
        std::cout << "WARNING: the shared probe does not fit " << options.blocksPerSm << " blocks per SM.\n";
    }

    // probeScatter and probePartial number the sectors of the whole grid and advance every thread by the sectors of the
    // grid per step, so a thread comes back to its first sector after footprint / gcd(step, footprint) steps. At the L2
    // footprint (their controls) the data of one SM is therefore the footprint times the blocks per SM, divided by the
    // largest power of two in the number of blocks: half of it on an RTX 3090 (82 SMs), but 1/16 with 80 SMs and 1/64 with
    // 64, where it fits in L1 and the rows labelled L2 are L1 hits. The counters of a --profile run show it, and the
    // model prices the events that happened; the studies' L2 control and the level labels are then wrong.
    if (blocks > 0 && (options.dramLocality || options.subSector))
    {
        const uint64_t sectors = footprints.l2 / SectorBytes;
        const uint64_t l1 = hardware.l1Size.value_or(4 * footprints.l1);
        auto perSm = [&](const uint64_t sectorsPerBlock)
        {
            const uint64_t step = blocks * sectorsPerBlock;
            return std::min(footprints.l2, sectors / std::gcd(step, sectors) * sectorsPerBlock * SectorBytes * options.blocksPerSm);
        };
        const uint64_t scatter = perSm(options.blockSize * DefaultWidth / SectorBytes); // 2 threads per sector
        const uint64_t partial = perSm(options.blockSize);                               // a sector per thread

        if (options.dramLocality && scatter < 4 * l1)
        {
            std::cout << "WARNING: with " << blocks << " blocks the locality probe walks only " << FormatBytes(scatter)
                << " per SM at the L2 footprint (less than 4x the L1): its L2 control will mostly hit L1. Check its rows "
                << "with --profile (tools/check_run.py --counters).\n";
        }

        if (options.subSector && partial < 4 * l1)
        {
            std::cout << "WARNING: with " << blocks << " blocks the sub-sector probe walks only " << FormatBytes(partial)
                << " per SM at the L2 footprint (less than 4x the L1): its L2 rows will mostly hit L1. Check its rows "
                << "with --profile (tools/check_run.py --counters).\n";
        }
    }

    for (const uint64_t count : options.smSweep)
    {
        if (count > hardware.multiprocessors)
        {
            std::cout << "WARNING: --sm-sweep " << count << " is more than the " << hardware.multiprocessors
                << " SMs of this GPU and is left out.\n";
        }
    }

    std::vector<std::string> payloadNames;

    for (const uint64_t payload : options.payloads)
    {
        payloadNames.push_back(PayloadName(payload));
    }

    std::string payloads;

    for (const auto& name : payloadNames)
    {
        payloads += (payloads.empty() ? "" : " and ") + name;
    }

    const std::string duties = JoinList(options.duties, "/");
    std::cout << "Planned configurations" << (options.randomOrder ? " (probes of a group in random order)" : "") << ":\n";

    if (options.idleS > 0)
    {
        std::cout << "  idle power without a kernel, " << options.idleS << " s each before KTT starts (no CUDA context), before "
            << "the warm-up and at the end\n";
    }

    if (options.warmupS > 0)
    {
        std::cout << "  warm-up " << options.warmupS << " s (not reported)\n";
    }

    std::cout << "  global probes on " << payloads << " data, each at duty " << duties << " % (bursts of "
        << options.burstAccesses << " accesses per thread, each followed by a sleep; fit over duties >= " << options.fitMinDuty
        << " %):\n";

    for (const auto& [operation, footprint, width] : plan)
    {
        std::cout << "    " << std::left << std::setw(8) << OperationName(operation) << std::setw(10) << FormatBytes(footprint)
            << std::setw(6) << (std::to_string(width) + " B") << "expected " << ExpectedLevel(operation, footprint, hardware,
            footprints) << std::right;

        if (footprint >= footprints.disjointFrom)
        {
            std::cout << ", split over all blocks (" << FormatBytes(footprint / footprints.gridStripes * blocks) << " accessed)";
        }

        std::cout << "\n";
    }

    const std::vector<uint64_t> computeOps = ComputeOps(hardware);

    if (options.compute)
    {
        std::cout << "  arithmetic probes on " << payloads << " operands, each at duty " << duties << " %:";

        for (const uint64_t operation : computeOps)
        {
            std::cout << " " << ComputeName(operation);
        }

        std::cout << "\n";
    }

    std::cout << "  shared probes (ld.shared, st.shared" << (options.atomics ? ", red.shared" : "") << "), "
        << FormatBytes(options.sharedBytes) << " per block, on " << payloads << " data at duty " << duties << " %\n";

    if (options.constant)
    {
        std::cout << "  constant probes (ld.const, one word per warp access) on random data at duty " << duties << " %: "
            << FormatBytes(footprints.constantL1) << " expected " << ExpectedConstantLevel(footprints.constantL1, hardware,
            footprints) << ", " << FormatBytes(footprints.constantL15) << " expected " << ExpectedConstantLevel(
            footprints.constantL15, hardware, footprints) << " (constant L1 "
            << (hardware.constantL1Size ? FormatBytes(hardware.constantL1Size.value()) : "unknown") << ")\n";
    }

    if (options.texture)
    {
        std::cout << "  texture probes (tex.1d, 16-byte texels) on " << payloads << " data at duty " << duties << " %: "
            << FormatBytes(footprints.l1) << ", " << FormatBytes(footprints.l2) << ", " << FormatBytes(footprints.dram)
            << " (the footprints of the plain loads)\n";
    }

    if (options.dramLocality)
    {
        std::cout << "  DRAM locality probes (ld, st) on " << payloads << " data at duty " << duties << " %, "
            << FormatBytes(footprints.dram) << " in one stream or in runs of 32 B (one sector per L2 request), 64 B, 128 B, "
            << "512 B, 2 KiB or 8 KiB placed apart; control: stream and 32 B runs at " << FormatBytes(footprints.l2) << "\n";
    }

    if (options.subSector)
    {
        std::cout << "  sub-sector probes (ld, st of 4 and 8 bytes, every thread walking a sector of its own) on " << payloads
            << " data at duty " << duties << " %: " << FormatBytes(footprints.dram) << " with the sectors of a warp adjacent "
            << "or scattered, " << FormatBytes(footprints.l2) << " scattered\n";
    }

    if (options.bankConflicts)
    {
        std::cout << "  bank-conflict probes (ld.shared, st.shared; " << JoinList(BankWaysOfWidth(4), "/") << " times the "
            << "wavefronts of the conflict-free access for 4 bytes, up to " << BankWaysOfWidth(8).back() << " for 8 and "
            << BankWaysOfWidth(16).back() << " for 16 bytes) on random data at duty " << duties << " %\n";
    }

    if (options.indexing)
    {
        std::cout << "  indexing probes (32 FFMA accumulators, static and runtime indices) on " << payloads << " data at duty "
            << duties << " %\n";
    }

    if (options.mixed)
    {
        std::cout << "  mixed-intensity probes (16-byte ld with " << JoinList(MixedFmas, "/") << " FFMAs per load) on random "
            << "data at duty 100: " << FormatBytes(footprints.l1) << ", " << FormatBytes(footprints.l2) << ", "
            << FormatBytes(footprints.dram) << "\n";
    }

    if (options.mixedInt)
    {
        std::cout << "  loads with integer work (16-byte ld with 0/4/8/16 add, shift, and per load) on random data at duty "
            << "100: " << FormatBytes(footprints.l1) << ", " << FormatBytes(footprints.l2) << "\n";
    }

    if (options.writes)
    {
        std::cout << "  write probes (copy, update in place: " << FormatBytes(footprints.l2) << ", "
            << FormatBytes(footprints.dram) << "; local memory: " << FormatBytes(footprints.l1) << " and "
            << FormatBytes(WriteLocalFrameBytes * options.blockSize * options.blocksPerSm) << " per SM; "
            << JoinList(WriteWidths, "/") << " B) on random data at duty " << duties << " %\n";
    }

    if (options.latency)
    {
        std::cout << "  latency probes (chains of dependent 4-byte loads; " << JoinList(LatencyChains, "/")
            << " chains per warp, " << LatencyWarps.front() * options.blocksPerSm << "-" << LatencyWarps.back() * options.blocksPerSm
            << " working warps per SM) on random data at duty 100: " << FormatBytes(footprints.l2) << ", "
            << FormatBytes(footprints.dram) << "\n";
    }

    std::cout << "  order: sleep baseline, then for each duty: global" << (options.compute ? " and arithmetic" : "")
        << " probes + sleep baseline, then shared" << (options.latency ? ", latency" : "") << (options.bankConflicts ? ", bank conflicts" : "") << (options.constant ? ", constant" : "") << (options.texture ? ", texture" : "")
        << (options.dramLocality ? ", DRAM locality" : "") << (options.subSector ? ", sub-sector" : "")
        << (options.indexing ? ", indexing" : "") << (options.lanes ? ", lanes" : "")
        << (options.misaligned ? ", misaligned" : "") << (options.writePath ? ", write path" : "")
        << (options.instrClasses ? ", instruction classes" : "")
        << (options.mixed ? ", mixed intensity" : "")
        << (options.writes ? ", writes" : "") << (options.payloadBits.empty() ? "" : ", payload-bit loads, operands")
        << ", each followed by a sleep baseline, with the spin reference (heavy instruction issue) before the last\n";

    if (options.l1Sweep)
    {
        std::cout << "  L1 sweep: ld and st at " << FormatBytes(footprints.l1) << " with 1-32 sectors in 1/4/32 lines per warp "
            << "instruction (7 patterns), 4- and 16-byte words, at duty " << JoinList(options.duties, "/") << " %\n";
    }

    if (options.writePath)
    {
        std::cout << "  write path: 4-byte ld and st of 1/2/4/8 words per sector, 4/2/1 sectors per L2 request, at "
            << FormatBytes(footprints.l2) << " and " << FormatBytes(footprints.dram) << ", at duty "
            << JoinList(options.duties, "/") << " %\n";
    }

    if (options.misaligned)
    {
        std::cout << "  misaligned: 4- and 16-byte stream ld at " << FormatBytes(footprints.l1) << ", " << FormatBytes(footprints.l2)
            << ", " << FormatBytes(footprints.dram) << " and st at the last two, started 16/32/64/96 bytes off the line, at duty " << JoinList(options.duties, "/") << " %\n";
    }

    if (options.lanes)
    {
        std::cout << "  lanes: FFMA and IADD with " << JoinList(LaneCounts, "/") << " of 32 lanes active (guard predicate, "
            << "branch), at duty " << JoinList(options.duties, "/") << " %\n";
    }

    if (!options.payloadBits.empty())
    {
        std::cout << "  payload bits: 16-byte ld at " << FormatBytes(footprints.l1) << ", " << FormatBytes(footprints.l2)
            << ", " << FormatBytes(footprints.dram) << ", the no-load control, and FFMA, DFMA" << (hardware.computeCapabilityMajor >= 8 ? ", HMMA"
            : "") << " on operands from the payload (probeOperands), each on random, zero and";

        for (const auto& pattern : options.payloadBits)
        {
            std::cout << " " << pattern;
        }

        std::cout << ", at duty " << JoinList(options.duties, "/") << " %\n";
    }

    const std::vector<uint64_t> sweep = SweepCounts(hardware, options);

    if (!sweep.empty())
    {
        std::cout << "  SM sweep at duty 100 on random data, for " << JoinList(sweep, "/") << " working SMs: sleep baseline"
            << (options.sweepStudies ? ", ld L1, no-load, ld L2, ld.shared" : "")
            << (options.compute && options.sweepStudies ? ", ffma" : "")
            << (options.latency ? ", dependent loads (1 and 16 chains) at L2 and DRAM" : "") << "\n";
    }

    size_t bankDegrees = 0; // (width, degree) combinations of the bank-conflict probe

    for (const uint64_t width : BankWidths)
    {
        bankDegrees += BankWaysOfWidth(width).size();
    }

    const size_t perDuty = options.payloads.size() * (plan.size() + (options.compute ? computeOps.size() : 0)
        + SharedOps(options).size() + (options.texture ? 3 : 0) + (options.dramLocality ? 2 * LocalityRuns.size() + 4 : 0)
        + (options.subSector ? 12 : 0) + (options.indexing ? 2 : 0))
        + (options.constant ? 2 : 0) + (options.bankConflicts ? 2 * bankDegrees : 0);
    // A baseline before the probes, after every duty group and after every group that holds all duties (as SleepCount
    // in ProbeRun.cpp).
    const size_t groups = 1 + (options.bankConflicts ? 1 : 0) + (options.constant ? 1 : 0) + (options.texture ? 1 : 0)
        + (options.dramLocality ? 1 : 0) + (options.subSector ? 1 : 0) + (options.indexing ? 1 : 0)
        + (options.lanes ? 1 : 0) + (options.misaligned ? 1 : 0) + (options.writePath ? 1 : 0)
        + (options.instrClasses ? 1 : 0) + (options.mixed || options.mixedInt ? 1 : 0) + (options.writes ? 1 : 0)
        + (options.l1Sweep ? 1 : 0)
        + (options.latency ? 1 : 0)
        + (options.payloadBits.empty() ? 0 : 2);
    const size_t sleeps = 1 + options.duties.size() + groups;
    const size_t sweepConfigurations = sweep.size() * (1 + (options.sweepStudies ? 4 + (options.compute ? 1 : 0) : 0)
        + (options.latency ? 4 : 0));
    const size_t mixedConfigurations = (options.mixed ? 3 * MixedFmas.size() : 0) + (options.mixedInt ? 2 * 4 : 0); // duty 100
    // Three patterns at two footprints each, random data only.
    const size_t writeConfigurations = options.writes ? 3 * 2 * WriteWidths.size() * options.duties.size() : 0;
    const size_t latencyConfigurations = options.latency ? 2 * LatencyChains.size() * LatencyWarps.size() : 0;
    // --payload-bits: 16-byte loads at three footprints and FFMA, DFMA (and HMMA from sm_80) on random, zero and the
    // patterns, at every duty.
    const size_t patternPayloads = options.payloadBits.empty() ? 0 : 2 + options.payloadBits.size();
    const size_t patternConfigurations = patternPayloads * options.duties.size()
        * (4 + 2 + (hardware.computeCapabilityMajor >= 8 ? 1 : 0));
    // --lanes: 4 operations at 7 lane counts, random data, every duty.
    const size_t laneConfigurations = options.lanes ? 5 * LaneCounts.size() * options.duties.size() : 0;
    // --misaligned: 10 streams (4 and 16 bytes) at every offset.
    const size_t misalignedConfigurations = options.misaligned ? 10 * MisalignedOffsets.size() * options.duties.size() : 0;
    const size_t configurations = perDuty * options.duties.size() + 1 + sleeps + sweepConfigurations + mixedConfigurations
        + writeConfigurations + latencyConfigurations + patternConfigurations + laneConfigurations
        + misalignedConfigurations + (options.writePath ? 2 * 2 * 3 * 4 * options.duties.size() : 0)
        + (options.instrClasses ? InstructionClassOps.size() * options.duties.size() : 0)
        + (options.l1Sweep ? 2 * 2 * 7 * options.duties.size() : 0);
    const uint64_t fixedS = options.warmupS + 3 * options.idleS;
    std::cout << configurations << " configurations (" << sleeps + sweep.size() << " of them baselines), about "
        << (configurations * options.minTimeMs / 1000 + fixedS) / 60 << "-"
        << (configurations * options.maxTimeMs / 1000 + fixedS) / 60 << " min of measurement\n";
    // A configuration whose power does not settle runs to --max-time-ms (266 of 816 on an RTX 2080 Ti, where a run took
    // 7.1 h instead of 2.2 h), so a fixed queue timeout can kill a run that is fine.
    std::cout << "A timeout around the run must allow the upper bound: at least "
        << (configurations * options.maxTimeMs / 1000 + fixedS) / 60 + 10 << " min (timeout "
        << (configurations * options.maxTimeMs / 1000 + fixedS + 600) << ").\n";
}
