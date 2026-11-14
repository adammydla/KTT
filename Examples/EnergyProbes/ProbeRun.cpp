#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Ktt.h>

#include "Analysis.h"
#include "CounterTable.h"
#include "GpuSession.h"
#include "ProbeKernels.h"
#include "ProbePlan.h"
#include "ProbeRun.h"
#include "Report.h"

namespace
{

// The kernels of the SM sweep for one number of working SMs.
struct SweepKernels
{
    uint64_t count;
    ktt::KernelId sleep;
    std::optional<ktt::KernelId> global;   // the study probes (not with --model)
    std::optional<ktt::KernelId> shared;
    std::optional<ktt::KernelId> compute;
    std::optional<ktt::KernelId> latency;
};

// The KTT kernels of a probe run besides the warm-up and the baselines (KTT needs every kernel before the first one
// runs). Every kernel holds the configurations of one group: its parameters are the probe's variants, and KTT runs all
// configurations of a kernel in turn.
struct ProbeRunKernels
{
    std::vector<ktt::KernelId> globals;  // one per duty, so that baselines can be measured between the groups
    std::vector<ktt::KernelId> computes; // one per duty (--compute)
    ktt::KernelId shared;
    ktt::KernelId spin;
    std::optional<ktt::KernelId> constant;
    std::optional<ktt::KernelId> scatter;
    std::optional<ktt::KernelId> partial;
    std::optional<ktt::KernelId> banks;
    std::optional<ktt::KernelId> indexing;
    std::optional<ktt::KernelId> lanes; // --lanes
    std::optional<ktt::KernelId> misaligned; // --misaligned
    std::optional<ktt::KernelId> writePath;  // --write-path
    std::optional<ktt::KernelId> l1Sweep;    // --l1-sweep
    std::optional<ktt::KernelId> classes;    // --instr-classes
    std::optional<ktt::KernelId> mixed;
    std::optional<ktt::KernelId> write;
    std::optional<ktt::KernelId> latency;
    std::optional<ktt::KernelId> texture;
    std::optional<ktt::KernelId> patternLoads; // --payload-bits: 16-byte loads on the patterns
    std::optional<ktt::KernelId> operands;     // --payload-bits: probeOperands
    std::vector<SweepKernels> sweep;
};

// The parameters of a global-probe kernel: the (LOAD_OP, FOOTPRINT, WIDTH) combinations of `plan`.
void AddGlobalParameters(ktt::Tuner& tuner, const ktt::KernelId kernel, const Plan& plan, const Footprints& footprints)
{
    const std::vector<uint64_t> loadOps = {LoadDefault, LoadCg, LoadNone, LoadCa, LoadNc, StoreDefault, AtomicRed, AtomicAdd};
    std::set<uint64_t> kernelFootprints;
    std::set<uint64_t> kernelWidths;

    for (const auto& [operation, footprint, width] : plan)
    {
        kernelFootprints.insert(footprint);
        kernelWidths.insert(width);
    }

    tuner.AddParameter(kernel, "LOAD_OP", loadOps);
    tuner.AddParameter(kernel, "FOOTPRINT", std::vector<uint64_t>(kernelFootprints.cbegin(), kernelFootprints.cend()));
    tuner.AddParameter(kernel, "WIDTH", std::vector<uint64_t>(kernelWidths.cbegin(), kernelWidths.cend()));
    tuner.AddParameter(kernel, "GRID_STRIPES", std::vector<uint64_t>{footprints.gridStripes});
    tuner.AddConstraint(kernel, {"LOAD_OP", "FOOTPRINT", "WIDTH"}, [plan](const std::vector<uint64_t>& values)
    {
        return plan.count({values[0], values[1], values[2]}) > 0;
    });
}

// The global (and arithmetic) probes of every duty. The payloads of a duty are in the same kernel, so that random and
// zero rows run under the same thermal conditions.
void CreateDutyKernels(GpuSession& session, const ProbeDefinitions& definitions, const ProbeSettings& settings,
    const ktt::ArgumentId counts, const Plan& plan, const Options& options, ProbeRunKernels& kernels)
{
    ktt::Tuner& tuner = session.GetTuner();
    const std::vector<uint64_t> computeOps = ComputeOps(session.GetHardware());

    for (const uint64_t duty : options.duties)
    {
        const ktt::KernelId global = session.CreateKernel("Global" + std::to_string(duty), definitions.global,
            settings.global, true, counts);
        AddGlobalParameters(tuner, global, plan, session.GetFootprints());
        tuner.AddParameter(global, "DUTY", std::vector<uint64_t>{duty});
        tuner.AddParameter(global, "PAYLOAD", options.payloads);
        kernels.globals.push_back(global);

        if (options.compute)
        {
            const ktt::KernelId compute = session.CreateKernel("Compute" + std::to_string(duty), definitions.compute,
                settings.compute, false, counts);
            tuner.AddParameter(compute, "COMPUTE_OP", computeOps);
            tuner.AddParameter(compute, "DUTY", std::vector<uint64_t>{duty});
            tuner.AddParameter(compute, "PAYLOAD", options.payloads);
            kernels.computes.push_back(compute);
        }
    }
}

// The probes that hold all duties in one kernel: shared memory, the spin reference and the optional studies.
void CreateGroupKernels(GpuSession& session, const ProbeDefinitions& definitions, const ProbeSettings& settings,
    const ktt::ArgumentId counts, const Options& options, ProbeRunKernels& kernels)
{
    ktt::Tuner& tuner = session.GetTuner();
    const Footprints& footprints = session.GetFootprints();

    kernels.shared = session.CreateKernel("Shared", definitions.shared, settings.common, false, counts);
    kernels.spin = session.CreateKernel("Spin", definitions.spin, Configure{}, false, counts);
    tuner.AddParameter(kernels.shared, "SMEM_BYTES", std::vector<uint64_t>{options.sharedBytes});
    tuner.AddParameter(kernels.shared, "SHARED_OP", SharedOps(options));
    tuner.AddParameter(kernels.shared, "DUTY", options.duties);
    tuner.AddParameter(kernels.shared, "PAYLOAD", options.payloads);

    // Constant memory holds random words only (zeros would need a second 64 KiB symbol, more than a module may have).
    if (definitions.constant.has_value())
    {
        kernels.constant = session.CreateKernel("Constant", definitions.constant.value(), settings.constant, true, counts);
        tuner.AddParameter(kernels.constant.value(), "CONST_BYTES", std::vector<uint64_t>{footprints.constantL1,
            footprints.constantL15});
        tuner.AddParameter(kernels.constant.value(), "DUTY", options.duties);
        tuner.AddParameter(kernels.constant.value(), "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
    }

    // Granularity and DRAM row locality: loads and stores at the DRAM footprint in one stream or in runs of RUN_BYTES placed
    // apart, and as a control the stream and 32-byte runs at the L2 footprint (same SASS, no DRAM).
    if (definitions.scatter.has_value())
    {
        kernels.scatter = session.CreateKernel("Scatter", definitions.scatter.value(), settings.scatter, false, counts);
        tuner.AddParameter(kernels.scatter.value(), "LOAD_OP", std::vector<uint64_t>{LoadDefault, StoreDefault});
        tuner.AddParameter(kernels.scatter.value(), "FOOTPRINT", std::vector<uint64_t>{footprints.l2, footprints.dram});
        tuner.AddParameter(kernels.scatter.value(), "WIDTH", std::vector<uint64_t>{DefaultWidth});
        tuner.AddParameter(kernels.scatter.value(), "RUN_BYTES", LocalityRuns);
        tuner.AddParameter(kernels.scatter.value(), "DUTY", options.duties);
        tuner.AddParameter(kernels.scatter.value(), "PAYLOAD", options.payloads);
        const uint64_t dram = footprints.dram;
        tuner.AddConstraint(kernels.scatter.value(), {"LOAD_OP", "FOOTPRINT", "RUN_BYTES"},
            [dram](const std::vector<uint64_t>& values)
        {
            return values[1] == dram || values[2] <= SectorBytes;
        });
    }

    // Sub-sector accesses: 4- and 8-byte loads and stores at the DRAM footprint with the sectors of a warp adjacent or
    // scattered, and scattered at the L2 footprint.
    if (options.subSector)
    {
        kernels.partial = session.CreateKernel("Partial", definitions.partial.value(), settings.partial, false, counts);
        tuner.AddParameter(kernels.partial.value(), "LOAD_OP", std::vector<uint64_t>{LoadDefault, StoreDefault});
        tuner.AddParameter(kernels.partial.value(), "FOOTPRINT", std::vector<uint64_t>{footprints.l2, footprints.dram});
        tuner.AddParameter(kernels.partial.value(), "WIDTH", std::vector<uint64_t>{4, 8});
        tuner.AddParameter(kernels.partial.value(), "SPREAD", std::vector<uint64_t>{0, 1});
        tuner.AddParameter(kernels.partial.value(), "DUTY", options.duties);
        tuner.AddParameter(kernels.partial.value(), "PAYLOAD", options.payloads);
        const uint64_t dram = footprints.dram;
        tuner.AddConstraint(kernels.partial.value(), {"FOOTPRINT", "SPREAD"}, [dram](const std::vector<uint64_t>& values)
        {
            return values[0] == dram || values[1] != 0;
        });
    }

    // Bank conflicts: 4-, 8- and 16-byte shared loads and stores with every conflict degree of up to 32 wavefronts per
    // instruction, on random data.
    if (definitions.banks.has_value())
    {
        kernels.banks = session.CreateKernel("Banks", definitions.banks.value(), settings.banks, false, counts);
        tuner.AddParameter(kernels.banks.value(), "SMEM_BYTES", std::vector<uint64_t>{options.sharedBytes});
        tuner.AddParameter(kernels.banks.value(), "SHARED_OP", std::vector<uint64_t>{SharedLoad, SharedStore});
        tuner.AddParameter(kernels.banks.value(), "SHARED_WIDTH", BankWidths);
        tuner.AddParameter(kernels.banks.value(), "BANK_WAYS", BankWays);
        tuner.AddParameter(kernels.banks.value(), "DUTY", options.duties);
        tuner.AddParameter(kernels.banks.value(), "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
        tuner.AddConstraint(kernels.banks.value(), {"SHARED_WIDTH", "BANK_WAYS"}, [](const std::vector<uint64_t>& values)
        {
            return values[1] * values[0] <= WavefrontBytes;
        });
    }

    // Register-array indexing: the FFMA accumulators with static and with runtime indices, at every duty.
    if (options.indexing)
    {
        kernels.indexing = session.CreateKernel("Indexing", definitions.compute, settings.compute, false, counts);
        tuner.AddParameter(kernels.indexing.value(), "COMPUTE_OP", std::vector<uint64_t>{ComputeStaticIndex,
            ComputeDynamicIndex});
        tuner.AddParameter(kernels.indexing.value(), "DUTY", options.duties);
        tuner.AddParameter(kernels.indexing.value(), "PAYLOAD", options.payloads);
    }

    // --instr-classes: the instruction classes inside one event, at every duty.
    if (options.instrClasses)
    {
        kernels.classes = session.CreateKernel("Classes", definitions.compute, settings.compute, false, counts);
        tuner.AddParameter(kernels.classes.value(), "COMPUTE_OP", InstructionClassOps);
        tuner.AddParameter(kernels.classes.value(), "DUTY", options.duties);
        tuner.AddParameter(kernels.classes.value(), "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
    }

    // --write-path: 4-byte loads and stores of 1, 2, 4 or 8 words of a sector, with 1, 2 or 4 sectors per L2 request, at
    // the L2 and DRAM footprints (probePartial), random data, every duty.
    // --l1-sweep: L1 loads and stores with 1-32 sectors in 1, 4 or 32 lines per warp instruction (sectors between the
    // lines and four times the lines), 4- and 16-byte words, every duty.
    if (options.l1Sweep)
    {
        kernels.l1Sweep = session.CreateKernel("L1Sweep", definitions.l1Sweep.value(), settings.l1Sweep, false, counts);
        tuner.AddParameter(kernels.l1Sweep.value(), "LOAD_OP", std::vector<uint64_t>{LoadDefault, StoreDefault});
        tuner.AddParameter(kernels.l1Sweep.value(), "FOOTPRINT", std::vector<uint64_t>{footprints.l1});
        tuner.AddParameter(kernels.l1Sweep.value(), "WIDTH", std::vector<uint64_t>{4, 16});
        tuner.AddParameter(kernels.l1Sweep.value(), "SWEEP_SECTORS", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
        tuner.AddParameter(kernels.l1Sweep.value(), "SWEEP_LINES", std::vector<uint64_t>{1, 4, 32});
        tuner.AddConstraint(kernels.l1Sweep.value(), {"SWEEP_SECTORS", "SWEEP_LINES"}, [](const std::vector<uint64_t>& v)
            { return v[1] <= v[0] && v[0] <= 4 * v[1]; });
        tuner.AddParameter(kernels.l1Sweep.value(), "DUTY", options.duties);
        tuner.AddParameter(kernels.l1Sweep.value(), "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
    }

    if (options.writePath)
    {
        kernels.writePath = session.CreateKernel("WritePath", definitions.partial.value(), settings.partial, false, counts);
        tuner.AddParameter(kernels.writePath.value(), "LOAD_OP", std::vector<uint64_t>{LoadDefault, StoreDefault});
        tuner.AddParameter(kernels.writePath.value(), "FOOTPRINT", std::vector<uint64_t>{footprints.l2, footprints.dram});
        tuner.AddParameter(kernels.writePath.value(), "WIDTH", std::vector<uint64_t>{4});
        tuner.AddParameter(kernels.writePath.value(), "SPREAD", std::vector<uint64_t>{0, 2, 1});
        tuner.AddParameter(kernels.writePath.value(), "PARTIAL_WORDS", std::vector<uint64_t>{1, 2, 4, 8});
        tuner.AddParameter(kernels.writePath.value(), "DUTY", options.duties);
        tuner.AddParameter(kernels.writePath.value(), "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
    }

    // --misaligned: the stream load and store with their start off the line (probeGlobal on views of the payload).
    if (options.misaligned)
    {
        const Plan streams = {{LoadDefault, footprints.l1, 4}, {LoadDefault, footprints.l1, 16}, {LoadDefault, footprints.l2, 4},
            {LoadDefault, footprints.l2, 16}, {LoadDefault, footprints.dram, 4}, {LoadDefault, footprints.dram, 16},
            {StoreDefault, footprints.l2, 4}, {StoreDefault, footprints.l2, 16}, {StoreDefault, footprints.dram, 4},
            {StoreDefault, footprints.dram, 16}};
        kernels.misaligned = session.CreateKernel("Misaligned", definitions.global, settings.global, true, counts);
        AddGlobalParameters(tuner, kernels.misaligned.value(), streams, footprints);
        tuner.AddParameter(kernels.misaligned.value(), "OFFSET_BYTES", MisalignedOffsets);
        tuner.AddParameter(kernels.misaligned.value(), "DUTY", options.duties);
        tuner.AddParameter(kernels.misaligned.value(), "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
    }

    // --lanes: FFMA and IADD with some of the 32 lanes active, by predicate and by branch, at every duty.
    if (options.lanes)
    {
        kernels.lanes = session.CreateKernel("Lanes", definitions.compute, settings.compute, false, counts);
        tuner.AddParameter(kernels.lanes.value(), "COMPUTE_OP", std::vector<uint64_t>{ComputeFfmaLanes,
            ComputeFfmaLanesBranch, ComputeIaddLanes, ComputeIaddLanesBranch, ComputeFfmaLanesRotating});
        tuner.AddParameter(kernels.lanes.value(), "LANES", LaneCounts);
        tuner.AddParameter(kernels.lanes.value(), "DUTY", options.duties);
        tuner.AddParameter(kernels.lanes.value(), "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
    }

    // Mixed intensity: the 16-byte loads at the three footprints with 0 to 1024 FFMAs per load, on random data at full rate
    // only (with the per-warp sleeps of a lower duty, the other warps would fill the FFMA pipeline; see IdleForDutyWindow).
    if (definitions.mixed.has_value())
    {
        kernels.mixed = session.CreateKernel("Mixed", definitions.mixed.value(), settings.mixed, true, counts);
        tuner.AddParameter(kernels.mixed.value(), "FOOTPRINT", std::vector<uint64_t>{footprints.l1, footprints.l2,
            footprints.dram});
        tuner.AddParameter(kernels.mixed.value(), "WIDTH", std::vector<uint64_t>{DefaultWidth});
        tuner.AddParameter(kernels.mixed.value(), "GRID_STRIPES", std::vector<uint64_t>{footprints.gridStripes});
        tuner.AddParameter(kernels.mixed.value(), "MIX_FMAS", MixedFmas);
        // MIX_OP 0: FFMAs (--mixed, every count and footprint); 1: integer work (--mixed-int, 0-16 per load at L1, L2).
        std::vector<uint64_t> mixOps;

        if (options.mixed)
        {
            mixOps.push_back(0);
        }

        if (options.mixedInt)
        {
            mixOps.push_back(1);
        }

        tuner.AddParameter(kernels.mixed.value(), "MIX_OP", mixOps);
        const uint64_t dram = footprints.dram;
        tuner.AddConstraint(kernels.mixed.value(), {"MIX_OP", "MIX_FMAS", "FOOTPRINT"}, [dram](const std::vector<uint64_t>& v)
            { return v[0] == 0 || (v[1] <= 16 && v[2] != dram); });
        tuner.AddParameter(kernels.mixed.value(), "DUTY", std::vector<uint64_t>{100});
        tuner.AddParameter(kernels.mixed.value(), "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
    }

    // Dependent loads at the L2 and DRAM footprints with 1-16 chains per warp and 1-8 working warps per block, at duty 100
    // (the access rate varies with the parallelism instead), on random data.
    if (definitions.latency.has_value())
    {
        kernels.latency = session.CreateKernel("Latency", definitions.latency.value(), settings.latency, false, counts);
        tuner.AddParameter(kernels.latency.value(), "FOOTPRINT", std::vector<uint64_t>{footprints.l2, footprints.dram});
        tuner.AddParameter(kernels.latency.value(), "WIDTH", std::vector<uint64_t>{4});
        tuner.AddParameter(kernels.latency.value(), "CHAINS", LatencyChains);
        tuner.AddParameter(kernels.latency.value(), "ACTIVE_WARPS", LatencyWarps);
        tuner.AddParameter(kernels.latency.value(), "DUTY", std::vector<uint64_t>{100});
        tuner.AddParameter(kernels.latency.value(), "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
    }

    // Loads and stores together, on random data: copy and update in place at the L2 and DRAM footprints, local memory at
    // the L1 footprint (bytes per SM) and with the whole frame of every thread (far beyond L2 for all SMs together).
    if (definitions.write.has_value())
    {
        const ktt::KernelId write = session.CreateKernel("Write", definitions.write.value(), settings.write, true, counts);
        const uint64_t l1 = footprints.l1;
        const uint64_t l2 = footprints.l2;
        const uint64_t dram = footprints.dram;
        const uint64_t localDram = WriteLocalFrameBytes * options.blockSize * options.blocksPerSm;
        const std::set<uint64_t> writeFootprints{l1, l2, dram, localDram};
        kernels.write = write;
        tuner.AddParameter(write, "WRITE_OP", std::vector<uint64_t>{WriteCopy, WriteUpdate, WriteLocal});
        tuner.AddParameter(write, "FOOTPRINT", std::vector<uint64_t>(writeFootprints.cbegin(), writeFootprints.cend()));
        tuner.AddParameter(write, "WIDTH", WriteWidths);
        tuner.AddParameter(write, "GRID_STRIPES", std::vector<uint64_t>{footprints.gridStripes});
        tuner.AddParameter(write, "DUTY", options.duties);
        tuner.AddParameter(write, "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
        tuner.AddConstraint(write, {"WRITE_OP", "FOOTPRINT"}, [l1, l2, dram, localDram](const std::vector<uint64_t>& values)
        {
            return values[0] == WriteLocal ? values[1] == l1 || values[1] == localDram : values[1] == l2 || values[1] == dram;
        });
    }

    // Texture fetches at the footprints of the plain loads (16-byte texels).
    if (definitions.texture.has_value())
    {
        kernels.texture = session.CreateKernel("Texture", definitions.texture.value(), settings.texture, true, counts);
        tuner.AddParameter(kernels.texture.value(), "FOOTPRINT", std::vector<uint64_t>{footprints.l1, footprints.l2,
            footprints.dram});
        tuner.AddParameter(kernels.texture.value(), "WIDTH", std::vector<uint64_t>{DefaultWidth});
        tuner.AddParameter(kernels.texture.value(), "GRID_STRIPES", std::vector<uint64_t>{footprints.gridStripes});
        tuner.AddParameter(kernels.texture.value(), "DUTY", options.duties);
        tuner.AddParameter(kernels.texture.value(), "PAYLOAD", options.payloads);
    }

    // --payload-bits: 16-byte loads at the L1, L2 and DRAM footprints and FFMA, DFMA and HMMA (sm_80+) on every payload
    // (random, zero and the patterns), at every duty.
    if (!options.payloadBits.empty())
    {
        std::vector<uint64_t> payloads = {PayloadRandom, PayloadZero};

        for (size_t i = 0; i < options.payloadBits.size(); ++i)
        {
            payloads.push_back(PayloadPatternFirst + i);
        }

        // The no-load loop reads no data: a control whose energy must not depend on the payload.
        const Plan loads = {{LoadDefault, footprints.l1, 16}, {LoadDefault, footprints.l2, 16},
            {LoadDefault, footprints.dram, 16}, {LoadNone, footprints.l1, 16}};
        kernels.patternLoads = session.CreateKernel("PatternLoads", definitions.global, settings.global, true, counts);
        AddGlobalParameters(tuner, kernels.patternLoads.value(), loads, footprints);
        tuner.AddParameter(kernels.patternLoads.value(), "DUTY", options.duties);
        tuner.AddParameter(kernels.patternLoads.value(), "PAYLOAD", payloads);

        std::vector<uint64_t> operations = {OperandFfma, OperandDfma};

        if (session.GetHardware().computeCapabilityMajor >= 8)
        {
            operations.push_back(OperandHmma);
        }

        kernels.operands = session.CreateKernel("Operands", definitions.operands.value(), settings.operands, false, counts);
        tuner.AddParameter(kernels.operands.value(), "OPERAND_OP", operations);
        tuner.AddParameter(kernels.operands.value(), "DUTY", options.duties);
        tuner.AddParameter(kernels.operands.value(), "PAYLOAD", payloads);
    }
}

// SM sweep: for every count, a sleep baseline and the SM-bound and L2 probes with only that many SMs working.
void CreateSweepKernels(GpuSession& session, const ProbeDefinitions& definitions, const ProbeSettings& settings,
    const ktt::ArgumentId counts, const Options& options, ProbeRunKernels& kernels)
{
    ktt::Tuner& tuner = session.GetTuner();
    const Footprints& footprints = session.GetFootprints();
    const Plan sweepPlan = {{LoadDefault, footprints.l1, DefaultWidth}, {LoadDefault, footprints.l2, DefaultWidth},
        {LoadNone, footprints.l1, DefaultWidth}};

    for (const uint64_t count : SweepCounts(session.GetHardware(), options))
    {
        const std::string suffix = "Sm" + std::to_string(count);
        const std::vector<uint64_t> activeSms = {count};
        SweepKernels sweep{count,
            session.CreateKernel("Sleep" + suffix, definitions.sleep, settings.common, false, counts),
            std::nullopt, std::nullopt, std::nullopt, std::nullopt};
        tuner.AddParameter(sweep.sleep, "ACTIVE_SMS", activeSms);

        // The study probes: L1, L2, no-load, shared and FFMA on `count` SMs (power of awake and active SMs). With
        // --model only the baselines and the dependent loads run, which are what the model uses (tools/fit_busy.py).
        if (options.sweepStudies)
        {
            sweep.global = session.CreateKernel("Global" + suffix, definitions.global, settings.global, true, counts);
            sweep.shared = session.CreateKernel("Shared" + suffix, definitions.shared, settings.common, false, counts);
            AddGlobalParameters(tuner, sweep.global.value(), sweepPlan, footprints);

            for (const ktt::KernelId kernel : {sweep.global.value(), sweep.shared.value()})
            {
                tuner.AddParameter(kernel, "DUTY", std::vector<uint64_t>{100});
                tuner.AddParameter(kernel, "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
                tuner.AddParameter(kernel, "ACTIVE_SMS", activeSms);
            }

            tuner.AddParameter(sweep.shared.value(), "SMEM_BYTES", std::vector<uint64_t>{options.sharedBytes});
            tuner.AddParameter(sweep.shared.value(), "SHARED_OP", std::vector<uint64_t>{0});
        }

        if (options.compute && options.sweepStudies)
        {
            sweep.compute = session.CreateKernel("Compute" + suffix, definitions.compute, settings.compute, false, counts);
            tuner.AddParameter(sweep.compute.value(), "COMPUTE_OP", std::vector<uint64_t>{ComputeFfma});
            tuner.AddParameter(sweep.compute.value(), "DUTY", std::vector<uint64_t>{100});
            tuner.AddParameter(sweep.compute.value(), "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
            tuner.AddParameter(sweep.compute.value(), "ACTIVE_SMS", activeSms);
        }

        // With --latency: one working warp per block with 1 or 16 chains at the L2 and DRAM footprints. Its lines are random
        // over the whole footprint, so it stays at its level on any number of SMs (the stripes of the plain DRAM load would
        // fit in L2 on a few SMs), and the access rate falls with the SMs: steady traffic at low rates. The two chain
        // counts give two rates on the same SMs and warps, which separates a power per working SM from one per access.
        if (definitions.latency.has_value())
        {
            sweep.latency = session.CreateKernel("Latency" + suffix, definitions.latency.value(), settings.latency, false,
                counts);
            tuner.AddParameter(sweep.latency.value(), "FOOTPRINT", std::vector<uint64_t>{footprints.l2, footprints.dram});
            tuner.AddParameter(sweep.latency.value(), "WIDTH", std::vector<uint64_t>{4});
            tuner.AddParameter(sweep.latency.value(), "CHAINS", std::vector<uint64_t>{1, 16});
            tuner.AddParameter(sweep.latency.value(), "ACTIVE_WARPS", std::vector<uint64_t>{1});
            tuner.AddParameter(sweep.latency.value(), "DUTY", std::vector<uint64_t>{100});
            tuner.AddParameter(sweep.latency.value(), "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
            tuner.AddParameter(sweep.latency.value(), "ACTIVE_SMS", activeSms);
        }

        kernels.sweep.push_back(sweep);
    }
}

// A group of probes that holds all duties: its kernel and the labels of its rows.
struct GroupRun
{
    ktt::KernelId kernel;
    std::string label;
    std::string probe;
    std::string operation;
};

// The groups that hold all duties, in the order they run. The dependent loads come right after the shared probes, before
// the groups that load DRAM hardest (locality, sub-sector, mixed, writes): after them, their 16-chain DRAM row (about
// 280 W) hit a thermal limit (sw-thermal, SM clock -28 %) in two runs on an RTX 3090 while the GPU read only 72 C, most
// likely the GDDR6X memory, whose temperature NVML does not report on GeForce cards. Right after the shared probes, as
// in the runs Instr and LowRate, it ran at the locked clock.
std::vector<GroupRun> GroupRuns(const ProbeRunKernels& kernels)
{
    std::vector<GroupRun> groups = {{kernels.shared, "shared", "shared", "ld.shared"}};
    const std::vector<std::pair<std::optional<ktt::KernelId>, GroupRun>> optional = {
        {kernels.latency, {0, "latency", "latency", ""}},
        {kernels.banks, {0, "banks", "banks", ""}},
        {kernels.constant, {0, "constant", "constant", "ld.const"}},
        {kernels.texture, {0, "texture", "texture", "tex"}},
        {kernels.scatter, {0, "scatter", "locality", ""}},
        {kernels.partial, {0, "partial", "sub-sector", ""}},
        {kernels.indexing, {0, "indexing", "indexing", ""}},
        {kernels.lanes, {0, "lanes", "lanes", ""}},
        {kernels.misaligned, {0, "misaligned", "misaligned", ""}},
        {kernels.writePath, {0, "write_path", "write path", ""}},
        {kernels.l1Sweep, {0, "l1_sweep", "L1 sweep", ""}},
        {kernels.classes, {0, "classes", "instruction classes", ""}},
        {kernels.mixed, {0, "mixed", "mixed", ""}},
        {kernels.write, {0, "write", "write", ""}},
        {kernels.patternLoads, {0, "pattern_loads", "payload-bits", ""}},
        {kernels.operands, {0, "operands", "operands", ""}},
    };

    for (const auto& [kernel, group] : optional)
    {
        if (kernel.has_value())
        {
            groups.push_back(group);
            groups.back().kernel = kernel.value();
        }
    }

    return groups;
}

// Sleep baselines of a probe run: one before the probes, one after every duty group, one after every group that holds
// all duties but the last, and one after the spin reference (which follows the last group).
size_t SleepCount(const ProbeRunKernels& kernels, const Options& options)
{
    return 1 + options.duties.size() + GroupRuns(kernels).size();
}

// The order of a probe run: a baseline, then for each duty the global (and arithmetic) probes and a baseline, then the
// groups that hold all duties, each followed by a baseline, the spin reference after the last one and a baseline, and
// last the SM sweep (with its own baselines). A baseline after every group runs at about the temperature its probes
// left behind (the GPU cools by only 1-2 C in a 10 s sleep), so the baselines span the probes' temperatures and the
// temperature model of the baseline is interpolated rather than extrapolated.
void MeasureProbes(GpuSession& session, const ProbeRunKernels& kernels, SleepBaselines& sleeps, const Options& options)
{
    sleeps.RunNext();

    for (size_t i = 0; i < kernels.globals.size(); ++i)
    {
        const std::string duty = std::to_string(options.duties[i]);
        session.Run(kernels.globals[i], "global_duty" + duty, "global", "", "");

        if (options.compute)
        {
            session.Run(kernels.computes[i], "compute_duty" + duty, "compute", "", "");
        }

        sleeps.RunNext();
    }

    const std::vector<GroupRun> groups = GroupRuns(kernels);

    for (size_t i = 0; i < groups.size(); ++i)
    {
        session.Run(groups[i].kernel, groups[i].label, groups[i].probe, groups[i].operation, "");

        if (i + 1 < groups.size())
        {
            sleeps.RunNext();
        }
    }

    session.Run(kernels.spin, "spin", "spin", "spin", "reference");
    sleeps.RunNext();

    for (const auto& sweep : kernels.sweep)
    {
        const std::string count = std::to_string(sweep.count);
        session.Run(sweep.sleep, "sleep_sm" + count, "sleep-sm" + count, "sleep", "baseline");

        if (sweep.global.has_value())
        {
            session.Run(sweep.global.value(), "global_sm" + count, "sm-sweep", "", "");
            session.Run(sweep.shared.value(), "shared_sm" + count, "sm-sweep", "ld.shared", "");
        }

        if (sweep.compute.has_value())
        {
            session.Run(sweep.compute.value(), "compute_sm" + count, "sm-sweep", "", "");
        }

        if (sweep.latency.has_value())
        {
            session.Run(sweep.latency.value(), "latency_sm" + count, "sm-sweep", "", "");
        }
    }
}

} // namespace

int RunProbes(const Options& options)
{
    GpuSession session(options);
    const Plan plan = BuildPlan(options, session.GetFootprints());
    DescribeSetup(session.GetHardware(), options, session.GetFootprints(), plan);
    session.OpenMonitor();

    ktt::Tuner& tuner = session.GetTuner();
    std::vector<uint32_t> payload = RandomPayload(session.GetFootprints().dram);
    const ProbeArguments arguments = AddProbeArguments(tuner, payload, options, session.GetBlocks());
    const ProbeDefinitions definitions = AddProbeDefinitions(tuner, arguments, options,
        session.GetHardware().multiprocessors);
    const ProbeTextures textures = CreateProbeTextures(payload, options);
    const ProbeSettings settings = MakeProbeSettings(arguments, definitions, session.GetFootprints(), textures);

    const ktt::KernelId warmup = session.CreateKernel("Warmup", definitions.spin, Configure{}, false, arguments.counts);
    ProbeRunKernels kernels;
    CreateDutyKernels(session, definitions, settings, arguments.counts, plan, options, kernels);
    CreateGroupKernels(session, definitions, settings, arguments.counts, options, kernels);
    CreateSweepKernels(session, definitions, settings, arguments.counts, options, kernels);
    SleepBaselines sleeps(session, definitions.sleep, settings.common, arguments.counts, SleepCount(kernels, options));

    if (options.profile)
    {
        try
        {
            tuner.SetProfiling(true);
            tuner.SetProfilingCounters(ModelCounters(options.counterTable, options.studyCounters));
        }
        catch (const ktt::KttException& error)
        {
            std::cerr << "Profiling could not be enabled (build KTT with --profiling=cupti): " << error.what() << "\n";
            return 1;
        }
    }

    session.MeasureIdle("idle-start");
    session.WarmUp(warmup);
    MeasureProbes(session, kernels, sleeps, options);
    session.MeasureIdle("idle-end");
    session.WriteTrace();

    // Baseline: the sleep runs between the probe groups, modelled over temperature.
    std::vector<Row>& rows = session.GetRows();
    const BaselineModel baseline = FitBaseline(rows);

    if (baseline.runs == 0)
    {
        PrintRows(rows);
        WriteCsv(rows, options.outputPrefix + ".csv");
        std::cout << "\nNo power data was collected. Rebuild KTT with --power-usage and run on a GPU that reports power "
            << "through NVML.\n";
        return 1;
    }

    ReportProbeRun(rows, baseline, session.GetFootprints(), session.GetHardware(), options, session.GetMonitor());

    if (!options.payloadBits.empty())
    {
        std::ofstream statistics(options.outputPrefix + "_payloads.json");
        statistics << arguments.payloadStatistics.dump(1) << "\n";
        std::cout << "Wrote " << options.outputPrefix << "_payloads.json (bit statistics of the payloads; "
            << "tools/fit_payload.py fits them)\n";
    }
    std::cout << "\nWrote " << options.outputPrefix << ".csv, " << options.outputPrefix << "_fits.csv, " << options.outputPrefix
        << "_energy_table.json, " << options.outputPrefix << "_*.json (raw KTT results)"
        << (session.GetMonitor() != nullptr ? " and " + options.outputPrefix + "_trace.csv (NVML trace)" : "") << "\n"
        << "These are prototype numbers: see docs/CALIBRATION.md for what they include and how to validate them.\n";
    return 0;
}

// --thermal-curve: the sleep baseline's power over temperature. A probe run fits the baseline as a line through sleeps that
// each last 10 s and span a few degrees; here one sleep lasts minutes while the GPU cools, traced every --trace-ms, so
// tools/fit_thermal.py fits power against temperature within a single measurement. The GPU is heated twice, by compute
// (the spin kernel, SMs only) and by DRAM loads (16-byte ld at the DRAM footprint, which also heats the memory, whose
// temperature NVML does not report on GeForce cards): if the two cooling curves differ at the same GPU temperature, the
// baseline depends on more than the GPU's temperature. Last, short heating steps each followed by a 10 s sleep, as in a
// probe run but at rising temperatures.
int RunThermalCurve(const Options& options)
{
    GpuSession session(options);
    std::cout << "Thermal curve: sleep 120 s; spin 120 s, sleep 180 s; DRAM loads 120 s, sleep 180 s; 6 x (DRAM loads 20 s, "
        << "sleep 10 s). About 15 min, traced every " << options.traceIntervalMs << " ms.\n";
    session.OpenMonitor();

    ktt::Tuner& tuner = session.GetTuner();
    std::vector<uint32_t> payload = RandomPayload(session.GetFootprints().dram);
    const ProbeArguments arguments = AddProbeArguments(tuner, payload, options, session.GetBlocks());
    const ProbeDefinitions definitions = AddProbeDefinitions(tuner, arguments, options,
        session.GetHardware().multiprocessors);
    const ProbeTextures textures = CreateProbeTextures(payload, options);
    const ProbeSettings settings = MakeProbeSettings(arguments, definitions, session.GetFootprints(), textures);

    // KTT needs every kernel before the first one runs, and the launcher traces only the first run of a kernel and
    // configuration, so every phase has a kernel of its own.
    const Plan dram = {{LoadDefault, session.GetFootprints().dram, 16}};
    auto heater = [&](const std::string& name)
    {
        const ktt::KernelId kernel = session.CreateKernel(name, definitions.global, settings.global, false, arguments.counts);
        AddGlobalParameters(tuner, kernel, dram, session.GetFootprints());
        tuner.AddParameter(kernel, "DUTY", std::vector<uint64_t>{100});
        tuner.AddParameter(kernel, "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
        return kernel;
    };
    auto sleeper = [&](const std::string& name)
    {
        return session.CreateKernel(name, definitions.sleep, settings.common, false, arguments.counts);
    };

    const ktt::KernelId settle = sleeper("SleepSettle");
    const ktt::KernelId spin = session.CreateKernel("HeatSpin", definitions.spin, Configure{}, false, arguments.counts);
    const ktt::KernelId coolSpin = sleeper("CoolAfterSpin");
    const ktt::KernelId heatDram = heater("HeatDram");
    const ktt::KernelId coolDram = sleeper("CoolAfterDram");
    std::vector<std::pair<ktt::KernelId, ktt::KernelId>> steps;

    for (int i = 0; i < 6; ++i)
    {
        steps.emplace_back(heater("StepHeat" + std::to_string(i)), sleeper("StepSleep" + std::to_string(i)));
    }

    session.MeasureIdle("idle-start");
    session.RunFor(settle, 120, "sleep_settle", "sleep-settle", "sleep", "baseline");
    session.RunFor(spin, 120, "heat_spin", "heat-spin", "spin", "reference");
    session.RunFor(coolSpin, 180, "cool_after_spin", "cool-spin", "sleep", "baseline");
    session.RunFor(heatDram, 120, "heat_dram", "heat-dram", "", "");
    session.RunFor(coolDram, 180, "cool_after_dram", "cool-dram", "sleep", "baseline");

    for (size_t i = 0; i < steps.size(); ++i)
    {
        session.RunFor(steps[i].first, 20, "step_heat" + std::to_string(i), "step-heat-" + std::to_string(i), "", "");
        session.RunFor(steps[i].second, 10, "step_sleep" + std::to_string(i), "step-sleep-" + std::to_string(i), "sleep",
            "baseline");
    }

    session.MeasureIdle("idle-end");
    session.WriteTrace();
    WriteCsv(session.GetRows(), options.outputPrefix + ".csv");
    std::cout << "\nWrote " << options.outputPrefix << ".csv and " << options.outputPrefix << "_trace.csv; fit them with "
        << "tools/fit_thermal.py " << options.outputPrefix << "\n";
    return 0;
}


// NEXT_STEPS_PLAN M1: does the power of a working GPU rise with the temperature faster than the sleep baseline does, and
// does that depend on the duty or on the dynamic power? Five probes that span the power range (the loop alone, FFMA with
// three registers, 16-byte L2 loads, DRAM loads, DRAM stores) at every --duty, and the sleep baseline, each held
// --thermal-hold seconds twice: once after 60 s of sleep (starting cool) and once after 60 s of DRAM stores at duty 100
// (starting hot), so that the temperature runs through the same range up and down and a drift in time can be told from
// the temperature. Every phase is a kernel of its own (the trace names the kernel); tools/fit_thermal_load.py fits it.
int RunThermalLoad(const Options& options)
{
    GpuSession session(options);
    session.OpenMonitor();

    ktt::Tuner& tuner = session.GetTuner();
    std::vector<uint32_t> payload = RandomPayload(session.GetFootprints().dram);
    const ProbeArguments arguments = AddProbeArguments(tuner, payload, options, session.GetBlocks());
    const ProbeDefinitions definitions = AddProbeDefinitions(tuner, arguments, options,
        session.GetHardware().multiprocessors);
    const ProbeTextures textures = CreateProbeTextures(payload, options);
    const ProbeSettings settings = MakeProbeSettings(arguments, definitions, session.GetFootprints(), textures);
    const Footprints& footprints = session.GetFootprints();

    struct Probe
    {
        std::string name;
        bool compute;
        uint64_t op;
        uint64_t footprint;
    };

    const std::vector<Probe> probes = {{"loop", true, ComputeLoop, 0}, {"ffma3", true, ComputeFfma3, 0},
        {"L2ld", false, LoadDefault, footprints.l2}, {"DRAMld", false, LoadDefault, footprints.dram},
        {"DRAMst", false, StoreDefault, footprints.dram}};

    auto global = [&](const std::string& name, const uint64_t op, const uint64_t footprint, const uint64_t duty)
    {
        const ktt::KernelId kernel = session.CreateKernel(name, definitions.global, settings.global, true, arguments.counts);
        AddGlobalParameters(tuner, kernel, Plan{{op, footprint, 16}}, footprints);
        tuner.AddParameter(kernel, "DUTY", std::vector<uint64_t>{duty});
        tuner.AddParameter(kernel, "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
        return kernel;
    };
    auto compute = [&](const std::string& name, const uint64_t op, const uint64_t duty)
    {
        const ktt::KernelId kernel = session.CreateKernel(name, definitions.compute, settings.compute, false, arguments.counts);
        tuner.AddParameter(kernel, "COMPUTE_OP", std::vector<uint64_t>{op});
        tuner.AddParameter(kernel, "DUTY", std::vector<uint64_t>{duty});
        tuner.AddParameter(kernel, "PAYLOAD", std::vector<uint64_t>{PayloadRandom});
        return kernel;
    };
    auto sleeper = [&](const std::string& name)
    {
        return session.CreateKernel(name, definitions.sleep, settings.common, false, arguments.counts);
    };

    // KTT needs every kernel before the first one runs.
    struct Phase
    {
        ktt::KernelId preparation;
        ktt::KernelId hold;
        std::string name;
        std::string operation;
        std::string level;
    };

    std::vector<Phase> phases;

    for (const char* start : {"cool", "hot"})
    {
        const bool hot = std::string(start) == "hot";
        auto preparation = [&](const std::string& name)
        {
            return hot ? global("Prep_" + name, StoreDefault, footprints.dram, 100) : sleeper("Prep_" + name);
        };
        const std::string sleepName = std::string("Hold_sleep_") + start;
        phases.push_back({preparation(sleepName), sleeper(sleepName), sleepName, "sleep", "baseline"});

        for (const auto& probe : probes)
        {
            for (const uint64_t duty : options.duties)
            {
                const std::string name = "Hold_" + probe.name + "_d" + std::to_string(duty) + "_" + start;
                const ktt::KernelId hold = probe.compute ? compute(name, probe.op, duty)
                    : global(name, probe.op, probe.footprint, duty);
                phases.push_back({preparation(name), hold, name, probe.name, "d" + std::to_string(duty)});
            }
        }
    }

    // Cool starts and hot starts alternate, so that a slow drift over the run is not mistaken for the start.
    std::vector<Phase> order;
    const size_t half = phases.size() / 2;

    for (size_t i = 0; i < half; ++i)
    {
        order.push_back(phases[i]);
        order.push_back(phases[half + i]);
    }

    std::cout << "Thermal load: " << order.size() << " phases of 60 s preparation + " << options.thermalHoldS
        << " s hold, traced every " << options.traceIntervalMs << " ms; about "
        << (order.size() * (60 + options.thermalHoldS) + 120) / 60 << " min.\n";
    session.MeasureIdle("idle-start");

    for (const auto& phase : order)
    {
        session.RunFor(phase.preparation, 60, "prep_" + phase.name, "prep", "", "");
        session.RunFor(phase.hold, options.thermalHoldS, phase.name, phase.name, phase.operation, phase.level);
    }

    session.MeasureIdle("idle-end");
    session.WriteTrace();
    WriteCsv(session.GetRows(), options.outputPrefix + ".csv");
    std::cout << "\nWrote " << options.outputPrefix << ".csv and " << options.outputPrefix << "_trace.csv; fit them with "
        << "tools/fit_thermal_load.py " << options.outputPrefix << "\n";
    return 0;
}
