#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

#include <cuda.h>

#include "CounterTable.h"
#include "ProbeKernels.h"

namespace
{

// Checks the SM ids and slots that a probe wrote for the last launch. Slots on an SM are only distinct while it gets
// at most BLOCKS_PER_SM blocks; otherwise blocks on that SM share a stripe and can run in each other's cached lines.
std::string CheckPlacement(const std::vector<uint64_t>& counts, const uint64_t blocks, const uint64_t blocksPerSm)
{
    std::map<uint32_t, uint64_t> blocksOnSm;
    std::map<uint32_t, std::set<uint32_t>> slotsOnSm;

    for (uint64_t i = 0; i < blocks; ++i)
    {
        const uint64_t entry = counts[2 * blocks + i];
        const uint32_t sm = static_cast<uint32_t>(entry >> 32);
        ++blocksOnSm[sm];
        slotsOnSm[sm].insert(static_cast<uint32_t>(entry));
    }

    size_t uneven = 0;
    size_t sharedStripes = 0;

    for (const auto& [sm, count] : blocksOnSm)
    {
        uneven += count != blocksPerSm ? 1 : 0;
        sharedStripes += slotsOnSm[sm].size() != count ? 1 : 0;
    }

    if (uneven == 0 && sharedStripes == 0)
    {
        return "ok";
    }

    return std::to_string(uneven) + " of " + std::to_string(blocksOnSm.size()) + " SMs without " + std::to_string(blocksPerSm)
        + " blocks and " + std::to_string(sharedStripes) + " with blocks sharing a stripe";
}

void CheckCuda(const CUresult result, const std::string& call)
{
    if (result != CUDA_SUCCESS)
    {
        const char* text = nullptr;
        cuGetErrorString(result, &text);
        throw std::runtime_error(call + " failed: " + (text != nullptr ? std::string(text) : std::to_string(result)));
    }
}

// Sleep time per burst time for a duty, (100 - duty) / duty in 16.16 fixed point (argument idleRatio of the kernels).
uint32_t IdleRatio(const uint64_t duty)
{
    return static_cast<uint32_t>(((100 - duty) * 65536 + duty / 2) / duty);
}

std::vector<uint32_t> RandomPayload(const uint64_t bytes)
{
    std::vector<uint32_t> payload(bytes / sizeof(uint32_t));
    std::mt19937_64 generator(12345);

    for (size_t i = 0; i + 1 < payload.size(); i += 2)
    {
        const uint64_t value = generator();
        payload[i] = static_cast<uint32_t>(value);
        payload[i + 1] = static_cast<uint32_t>(value >> 32);
    }

    return payload;
}

// The operands of the arithmetic recurrences (a, c): x -> x * x + c with c = -1.9 for floating point (half2: -1.9
// twice), the LCG x -> x * a + c for integers, x -> a * rsqrt(|x|) + c for MUFU. They stay bounded but keep changing, so
// the datapath toggles as with real data.
std::pair<uint32_t, uint32_t> ComputeOperands(const uint64_t operation)
{
    switch (operation)
    {
    case ComputeImad: return {1664525u, 1013904223u};
    case ComputeHfma2: return {0u, 0xbf9abf9au};
    case ComputeMufu: return {FloatBits(-1.0f), FloatBits(1.5f)};
    case ComputeFfma: case ComputeDfma: case ComputeDynamicIndex: case ComputeFfma3: case ComputeFfmaLoop:
        return {0u, FloatBits(-1.9f)};
    default: return {0u, 0u};
    }
}

} // namespace

std::optional<uint64_t> ParameterValue(const ktt::KernelConfiguration& configuration, const std::string& name)
{
    for (const auto& pair : configuration.GetPairs())
    {
        if (pair.GetName() == name)
        {
            return ktt::ParameterPair::GetParameterValue<uint64_t>(configuration.GetPairs(), name);
        }
    }

    return std::nullopt;
}

std::string TraceLabel(const std::string& kernelName, const ktt::KernelConfiguration& configuration)
{
    static const std::set<std::string> common = {"BLOCK_SIZE", "BLOCKS_PER_SM", "UNROLL", "BURST_ACCESSES", "GRID_STRIPES"};
    std::string label = kernelName;

    for (const auto& pair : configuration.GetPairs())
    {
        if (common.count(pair.GetName()) == 0)
        {
            label += " " + pair.GetName() + "=" + pair.GetValueString();
        }
    }

    return label;
}

std::string KernelSource(const std::string& directory, const std::string& file)
{
    auto read = [&directory](const std::string& name)
    {
        std::ifstream input(directory + name);

        if (!input)
        {
            throw std::runtime_error("Unable to open kernel file " + directory + name + " (see --kernel-dir)");
        }

        std::stringstream text;
        text << input.rdbuf();
        return text.str();
    };

    return "#line 1 \"Common.cuh\"\n" + read("Common.cuh") + "\n#line 1 \"" + file + "\"\n" + read(file);
}

BaselineKernels AddBaselineKernels(ktt::Tuner& tuner, const Options& options, const uint64_t blocks,
    const uint32_t multiprocessors)
{
    const std::string source = KernelSource(options.kernelDirectory, "Baseline.cu");
    const ktt::DimensionVector grid(multiprocessors); // multiplied by the launch parameters' thread modifiers
    const ktt::DimensionVector block(1);

    BaselineKernels kernels;
    kernels.sink = tuner.AddArgumentVector(std::vector<uint32_t>(blocks * options.blockSize, 0),
        ktt::ArgumentAccessType::WriteOnly);
    kernels.counts = tuner.AddArgumentVector(std::vector<uint64_t>(3 * blocks + 1, 0), ktt::ArgumentAccessType::ReadWrite);
    kernels.activeSms = tuner.AddArgumentScalar(AllSms);
    kernels.duration = tuner.AddArgumentScalar(static_cast<uint64_t>(options.launchMs * 1'000'000));
    kernels.sleep = tuner.AddKernelDefinition("probeSleep", source, grid, block);
    kernels.spin = tuner.AddKernelDefinition("probeSpin", source, grid, block);
    tuner.SetArguments(kernels.sleep, {kernels.sink, kernels.counts, kernels.activeSms, kernels.duration});
    tuner.SetArguments(kernels.spin, {kernels.sink, kernels.counts, kernels.duration});
    return kernels;
}

ktt::KernelId CreateProbeKernel(ktt::Tuner& tuner, const std::string& name, const ktt::KernelDefinitionId definition,
    const Options& options, const ktt::ArgumentId countsId, const uint64_t blocks, const Configure& configure,
    const bool checkPlacement, StatsMap& stats, NvmlMonitor* monitor)
{
    const ktt::KernelId kernel = tuner.CreateSimpleKernel(name, definition);
    tuner.AddParameter(kernel, "BLOCK_SIZE", std::vector<uint64_t>{options.blockSize});
    tuner.AddParameter(kernel, "BLOCKS_PER_SM", std::vector<uint64_t>{options.blocksPerSm});
    tuner.AddParameter(kernel, "UNROLL", std::vector<uint64_t>{options.unroll});
    tuner.AddParameter(kernel, "BURST_ACCESSES", std::vector<uint64_t>{options.burstAccesses});
    // Definitions are added with grid = number of SMs and block = 1 (GlobalSizeType::CUDA).
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK_SIZE",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "BLOCKS_PER_SM",
        ktt::ModifierAction::Multiply);

    if (options.randomOrder)
    {
        tuner.SetSearcher(kernel, std::make_unique<ktt::RandomSearcher>());
    }
    else
    {
        tuner.SetSearcher(kernel, std::make_unique<ktt::DeterministicSearcher>());
    }

    const uint64_t traceIntervalMs = options.traceIntervalMs;
    const uint64_t blocksPerSm = options.blocksPerSm;
    const double minTimeS = static_cast<double>(options.minTimeMs) / 1000.0;
    const double maxTimeS = static_cast<double>(options.maxTimeMs) / 1000.0;

    tuner.SetLauncher(kernel, [&stats, name, definition, countsId, blocks, configure, checkPlacement, blocksPerSm, monitor,
        traceIntervalMs, minTimeS, maxTimeS](ktt::ComputeInterface& interface)
    {
        const auto& configuration = interface.GetCurrentConfiguration();
        const std::string key = name + "|" + configuration.GetString();

        if (configure)
        {
            configure(interface, configuration);
        }

        // The kernels add their counts to the buffer on every launch (and count the launches), so it is cleared once for
        // all launches of this configuration.
        std::vector<uint64_t> counts(3 * blocks + 1, 0);
        interface.UpdateBuffer(countsId, counts.data(), counts.size() * sizeof(uint64_t));

        // Profiling replays the kernel. Only the first pass carries power data, so only that one is traced and kept.
        const bool firstPass = stats.count(key) == 0;

        if (firstPass && monitor != nullptr)
        {
            monitor->Start(TraceLabel(name, configuration), traceIntervalMs);
        }

        const auto begin = std::chrono::steady_clock::now();

        if (interface.GetProfiling(definition))
        {
            interface.RunKernelWithProfiling(definition);
        }
        else
        {
            interface.RunKernel(definition);
        }

        LaunchStats launch;
        launch.measureS = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

        if (firstPass && monitor != nullptr)
        {
            // KTT judges the power over its samples of about the last --min-time-ms, after sliding past the start of the
            // measurement, where NVML's 1 s average still holds the previous configuration's power. The trace starts a
            // little earlier than KTT's timer, so its last 90 % of --min-time-ms matches KTT's window.
            launch.trace = monitor->Stop(0.9 * minTimeS);
        }

        // KTT repeats the kernel until its power samples of the last --min-time-ms stay within --max-power-diff of their
        // mean, or until --max-time-ms, when it only warns on the console: such a measurement ran out of time.
        if (firstPass && minTimeS > 0.0 && maxTimeS > minTimeS)
        {
            launch.powerSettled = launch.measureS < maxTimeS ? 1 : 0;
        }

        // Sums over all launches of the configuration, so that accesses per launch match the power and duration, which
        // KTT averages over the same launches.
        interface.DownloadBuffer(countsId, counts.data(), counts.size() * sizeof(uint64_t));
        launch.launches = counts[3 * blocks];
        const double launches = static_cast<double>(std::max<uint64_t>(launch.launches, 1));
        uint64_t accesses = 0;
        uint64_t workingBlocks = 0;
        double activeNs = 0.0;

        for (uint64_t i = 0; i < blocks; ++i)
        {
            accesses += counts[i];
            activeNs += static_cast<double>(counts[blocks + i]);
            workingBlocks += counts[i] > 0 ? 1 : 0;
        }

        launch.accesses = static_cast<uint64_t>(std::llround(static_cast<double>(accesses) / launches));
        launch.meanActiveSeconds = workingBlocks > 0 ? activeNs / static_cast<double>(workingBlocks) / launches * 1e-9 : 0.0;

        if (checkPlacement)
        {
            launch.placement = CheckPlacement(counts, blocks, blocksPerSm);
        }

        stats.emplace(key, launch);
    });

    return kernel;
}

Row MakeRow(const ktt::KernelResult& result, const std::string& probe, const std::string& operation,
    const std::string& level, const StatsMap& stats, const Hardware& hardware, const Footprints& footprints)
{
    Row row;
    row.probe = probe;
    row.operation = operation;
    row.level = level;
    row.valid = result.IsValid();

    const auto& configuration = result.GetConfiguration();
    const auto parameter = [&configuration](const std::string& name) { return ParameterValue(configuration, name); };
    const auto loadOp = parameter("LOAD_OP");

    // Every variant gets an operation name of its own, so that its rows are fitted together and with no other probe's.
    if (loadOp.has_value())
    {
        row.operation = OperationName(loadOp.value());
    }

    if (const auto run = parameter("RUN_BYTES"))
    {
        row.operation += run.value() == 0 ? "-stream" : "-run" + std::to_string(run.value());
    }

    if (const auto offset = parameter("OFFSET_BYTES"))
    {
        row.operation += "-off" + std::to_string(offset.value());
    }

    if (const auto spread = parameter("SPREAD"))
    {
        row.operation += spread.value() == 2 ? "-sub-pairs" : spread.value() != 0 ? "-sub-scattered" : "-sub-adjacent";
    }

    if (const auto words = parameter("PARTIAL_WORDS"))
    {
        row.operation += "-" + std::to_string(words.value()) + "words";
    }

    if (const auto footprint = parameter("FOOTPRINT"))
    {
        row.footprint = footprint.value();
        row.width = parameter("WIDTH").value_or(DefaultWidth);
        row.level = ExpectedLevel(loadOp.value_or(LoadDefault), row.footprint, hardware, footprints);
    }
    else if (const auto sharedBytes = parameter("SMEM_BYTES"))
    {
        const uint64_t sharedOp = parameter("SHARED_OP").value_or(SharedLoad);
        row.footprint = sharedBytes.value();
        row.width = sharedOp == SharedRed ? AtomicWidth : parameter("SHARED_WIDTH").value_or(DefaultWidth);
        row.operation = sharedOp == SharedRed ? "red.shared" : (sharedOp == SharedStore ? "st.shared" : "ld.shared");
        row.level = "shared";

        if (const auto ways = parameter("BANK_WAYS"))
        {
            row.operation += "-" + std::to_string(ways.value()) + "way";
        }
    }
    else if (const auto constantBytes = parameter("CONST_BYTES"))
    {
        // A warp reads one word per access, so bytes per thread access mean nothing here: no width, energies per access.
        row.footprint = constantBytes.value();
        row.operation = "ld.const";
        row.level = ExpectedConstantLevel(row.footprint, hardware, footprints);
    }
    else if (const auto computeOp = parameter("COMPUTE_OP"))
    {
        row.operation = ComputeName(computeOp.value());
        row.level = "compute";
    }

    if (const auto chains = parameter("CHAINS"))
    {
        row.operation = "chase-c" + std::to_string(chains.value()) + "-w"
            + std::to_string(parameter("ACTIVE_WARPS").value_or(1) * parameter("BLOCKS_PER_SM").value_or(1));
    }

    // Local memory is private to a thread and its footprint is per SM: at L1 up to the L1 footprint and, for all SMs
    // together far beyond L2, at DRAM above.
    if (const auto write = parameter("WRITE_OP"))
    {
        row.operation = write.value() == WriteCopy ? "copy" : write.value() == WriteUpdate ? "update" : "local";

        if (write.value() == WriteLocal)
        {
            row.level = row.footprint <= footprints.l1 ? "L1" : "DRAM";
        }
    }

    row.duty = parameter("DUTY").value_or(100);
    const auto statsIterator = stats.find(result.GetKernelName() + "|" + configuration.GetString());

    if (statsIterator != stats.cend())
    {
        const LaunchStats& launch = statsIterator->second;
        row.accesses = launch.accesses;
        row.launches = launch.launches;
        row.activeS = launch.meanActiveSeconds;
        row.measureS = launch.measureS;
        row.powerSettled = launch.powerSettled;

        if (!launch.placement.empty() && launch.placement != "ok")
        {
            AppendNote(row.note, "placement: " + launch.placement);
        }

        if (launch.trace.has_value())
        {
            row.counterPowerW = launch.trace->counterPowerW;
            row.tracePowerW = launch.trace->sampledPowerW;
            row.instantPowerW = launch.trace->instantPowerW;
            row.averagePowerW = launch.trace->averagePowerW;
            row.minClockMHz = launch.trace->minClockMHz;
            row.maxClockMHz = launch.trace->maxClockMHz;
            row.fanPct = launch.trace->meanFanPct;
            row.powerSpreadPct = launch.trace->tailSpreadPct;
            row.limiters = DescribeReasons(launch.trace->reasons, true);
        }
    }

    if (!row.valid || result.GetResults().empty())
    {
        AppendNote(row.note, "run failed");
        return row;
    }

    const auto& computation = result.GetResults()[0];
    row.durationS = static_cast<double>(computation.GetDuration()) * 1e-9;

    if (computation.HasCompilationData())
    {
        row.registers = computation.GetCompilationData().m_RegistersCount;
    }

    if (computation.HasPowerData())
    {
        row.hasPower = true;
        row.powerW = static_cast<double>(computation.GetPowerUsage()) / 1000.0;
        row.energyJ = row.powerW * row.durationS;
    }

    if (computation.HasSMFrequencyData())
    {
        row.smClockMHz = computation.GetSMFrequency();
    }

    if (computation.HasMemoryFrequencyData())
    {
        row.memoryClockMHz = computation.GetMemoryFrequency();
    }

    if (computation.HasTemperatureData())
    {
        row.temperatureC = computation.GetTemperature();
    }

    if (row.accesses > 0 && row.width > 0 && IsProbeRow(row))
    {
        row.bandwidthGBs = static_cast<double>(row.accesses * row.width) / row.durationS * 1e-9;
    }

    if (computation.HasProfilingData())
    {
        for (const auto& counter : computation.GetProfilingData().GetCounters())
        {
            row.counters[CanonicalCounter(counter.GetName())] = CounterValue(counter);
        }
    }

    return row;
}

std::vector<ktt::KernelResult> RunProbe(ktt::Tuner& tuner, const ktt::KernelId kernel, const std::string& label,
    const std::string& outputPrefix, const ktt::PreciseMeasurementParameters& precise)
{
    std::cout << "\n=== Probe " << label << " ===" << std::endl;
    const auto results = tuner.Tune(kernel, nullptr, precise);
    tuner.SaveResults(results, outputPrefix + "_" + label, ktt::OutputFormat::JSON);
    return results;
}

ProbeSet::ProbeSet(ktt::Tuner& tuner, const Options& options, const Footprints& footprints, const BaselineKernels& baseline,
    const uint32_t multiprocessors) :
    m_Payload(RandomPayload(footprints.dram))
{
    const size_t payloadBytes = m_Payload.size() * sizeof(uint32_t);
    const auto scalar = [&tuner](const uint32_t value) { return tuner.AddArgumentScalar(value); };

    // The stores and atomics overwrite the payload with words derived from it.
    const ktt::ArgumentId data = tuner.AddArgumentVector(m_Payload, ktt::ArgumentAccessType::ReadWrite,
        ktt::ArgumentMemoryLocation::Device, ktt::ArgumentManagementType::Framework, true);
    const ktt::ArgumentId smCounters = tuner.AddArgumentVector(std::vector<uint32_t>(SmCounterEntries, 0),
        ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId sink = baseline.sink;
    const ktt::ArgumentId counts = baseline.counts;
    const ktt::ArgumentId activeSms = baseline.activeSms;
    const ktt::ArgumentId duration = baseline.duration;
    // Set per configuration by the launchers. The kernel files describe what each one means.
    const ktt::ArgumentId gridSplit = scalar(0);
    const ktt::ArgumentId idleRatio = scalar(0);
    const ktt::ArgumentId payloadMask = scalar(0xffffffffu); // 0 would make the shared and arithmetic probes run on zeros
    const ktt::ArgumentId operandA = scalar(0);
    const ktt::ArgumentId operandC = scalar(0);
    const ktt::ArgumentId constantBytes = scalar(0);
    const ktt::ArgumentId sectorMask = scalar(0);
    const ktt::ArgumentId runBits = scalar(0);
    const ktt::ArgumentId stride = scalar(1);
    const ktt::ArgumentId bankWays = scalar(1);
    const ktt::ArgumentId activeWarps = scalar(0);
    const ktt::ArgumentId lineMask = scalar(0);
    const ktt::ArgumentId zero = scalar(0); // probeLatency: the dependency of an address on the previous load's data

    // probeConstant reads a __constant__ array, which KTT fills from the payload when it loads the module.
    auto constantPayload = std::make_unique<std::array<uint32_t, ConstantWords>>();
    std::copy_n(m_Payload.cbegin(), ConstantWords, constantPayload->begin());
    const ktt::ArgumentId constantData = tuner.AddArgumentSymbol(*constantPayload, "", "constantData");

    // The misaligned probes read the payload from a start moved off the 128-byte line. KTT does not expose the device
    // pointers of its buffers, so the payload is copied to a buffer allocated with the driver API in KTT's context (the
    // tuner creates it with cuCtxCreate, which leaves it current on this thread), with one KTT argument per start offset
    // over it. The probe's SASS stays the same: only the pointer it gets differs. cuMemAlloc aligns to 256 bytes at least.
    CUdeviceptr buffer = 0;
    CheckCuda(cuMemAlloc(&buffer, payloadBytes + LineBytes), "cuMemAlloc");
    m_OffsetBuffer = static_cast<uint64_t>(buffer);
    CheckCuda(cuMemcpyHtoD(buffer, m_Payload.data(), payloadBytes), "cuMemcpyHtoD");
    CheckCuda(cuMemcpyHtoD(buffer + payloadBytes, m_Payload.data(), LineBytes), "cuMemcpyHtoD");
    std::map<uint64_t, ktt::ArgumentId> offsetData;

    // Multiples of 16 only: the store probe seeds its values with a 16-byte read at the start of its data.
    for (uint64_t offset = 16; offset < LineBytes; offset += 16)
    {
        offsetData[offset] = tuner.AddArgumentVector(reinterpret_cast<ktt::ComputeBuffer>(buffer + offset), payloadBytes,
            sizeof(uint32_t), ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
    }

    const ktt::DimensionVector grid(multiprocessors);
    const ktt::DimensionVector block(1);
    const auto add = [&](const ProbeKernel kernel, const std::string& function, const std::string& file,
        const std::vector<ktt::ArgumentId>& arguments)
    {
        m_Definitions[kernel] = tuner.AddKernelDefinition(function, KernelSource(options.kernelDirectory, file), grid, block);
        tuner.SetArguments(m_Definitions[kernel], arguments);
    };

    const std::vector<ktt::ArgumentId> globalArguments = {data, sink, counts, smCounters, gridSplit, activeSms, idleRatio,
        duration};
    add(ProbeKernel::Global, "probeGlobal", "Global.cu", globalArguments);
    add(ProbeKernel::Compute, "probeCompute", "Compute.cu", {sink, counts, activeSms, idleRatio, payloadMask, operandA,
        operandC, duration});
    add(ProbeKernel::Shared, "probeShared", "Shared.cu", {sink, counts, activeSms, idleRatio, payloadMask, duration});
    add(ProbeKernel::Banks, "probeSharedBanks", "Shared.cu", {sink, counts, bankWays, activeSms, idleRatio, payloadMask,
        duration});
    add(ProbeKernel::Constant, "probeConstant", "Constant.cu", {sink, counts, smCounters, constantBytes, activeSms,
        idleRatio, duration, constantData});
    add(ProbeKernel::Scatter, "probeScatter", "Scatter.cu", {data, sink, counts, sectorMask, runBits, stride, activeSms,
        idleRatio, duration});
    add(ProbeKernel::Partial, "probePartial", "Partial.cu", {data, sink, counts, sectorMask, stride, activeSms, idleRatio,
        duration});
    add(ProbeKernel::Write, "probeWrite", "Write.cu", globalArguments);
    add(ProbeKernel::Latency, "probeLatency", "Latency.cu", {data, sink, counts, activeSms, activeWarps, lineMask, zero,
        idleRatio, duration});

    // Duty and the other variants that must not change the SASS are kernel arguments: every setting first sets the
    // duty, then its own arguments.
    using Interface = ktt::ComputeInterface&;
    using Configuration = const ktt::KernelConfiguration&;
    const auto get = [](Configuration configuration, const std::string& name)
    {
        return ParameterValue(configuration, name).value_or(0);
    };
    const auto set = [](Interface interface, const ktt::ArgumentId argument, const uint32_t value)
    {
        interface.UpdateScalarArgument(argument, &value);
    };
    const Configure duty = [=](Interface interface, Configuration configuration)
    {
        set(interface, idleRatio, IdleRatio(ParameterValue(configuration, "DUTY").value_or(100)));
    };
    // Footprints from disjointFrom on are split over all blocks of the grid (kernels/Global.cu).
    const uint64_t disjointFrom = footprints.disjointFrom;
    const Configure split = [=](Interface interface, Configuration configuration)
    {
        duty(interface, configuration);
        set(interface, gridSplit, get(configuration, "FOOTPRINT") >= disjointFrom ? 1 : 0);
    };

    m_Settings[ProbeKernel::Shared] = duty;
    m_Settings[ProbeKernel::Write] = split;
    const ktt::KernelDefinitionId global = m_Definitions[ProbeKernel::Global];
    m_Settings[ProbeKernel::Global] = [=](Interface interface, Configuration configuration)
    {
        split(interface, configuration);
        ktt::ArgumentId source = data;

        // KTT creates the buffers of the arguments a definition had when the run started, so a view is uploaded on its
        // first use.
        if (const auto offset = ParameterValue(configuration, "OFFSET_BYTES"))
        {
            source = offsetData.at(offset.value());

            if (!interface.HasBuffer(source))
            {
                interface.UploadBuffer(source);
            }
        }

        std::vector<ktt::ArgumentId> arguments = globalArguments;
        arguments.front() = source;
        interface.ChangeArguments(global, arguments);
    };
    m_Settings[ProbeKernel::Compute] = [=](Interface interface, Configuration configuration)
    {
        duty(interface, configuration);
        const auto [a, c] = ComputeOperands(get(configuration, "COMPUTE_OP"));
        set(interface, operandA, a);
        set(interface, operandC, c);
    };
    // The conflict degree is an argument, so that every degree runs the same SASS.
    m_Settings[ProbeKernel::Banks] = [=](Interface interface, Configuration configuration)
    {
        duty(interface, configuration);
        set(interface, bankWays, static_cast<uint32_t>(get(configuration, "BANK_WAYS")));
    };
    m_Settings[ProbeKernel::Constant] = [=](Interface interface, Configuration configuration)
    {
        duty(interface, configuration);
        set(interface, constantBytes, static_cast<uint32_t>(get(configuration, "CONST_BYTES")));
    };
    // Runs of RUN_BYTES consecutive sectors, placed apart by ScatterStride (RUN_BYTES 0: one stream, stride 1).
    m_Settings[ProbeKernel::Scatter] = [=](Interface interface, Configuration configuration)
    {
        duty(interface, configuration);
        const uint64_t run = get(configuration, "RUN_BYTES");
        set(interface, sectorMask, static_cast<uint32_t>(get(configuration, "FOOTPRINT") / SectorBytes - 1));
        set(interface, runBits, run == 0 ? 0 : static_cast<uint32_t>(std::log2(static_cast<double>(run / SectorBytes))));
        set(interface, stride, run == 0 ? 1u : ScatterStride);
    };
    // SPREAD 0: the sectors of a warp next to each other (4 per L2 request), 2: every other sector (2 per request),
    // 1: scattered by ScatterStride (1 per request).
    m_Settings[ProbeKernel::Partial] = [=](Interface interface, Configuration configuration)
    {
        duty(interface, configuration);
        const uint64_t spread = get(configuration, "SPREAD");
        set(interface, sectorMask, static_cast<uint32_t>(get(configuration, "FOOTPRINT") / SectorBytes - 1));
        set(interface, stride, spread == 2 ? 2u : spread != 0 ? ScatterStride : 1u);
    };
    // The working warps per block and the footprint's lines - 1.
    m_Settings[ProbeKernel::Latency] = [=](Interface interface, Configuration configuration)
    {
        duty(interface, configuration);
        set(interface, activeWarps, static_cast<uint32_t>(ParameterValue(configuration, "ACTIVE_WARPS").value_or(1)));
        set(interface, lineMask, static_cast<uint32_t>(get(configuration, "FOOTPRINT") / LineBytes - 1));
    };
}

ProbeSet::~ProbeSet()
{
    if (m_OffsetBuffer != 0)
    {
        cuMemFree(static_cast<CUdeviceptr>(m_OffsetBuffer));
    }
}

ktt::KernelDefinitionId ProbeSet::Definition(const ProbeKernel kernel) const
{
    return m_Definitions.at(kernel);
}

Configure ProbeSet::Settings(const ProbeKernel kernel) const
{
    return m_Settings.at(kernel);
}
