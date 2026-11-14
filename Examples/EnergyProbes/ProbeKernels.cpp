#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "CounterTable.h"
#include "Payloads.h"
#include "ProbeKernels.h"

namespace
{

// Checks the SM ids and slots that probeGlobal wrote for the last launch. Slots on an SM are only distinct while it gets
// at most BLOCKS_PER_SM blocks; otherwise blocks on that SM share a stripe and can convoy again.
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

} // namespace

double CounterValue(const ktt::KernelProfilingCounter& counter)
{
    switch (counter.GetType())
    {
    case ktt::ProfilingCounterType::Int:
        return static_cast<double>(counter.GetValueInt());
    case ktt::ProfilingCounterType::Double:
    case ktt::ProfilingCounterType::Percent:
        return counter.GetValueDouble();
    default:
        return static_cast<double>(counter.GetValueUint());
    }
}

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
    std::string label = kernelName;

    for (const auto& pair : configuration.GetPairs())
    {
        const std::string& name = pair.GetName();

        if (name != "BLOCK_SIZE" && name != "BLOCKS_PER_SM" && name != "UNROLL" && name != "BURST_ACCESSES"
            && name != "GRID_STRIPES")
        {
            label += " " + name + "=" + pair.GetValueString();
        }
    }

    return label;
}

uint32_t IdleRatio(const uint64_t duty)
{
    return static_cast<uint32_t>(((100 - duty) * 65536 + duty / 2) / duty);
}

uint32_t ActiveSms(const ktt::KernelConfiguration& configuration)
{
    return static_cast<uint32_t>(ParameterValue(configuration, "ACTIVE_SMS").value_or(AllSms));
}

void AddProbeLaunchParameters(ktt::Tuner& tuner, const ktt::KernelId kernel, const ktt::KernelDefinitionId definition,
    const Options& options)
{
    tuner.AddParameter(kernel, "BLOCK_SIZE", std::vector<uint64_t>{options.blockSize});
    tuner.AddParameter(kernel, "BLOCKS_PER_SM", std::vector<uint64_t>{options.blocksPerSm});
    tuner.AddParameter(kernel, "UNROLL", std::vector<uint64_t>{options.unroll});
    tuner.AddParameter(kernel, "BURST_ACCESSES", std::vector<uint64_t>{options.burstAccesses});

    // Definitions are added with grid = number of SMs and block = 1 (GlobalSizeType::CUDA).
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "BLOCK_SIZE",
        ktt::ModifierAction::Multiply);
    tuner.AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "BLOCKS_PER_SM",
        ktt::ModifierAction::Multiply);
}

ktt::KernelId CreateProbeKernel(ktt::Tuner& tuner, const std::string& name, const ktt::KernelDefinitionId definition,
    const Options& options, const ktt::ArgumentId& countsId, const uint64_t blocks, const Configure& configure,
    const bool checkPlacement, StatsMap& stats, NvmlMonitor* monitor)
{
    const ktt::KernelId kernel = tuner.CreateSimpleKernel(name, definition);
    AddProbeLaunchParameters(tuner, kernel, definition, options);

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
            // measurement, where the 1 s average still holds the previous configuration's power. The trace starts a little
            // earlier than KTT's timer, so its last 90 % of --min-time-ms matches KTT's window: on an RTX 3090 the rows KTT
            // found settled (within 2 %) spread by 0.04-2.05 % there, while the last 100 % reached into the start (13-21 %).
            launch.trace = monitor->Stop(0.9 * minTimeS);
        }

        // KTT repeats the kernel until its power samples of the last --min-time-ms stay within --max-power-diff of their
        // mean, or until --max-time-ms; in the second case it only warns on the console ("The power measuring budget has
        // been exhausted without reaching target precision"). A measurement that lasted --max-time-ms ran out of time.
        if (firstPass && minTimeS > 0.0 && maxTimeS > minTimeS)
        {
            launch.powerSettled = launch.measureS < maxTimeS ? 1 : 0;
        }

        // Sums over all launches of the configuration (KTT repeats the kernel inside RunKernel until the power reading is
        // stable), so accesses per launch match the power and duration, which KTT averages over the same launches.
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
    const std::string& levelWithoutParameters, const StatsMap& stats, const Hardware& hardware, const Footprints& footprints)
{
    Row row;
    row.probe = probe;
    row.operation = operation;
    row.valid = result.IsValid();

    const auto& configuration = result.GetConfiguration();
    const auto loadOp = ParameterValue(configuration, "LOAD_OP");
    const auto footprint = ParameterValue(configuration, "FOOTPRINT");
    const auto sharedBytes = ParameterValue(configuration, "SMEM_BYTES");
    const auto computeOp = ParameterValue(configuration, "COMPUTE_OP");
    const auto constantBytes = ParameterValue(configuration, "CONST_BYTES");

    if (loadOp.has_value())
    {
        row.operation = OperationName(loadOp.value());
    }

    // The locality probe: its own operation names, so that its rows are not fitted together with probeGlobal's.
    if (const auto run = ParameterValue(configuration, "RUN_BYTES"))
    {
        row.operation += run.value() == 0 ? "-stream" : "-run" + std::to_string(run.value());
    }

    // --misaligned: an operation of its own per start offset, so that its rows are not fitted with the aligned probes.
    if (const auto offset = ParameterValue(configuration, "OFFSET_BYTES"))
    {
        row.operation += "-off" + std::to_string(offset.value());
    }

    if (const auto spread = ParameterValue(configuration, "SPREAD"))
    {
        row.operation += spread.value() == 2 ? "-sub-pairs" : spread.value() != 0 ? "-sub-scattered" : "-sub-adjacent";
    }

    // --write-path: the words accessed per sector (of 32 / WIDTH).
    // --l1-sweep: sectors and lines per warp instruction.
    if (const auto sectors = ParameterValue(configuration, "SWEEP_SECTORS"))
    {
        row.operation += "-s" + std::to_string(sectors.value()) + "-l"
            + std::to_string(ParameterValue(configuration, "SWEEP_LINES").value_or(0));
    }

    if (const auto words = ParameterValue(configuration, "PARTIAL_WORDS"))
    {
        row.operation += "-" + std::to_string(words.value()) + "words";
    }

    // The mixed-intensity probe: an operation of its own for every number of FFMAs per load.
    if (const auto fmas = ParameterValue(configuration, "MIX_FMAS"))
    {
        row.operation = ParameterValue(configuration, "MIX_OP").value_or(0) == 1 ? "ld-int" + std::to_string(fmas.value())
            : MixedName(fmas.value());
    }

    if (footprint.has_value())
    {
        row.footprint = footprint.value();
        row.width = ParameterValue(configuration, "WIDTH").value_or(DefaultWidth);
        row.level = ExpectedLevel(loadOp.value_or(LoadDefault), row.footprint, hardware, footprints);
    }
    else if (sharedBytes.has_value())
    {
        const uint64_t sharedOp = ParameterValue(configuration, "SHARED_OP").value_or(SharedLoad);
        row.footprint = sharedBytes.value();
        row.width = sharedOp == SharedRed ? AtomicWidth : DefaultWidth;
        row.operation = sharedOp == SharedRed ? "red.shared" : (sharedOp == SharedStore ? "st.shared" : "ld.shared");
        row.level = "shared";

        // The bank-conflict probe: 4-, 8- or 16-byte accesses, an operation of its own for every conflict degree.
        if (const auto ways = ParameterValue(configuration, "BANK_WAYS"))
        {
            row.width = ParameterValue(configuration, "SHARED_WIDTH").value_or(BankWidths.front());
            row.operation += "-" + std::to_string(ways.value()) + "way";
        }
    }
    else if (constantBytes.has_value())
    {
        // A warp reads one word per access, so bytes per thread access mean nothing here: no width, energies per access.
        row.footprint = constantBytes.value();
        row.operation = "ld.const";
        row.level = ExpectedConstantLevel(row.footprint, hardware, footprints);
    }
    else if (const auto operandOp = ParameterValue(configuration, "OPERAND_OP"))
    {
        // probeOperands (--payload-bits): per operation of one thread, as the arithmetic probes.
        row.operation = OperandName(operandOp.value());
        row.flopsPerOp = operandOp.value() == OperandHmma ? 2.0 * 16 * 8 * 16 / 32 : 2.0;
        row.level = "compute";
    }
    else if (computeOp.has_value())
    {
        row.operation = ComputeName(computeOp.value());
        row.flopsPerOp = FlopsPerOperation(computeOp.value(), hardware.computeCapabilityMajor);
        row.level = "compute";

        // --lanes: the active lanes per warp; the accesses are counted per thread, active or not (per warp
        // instruction: times 32).
        if (const auto lanes = ParameterValue(configuration, "LANES"))
        {
            row.operation += "-" + std::to_string(lanes.value());
        }
    }
    else
    {
        row.level = levelWithoutParameters;
    }

    // The latency probe: an operation of its own for every number of chains and working warps per SM.
    if (const auto chains = ParameterValue(configuration, "CHAINS"))
    {
        row.operation = LatencyName(chains.value(), ParameterValue(configuration, "ACTIVE_WARPS").value_or(1)
            * ParameterValue(configuration, "BLOCKS_PER_SM").value_or(1));
    }

    // The write probe: an operation of its own for each pattern. Local memory is private to a thread and its footprint is
    // per SM, so it is at L1 up to the L1 footprint and, for all SMs together far beyond L2, at DRAM above.
    if (const auto write = ParameterValue(configuration, "WRITE_OP"))
    {
        row.operation = WriteName(write.value());

        if (write.value() == WriteLocal)
        {
            row.level = row.footprint <= footprints.l1 ? "L1" : "DRAM";
        }
    }

    if (const auto duty = ParameterValue(configuration, "DUTY"))
    {
        row.duty = duty.value();
    }

    if (const auto payload = ParameterValue(configuration, "PAYLOAD"))
    {
        row.payload = PayloadName(payload.value());
    }

    if (const auto activeSms = ParameterValue(configuration, "ACTIVE_SMS"))
    {
        row.activeSms = activeSms.value();
    }

    const auto statsIterator = stats.find(result.GetKernelName() + "|" + configuration.GetString());

    if (statsIterator != stats.cend())
    {
        const LaunchStats& launch = statsIterator->second;
        row.accesses = launch.accesses;
        row.launches = launch.launches;
        row.activeS = launch.meanActiveSeconds;
        row.placement = launch.placement;
        row.measureS = launch.measureS;
        row.powerSettled = launch.powerSettled;

        if (!row.placement.empty() && row.placement != "ok")
        {
            AppendNote(row.note, "placement: " + row.placement);
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

PayloadTexture::PayloadTexture(const uint32_t* data, const size_t bytes, const ktt::DeviceIndex device)
{
    CUcontext context = nullptr;
    CUdevice current = 0;
    CUdevice expected = 0;
    CheckCuda(cuCtxGetCurrent(&context), "cuCtxGetCurrent");

    if (context == nullptr)
    {
        throw std::runtime_error("no CUDA context is current, cannot create the texture of the texture probe");
    }

    CheckCuda(cuCtxGetDevice(&current), "cuCtxGetDevice");
    CheckCuda(cuDeviceGet(&expected, static_cast<int>(device)), "cuDeviceGet");

    if (current != expected)
    {
        throw std::runtime_error("the current CUDA context is not on the selected device");
    }

    int maxTexels = 0;
    CheckCuda(cuDeviceGetAttribute(&maxTexels, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH, current),
        "cuDeviceGetAttribute");

    if (bytes / 16 > static_cast<size_t>(maxTexels))
    {
        throw std::runtime_error("the DRAM footprint has more texels than a 1D texture may have ("
            + std::to_string(maxTexels) + ")");
    }

    CheckCuda(cuMemAlloc(&m_Buffer, bytes), "cuMemAlloc");

    if (data != nullptr)
    {
        CheckCuda(cuMemcpyHtoD(m_Buffer, data, bytes), "cuMemcpyHtoD");
    }
    else
    {
        CheckCuda(cuMemsetD32(m_Buffer, 0, bytes / 4), "cuMemsetD32");
    }

    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_LINEAR;
    resource.res.linear.devPtr = m_Buffer;
    resource.res.linear.format = CU_AD_FORMAT_UNSIGNED_INT32;
    resource.res.linear.numChannels = 4;
    resource.res.linear.sizeInBytes = bytes;
    CUDA_TEXTURE_DESC description{};
    description.filterMode = CU_TR_FILTER_MODE_POINT;
    description.flags = CU_TRSF_READ_AS_INTEGER;
    CheckCuda(cuTexObjectCreate(&m_Texture, &resource, &description, nullptr), "cuTexObjectCreate");
}

OffsetBuffer::OffsetBuffer(const uint32_t* data, const size_t bytes, const size_t slack, const ktt::DeviceIndex device)
{
    CUcontext context = nullptr;
    CUdevice current = 0;
    CUdevice expected = 0;
    CheckCuda(cuCtxGetCurrent(&context), "cuCtxGetCurrent");

    if (context == nullptr)
    {
        throw std::runtime_error("no CUDA context is current, cannot create the buffer of --misaligned");
    }

    CheckCuda(cuCtxGetDevice(&current), "cuCtxGetDevice");
    CheckCuda(cuDeviceGet(&expected, static_cast<int>(device)), "cuDeviceGet");

    if (current != expected)
    {
        throw std::runtime_error("the current CUDA context is not on the selected device");
    }

    // cuMemAlloc aligns to at least 256 bytes, so offset 0 is on a line.
    CheckCuda(cuMemAlloc(&m_Buffer, bytes + slack), "cuMemAlloc");
    CheckCuda(cuMemcpyHtoD(m_Buffer, data, bytes), "cuMemcpyHtoD");
    CheckCuda(cuMemcpyHtoD(m_Buffer + bytes, data, slack), "cuMemcpyHtoD");
}

OffsetBuffer::~OffsetBuffer()
{
    if (m_Buffer != 0)
    {
        cuMemFree(m_Buffer);
    }
}

PayloadTexture::~PayloadTexture()
{
    if (m_Texture != 0)
    {
        cuTexObjectDestroy(m_Texture);
    }

    if (m_Buffer != 0)
    {
        cuMemFree(m_Buffer);
    }
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

std::vector<uint32_t> RandomPayload(const uint64_t bytes)
{
    std::cout << "Generating " << FormatBytes(bytes) << " of random payload..." << std::flush;
    std::vector<uint32_t> payload(bytes / sizeof(uint32_t));
    std::mt19937_64 generator(12345);

    for (size_t i = 0; i + 1 < payload.size(); i += 2)
    {
        const uint64_t value = generator();
        payload[i] = static_cast<uint32_t>(value);
        payload[i + 1] = static_cast<uint32_t>(value >> 32);
    }

    std::cout << " done\n";
    return payload;
}

ProbeArguments AddProbeArguments(ktt::Tuner& tuner, std::vector<uint32_t>& payload, const Options& options,
    const uint64_t blocks)
{
    // The zero buffer: with --payload zero, and with --payload-bits, whose groups measure zero next to the patterns (until
    // 2026-11-06 21:00 it was the random buffer there, so the "zero" rows of PayloadBits1-3 read random data).
    const bool zeroPayload = std::find(options.payloads.cbegin(), options.payloads.cend(), PayloadZero)
        != options.payloads.cend() || !options.payloadBits.empty();
    std::vector<uint32_t> sink(blocks * options.blockSize, 0);
    std::vector<uint64_t> counts(3 * blocks + 1, 0);
    std::vector<uint32_t> smCounters(SmCounterEntries, 0);

    ProbeArguments arguments;
    // Read and written: the store probe overwrites the payload with its own (random or zero) words.
    arguments.data = tuner.AddArgumentVector(payload, ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device,
        ktt::ArgumentManagementType::Framework, true);
    arguments.zeroData = zeroPayload ? tuner.AddArgumentVector(std::vector<uint32_t>(payload.size(), 0u),
        ktt::ArgumentAccessType::ReadWrite) : arguments.data;
    arguments.sink = tuner.AddArgumentVector(sink, ktt::ArgumentAccessType::WriteOnly);
    arguments.counts = tuner.AddArgumentVector(counts, ktt::ArgumentAccessType::ReadWrite);

    // --misaligned: the payload in a buffer of its own with 128 bytes to spare, and one argument per start offset.
    if (options.misaligned)
    {
        const size_t bytes = payload.size() * sizeof(uint32_t);
        constexpr size_t slack = 128;
        arguments.offsetBuffer = std::make_shared<OffsetBuffer>(payload.data(), bytes, slack, options.device);

        for (const uint64_t offset : MisalignedOffsets)
        {
            arguments.offsetData[offset] = tuner.AddArgumentVector(
                reinterpret_cast<ktt::ComputeBuffer>(arguments.offsetBuffer->Address() + offset), bytes, sizeof(uint32_t),
                ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
        }
    }

    arguments.smCounters = tuner.AddArgumentVector(smCounters, ktt::ArgumentAccessType::ReadWrite);
    // Set per configuration by the launchers.
    arguments.gridSplit = tuner.AddArgumentScalar(static_cast<uint32_t>(0));
    arguments.activeSms = tuner.AddArgumentScalar(AllSms);
    arguments.idleRatio = tuner.AddArgumentScalar(static_cast<uint32_t>(0));
    arguments.payloadMask = tuner.AddArgumentScalar(static_cast<uint32_t>(0xffffffffu));
    arguments.operandA = tuner.AddArgumentScalar(static_cast<uint32_t>(0));
    arguments.operandC = tuner.AddArgumentScalar(static_cast<uint32_t>(0));
    arguments.duration = tuner.AddArgumentScalar(static_cast<uint64_t>(options.launchMs * 1'000'000));
    arguments.constantBytes = tuner.AddArgumentScalar(static_cast<uint32_t>(0));
    arguments.texture = tuner.AddArgumentScalar(static_cast<uint64_t>(0)); // texture object handle
    arguments.sectorMask = tuner.AddArgumentScalar(static_cast<uint32_t>(0));
    arguments.runBits = tuner.AddArgumentScalar(static_cast<uint32_t>(0));
    arguments.stride = tuner.AddArgumentScalar(static_cast<uint32_t>(1));
    arguments.bankWays = tuner.AddArgumentScalar(static_cast<uint32_t>(1));
    arguments.fmaGroups = tuner.AddArgumentScalar(static_cast<uint32_t>(0));
    arguments.activeWarps = tuner.AddArgumentScalar(static_cast<uint32_t>(0));
    arguments.lineMask = tuner.AddArgumentScalar(static_cast<uint32_t>(0));
    arguments.zero = tuner.AddArgumentScalar(static_cast<uint32_t>(0));

    // The payloads of --payload-bits, each a buffer of its own (uploaded on first use, as the zeros), and the bit
    // statistics of every payload the probes can read.
    PayloadPatterns() = options.payloadBits;
    arguments.payloadStatistics["random"] = PayloadStatistics(payload);
    arguments.payloadStatistics["zero"] = PayloadStatistics(std::vector<uint32_t>(PatternTileWords, 0u));

    for (const auto& spec : options.payloadBits)
    {
        std::vector<uint32_t> words = PatternPayload(spec, payload.size());
        arguments.payloadStatistics[spec] = PayloadStatistics(words);
        arguments.patternData.push_back(tuner.AddArgumentVector(words, ktt::ArgumentAccessType::ReadWrite));
    }

    // probeConstant reads constantData, which KTT fills with random words from the payload when it loads the module.
    if (options.constant)
    {
        auto constantPayload = std::make_unique<std::array<uint32_t, ConstantWords>>();
        std::copy_n(payload.cbegin(), ConstantWords, constantPayload->begin());
        arguments.constantData = tuner.AddArgumentSymbol(*constantPayload, "", "constantData");
    }

    return arguments;
}

ktt::KernelDefinitionId AddMixedDefinition(ktt::Tuner& tuner, const ProbeArguments& a, const Options& options,
    const uint32_t multiprocessors)
{
    const ktt::KernelDefinitionId definition = tuner.AddKernelDefinition("probeMixed", KernelSource(options.kernelDirectory,
        "Mixed.cu"), ktt::DimensionVector(multiprocessors), ktt::DimensionVector(1));
    tuner.SetArguments(definition, {a.data, a.sink, a.counts, a.smCounters, a.gridSplit, a.activeSms, a.idleRatio,
        a.operandC, a.fmaGroups, a.duration});
    return definition;
}

ProbeDefinitions AddProbeDefinitions(ktt::Tuner& tuner, const ProbeArguments& a, const Options& options,
    const uint32_t multiprocessors)
{
    const ktt::DimensionVector grid(multiprocessors);
    const ktt::DimensionVector block(1);
    auto add = [&](const std::string& kernel, const std::string& file, const std::vector<ktt::ArgumentId>& arguments)
    {
        const ktt::KernelDefinitionId definition = tuner.AddKernelDefinition(kernel, KernelSource(options.kernelDirectory,
            file), grid, block);
        tuner.SetArguments(definition, arguments);
        return definition;
    };

    ProbeDefinitions definitions;
    definitions.sleep = add("probeSleep", "Baseline.cu", {a.sink, a.counts, a.activeSms, a.duration});
    definitions.spin = add("probeSpin", "Baseline.cu", {a.sink, a.counts, a.duration});
    definitions.global = add("probeGlobal", "Global.cu", {a.data, a.sink, a.counts, a.smCounters, a.gridSplit, a.activeSms,
        a.idleRatio, a.duration});
    definitions.shared = add("probeShared", "Shared.cu", {a.sink, a.counts, a.activeSms, a.idleRatio, a.payloadMask,
        a.duration});
    definitions.compute = add("probeCompute", "Compute.cu", {a.sink, a.counts, a.activeSms, a.idleRatio, a.payloadMask,
        a.operandA, a.operandC, a.duration});

    if (options.constant)
    {
        definitions.constant = add("probeConstant", "Constant.cu", {a.sink, a.counts, a.smCounters, a.constantBytes,
            a.activeSms, a.idleRatio, a.duration, a.constantData.value()});
    }

    if (options.texture)
    {
        definitions.texture = add("probeTexture", "Texture.cu", {a.texture, a.sink, a.counts, a.smCounters, a.gridSplit,
            a.activeSms, a.idleRatio, a.duration});
    }

    if (options.dramLocality)
    {
        definitions.scatter = add("probeScatter", "Scatter.cu", {a.data, a.sink, a.counts, a.sectorMask, a.runBits,
            a.stride, a.activeSms, a.idleRatio, a.duration});
    }

    if (options.subSector || options.writePath)
    {
        definitions.partial = add("probePartial", "Partial.cu", {a.data, a.sink, a.counts, a.sectorMask, a.stride,
            a.activeSms, a.idleRatio, a.duration});
    }

    if (options.l1Sweep)
    {
        definitions.l1Sweep = add("probeL1Sweep", "L1Sweep.cu", {a.data, a.sink, a.counts, a.sectorMask, a.activeSms,
            a.idleRatio, a.duration});
    }

    if (!options.payloadBits.empty())
    {
        definitions.operands = add("probeOperands", "Operands.cu", {a.data, a.sink, a.counts, a.activeSms, a.idleRatio,
            a.zero, a.duration});
    }

    if (options.bankConflicts)
    {
        definitions.banks = add("probeSharedBanks", "Shared.cu", {a.sink, a.counts, a.bankWays, a.activeSms, a.idleRatio,
            a.payloadMask, a.duration});
    }

    if (options.mixed || options.mixedInt)
    {
        definitions.mixed = AddMixedDefinition(tuner, a, options, multiprocessors);
    }

    if (options.writes)
    {
        definitions.write = add("probeWrite", "Write.cu", {a.data, a.sink, a.counts, a.smCounters, a.gridSplit, a.activeSms,
            a.idleRatio, a.duration});
    }

    if (options.latency)
    {
        definitions.latency = add("probeLatency", "Latency.cu", {a.data, a.sink, a.counts, a.activeSms, a.activeWarps,
            a.lineMask, a.zero, a.idleRatio, a.duration});
    }

    return definitions;
}

ProbeTextures CreateProbeTextures(const std::vector<uint32_t>& payload, const Options& options)
{
    ProbeTextures textures;

    if (options.texture)
    {
        const size_t bytes = payload.size() * sizeof(uint32_t);
        textures.random = std::make_unique<PayloadTexture>(payload.data(), bytes, options.device);

        if (std::find(options.payloads.cbegin(), options.payloads.cend(), PayloadZero) != options.payloads.cend())
        {
            textures.zero = std::make_unique<PayloadTexture>(nullptr, bytes, options.device);
        }
    }

    return textures;
}

ProbeSettings MakeProbeSettings(const ProbeArguments& a, const ProbeDefinitions& definitions, const Footprints& footprints,
    const ProbeTextures& textures)
{
    const uint64_t randomTexture = textures.random != nullptr ? textures.random->GetHandle() : 0;
    const uint64_t zeroTexture = textures.zero != nullptr ? textures.zero->GetHandle() : 0;
    const uint64_t disjointFrom = footprints.disjointFrom;

    auto isZero = [](const ktt::KernelConfiguration& configuration)
    {
        return ParameterValue(configuration, "PAYLOAD").value_or(PayloadRandom) == PayloadZero;
    };

    // The global-memory probes read a buffer of zeros for the zero payload. KTT only creates the buffers of the arguments
    // the definition had when the run started, so the zero buffer is uploaded on first use.
    auto selectData = [=](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        const uint64_t payload = ParameterValue(configuration, "PAYLOAD").value_or(PayloadRandom);

        // --misaligned: the random payload from a start OFFSET_BYTES into its own buffer.
        if (const auto offset = ParameterValue(configuration, "OFFSET_BYTES"))
        {
            const ktt::ArgumentId view = a.offsetData.at(offset.value());

            if (!interface.HasBuffer(view))
            {
                interface.UploadBuffer(view);
            }

            return view;
        }

        if (payload >= PayloadPatternFirst)
        {
            const ktt::ArgumentId pattern = a.patternData.at(payload - PayloadPatternFirst);

            if (!interface.HasBuffer(pattern))
            {
                interface.UploadBuffer(pattern);
            }

            return pattern;
        }

        if (isZero(configuration) && !interface.HasBuffer(a.zeroData))
        {
            interface.UploadBuffer(a.zeroData);
        }

        return isZero(configuration) ? a.zeroData : a.data;
    };

    ProbeSettings settings;
    settings.common = [=](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        const uint32_t activeSms = ActiveSms(configuration);
        const uint32_t idleRatio = IdleRatio(ParameterValue(configuration, "DUTY").value_or(100));
        const uint32_t payloadMask = isZero(configuration) ? 0u : 0xffffffffu;
        interface.UpdateScalarArgument(a.activeSms, &activeSms);
        interface.UpdateScalarArgument(a.idleRatio, &idleRatio);
        interface.UpdateScalarArgument(a.payloadMask, &payloadMask);
    };
    const Configure common = settings.common;

    // Footprints from disjointFrom on are split over all blocks of the grid (see probeGlobal).
    settings.global = [=](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        common(interface, configuration);
        const uint32_t gridSplit = ParameterValue(configuration, "FOOTPRINT").value_or(0) >= disjointFrom ? 1 : 0;
        interface.UpdateScalarArgument(a.gridSplit, &gridSplit);
        interface.ChangeArguments(definitions.global, {selectData(interface, configuration), a.sink, a.counts,
            a.smCounters, a.gridSplit, a.activeSms, a.idleRatio, a.duration});
    };

    // probeOperands reads its operands from the payload (--payload-bits).
    if (definitions.operands.has_value())
    {
        const ktt::KernelDefinitionId operands = definitions.operands.value();
        settings.operands = [=](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
        {
            common(interface, configuration);
            interface.ChangeArguments(operands, {selectData(interface, configuration), a.sink, a.counts, a.activeSms,
                a.idleRatio, a.zero, a.duration});
        };
    }

    settings.texture = [=](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        common(interface, configuration);
        const uint32_t gridSplit = ParameterValue(configuration, "FOOTPRINT").value_or(0) >= disjointFrom ? 1 : 0;
        const uint64_t handle = isZero(configuration) ? zeroTexture : randomTexture;
        interface.UpdateScalarArgument(a.gridSplit, &gridSplit);
        interface.UpdateScalarArgument(a.texture, &handle);
    };

    // Runs of RUN_BYTES consecutive sectors, placed apart by ScatterStride (RUN_BYTES 0: one stream, stride 1).
    settings.scatter = [=](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        common(interface, configuration);
        const uint64_t run = ParameterValue(configuration, "RUN_BYTES").value_or(0);
        const uint32_t sectorMask = static_cast<uint32_t>(ParameterValue(configuration, "FOOTPRINT").value_or(0) / SectorBytes
            - 1);
        const uint32_t runBits = run == 0 ? 0 : static_cast<uint32_t>(std::log2(static_cast<double>(run / SectorBytes)));
        const uint32_t stride = run == 0 ? 1u : ScatterStride;
        interface.UpdateScalarArgument(a.sectorMask, &sectorMask);
        interface.UpdateScalarArgument(a.runBits, &runBits);
        interface.UpdateScalarArgument(a.stride, &stride);
        interface.ChangeArguments(definitions.scatter.value(), {selectData(interface, configuration), a.sink, a.counts,
            a.sectorMask, a.runBits, a.stride, a.activeSms, a.idleRatio, a.duration});
    };

    // The sectors of a warp adjacent (SPREAD 0) or scattered by ScatterStride (SPREAD 1).
    settings.partial = [=](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        common(interface, configuration);
        const uint32_t sectorMask = static_cast<uint32_t>(ParameterValue(configuration, "FOOTPRINT").value_or(0) / SectorBytes
            - 1);
        // SPREAD 0: the warp's sectors next to each other (4 per L2 request), 2: every other sector (2 per request),
        // 1: scattered (1 per request).
        const uint64_t spread = ParameterValue(configuration, "SPREAD").value_or(0);
        const uint32_t stride = spread == 2 ? 2u : spread != 0 ? ScatterStride : 1u;
        interface.UpdateScalarArgument(a.sectorMask, &sectorMask);
        interface.UpdateScalarArgument(a.stride, &stride);
        interface.ChangeArguments(definitions.partial.value(), {selectData(interface, configuration), a.sink, a.counts,
            a.sectorMask, a.stride, a.activeSms, a.idleRatio, a.duration});
    };

    // --l1-sweep: the footprint's lines minus 1 in the argument sectorMask.
    settings.l1Sweep = [=](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        common(interface, configuration);
        const uint32_t lineMask = static_cast<uint32_t>(ParameterValue(configuration, "FOOTPRINT").value_or(0) / 128 - 1);
        interface.UpdateScalarArgument(a.sectorMask, &lineMask);
        interface.ChangeArguments(definitions.l1Sweep.value(), {selectData(interface, configuration), a.sink, a.counts,
            a.sectorMask, a.activeSms, a.idleRatio, a.duration});
    };

    // The conflict degree is an argument, so that every degree runs the same SASS.
    settings.banks = [=](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        common(interface, configuration);
        const uint32_t ways = static_cast<uint32_t>(ParameterValue(configuration, "BANK_WAYS").value_or(1));
        interface.UpdateScalarArgument(a.bankWays, &ways);
    };

    // The FFMAs per load are an argument (groups of MixedFmaGroup per UNROLL loads), so that every intensity runs the same
    // SASS; c = -1.9 as for the FFMA probe. Random data only.
    settings.mixed = [=](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        common(interface, configuration);
        const uint32_t gridSplit = ParameterValue(configuration, "FOOTPRINT").value_or(0) >= disjointFrom ? 1 : 0;
        const uint64_t fmas = ParameterValue(configuration, "MIX_FMAS").value_or(0);
        const uint32_t fmaGroups = static_cast<uint32_t>(fmas * ParameterValue(configuration, "UNROLL").value_or(16)
            / MixedFmaGroup);
        const uint32_t operandC = FloatBits(-1.9f);
        interface.UpdateScalarArgument(a.gridSplit, &gridSplit);
        interface.UpdateScalarArgument(a.fmaGroups, &fmaGroups);
        interface.UpdateScalarArgument(a.operandC, &operandC);
    };

    // The write probe: stripes and slots as for the global probe (local memory ignores them). Random data only.
    settings.write = [=](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        common(interface, configuration);
        const uint32_t gridSplit = ParameterValue(configuration, "FOOTPRINT").value_or(0) >= disjointFrom ? 1 : 0;
        interface.UpdateScalarArgument(a.gridSplit, &gridSplit);
    };

    // The latency probe: the working warps per block and the footprint's lines (128 bytes each) - 1.
    settings.latency = [=](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        common(interface, configuration);
        const uint32_t activeWarps = static_cast<uint32_t>(ParameterValue(configuration, "ACTIVE_WARPS").value_or(1));
        const uint32_t lineMask = static_cast<uint32_t>(ParameterValue(configuration, "FOOTPRINT").value_or(0) / 128 - 1);
        interface.UpdateScalarArgument(a.activeWarps, &activeWarps);
        interface.UpdateScalarArgument(a.lineMask, &lineMask);
    };

    settings.constant = [=](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        common(interface, configuration);
        const uint32_t constantBytes = static_cast<uint32_t>(ParameterValue(configuration, "CONST_BYTES").value_or(0));
        interface.UpdateScalarArgument(a.constantBytes, &constantBytes);
    };

    // The operands of the arithmetic recurrences: x -> x * x + c with c = -1.9 for floating point (half2 0xbf9a = -1.9
    // twice), the LCG x -> x * a + c for integers, x -> a * rsqrt(|x|) + c with a = -1, c = 1.5 for MUFU, x -> sin(a * x + c)
    // with a = 3, c = 0.5 for MUFU.SIN; all 0 for the zero payload.
    settings.compute = [=](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        common(interface, configuration);
        const uint64_t operation = ParameterValue(configuration, "COMPUTE_OP").value_or(ComputeLoop);
        uint32_t operandA = 0;
        uint32_t operandC = 0;

        if (!isZero(configuration))
        {
            if (operation == ComputeFfma || operation == ComputeDfma || operation == ComputeStaticIndex
                || operation == ComputeDynamicIndex || operation == ComputeFfma3 || operation == ComputeFfmaLanes
                || operation == ComputeFfmaLanesBranch || operation == ComputeFfmaLanesRotating)
            {
                operandC = FloatBits(-1.9f);
            }
            else if (operation == ComputeImad)
            {
                operandA = 1664525u;
                operandC = 1013904223u;
            }
            else if (operation == ComputeHfma2)
            {
                operandC = 0xbf9abf9au;
            }
            else if (operation == ComputeMufu)
            {
                operandA = FloatBits(-1.0f);
                operandC = FloatBits(1.5f);
            }
            else if (operation == ComputeFfmaLoop || operation == ComputeFmulFadd)
            {
                operandC = FloatBits(-1.9f);
            }
            else if (operation == ComputeMufuSin)
            {
                operandA = FloatBits(3.0f);
                operandC = FloatBits(0.5f);
            }
            else if (operation == ComputeFadd)
            {
                operandC = FloatBits(1.9f); // x -> 1.9 - x
            }
            else if (operation == ComputeFmul)
            {
                operandA = FloatBits(1.37f); // x -> x * 1.37, then x / 1.37 (as a product)
                operandC = FloatBits(1.0f / 1.37f);
            }
            else if (operation >= ComputeMufuRcp && operation <= ComputeMufuRsq)
            {
                operandC = FloatBits(-1.9f); // the FFMA recurrence that feeds the MUFU
            }
        }

        interface.UpdateScalarArgument(a.operandA, &operandA);
        interface.UpdateScalarArgument(a.operandC, &operandC);
    };

    return settings;
}
