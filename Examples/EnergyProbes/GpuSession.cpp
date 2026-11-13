#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>

#include <cuda.h>

#include "GpuSession.h"

#if !KTT_CUDA_EXAMPLE
#error "EnergyProbes requires the CUDA backend, power data comes from NVML"
#endif

namespace
{

// NVML readings for `seconds` while no kernel runs, as a row of level "idle".
Row IdleRow(NvmlMonitor& monitor, const std::string& label, const uint64_t seconds, const uint64_t traceIntervalMs)
{
    std::cout << "\n=== Idle " << label << ": " << seconds << " s without a kernel ===" << std::endl;
    monitor.Start(label, traceIntervalMs);
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    const TraceSummary summary = monitor.Stop();

    Row row;
    row.probe = label;
    row.operation = "idle";
    row.level = "idle";
    row.valid = true;
    row.hasPower = !std::isnan(summary.sampledPowerW);
    row.powerW = summary.sampledPowerW;
    row.durationS = static_cast<double>(seconds);
    row.energyJ = row.powerW * row.durationS;
    row.smClockMHz = summary.meanClockMHz;
    row.memoryClockMHz = summary.meanMemoryClockMHz;
    row.temperatureC = summary.meanTemperatureC;
    row.fanPct = summary.meanFanPct;
    row.counterPowerW = summary.counterPowerW;
    row.tracePowerW = summary.sampledPowerW;
    row.instantPowerW = summary.instantPowerW;
    row.averagePowerW = summary.averagePowerW;
    row.minClockMHz = summary.minClockMHz;
    row.maxClockMHz = summary.maxClockMHz;
    return row;
}

// The idle GPU before any CUDA context exists: with a context, the driver keeps the memory at its full clock, so this is
// the only reading of the lowest power state (Berkeley's and Wattchmen's constant power). Uses its own NVML session; the
// trace of it is not written. Empty without the monitor or with --idle-s 0.
std::vector<Row> IdleRowsWithoutContext(const Options& options)
{
    if (!options.monitor || options.idleS == 0)
    {
        return {};
    }

    try
    {
        if (cuInit(0) != CUDA_SUCCESS)
        {
            throw std::runtime_error("cuInit failed");
        }

        NvmlMonitor monitor(options.device);
        return {IdleRow(monitor, "idle-noctx", options.idleS, options.traceIntervalMs)};
    }
    catch (const std::exception& error)
    {
        std::cout << "WARNING: no idle reading without a CUDA context: " << error.what() << "\n";
        return {};
    }
}

} // namespace

GpuSession::GpuSession(const Options& options) :
    m_Options(options),
    m_Rows(IdleRowsWithoutContext(options)),
    m_Tuner(options.platform, options.device, ktt::ComputeApi::CUDA)
{
    m_Tuner.SetGlobalSizeType(ktt::GlobalSizeType::CUDA);
    m_Tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);
    m_Tuner.SetLoggingLevel(ktt::LoggingLevel::Info);

    const ktt::DeviceInfo deviceInfo = m_Tuner.GetCurrentDeviceInfo();
    m_Hardware.name = deviceInfo.GetName();
    m_Hardware.multiprocessors = deviceInfo.GetMaxComputeUnits();
    m_Hardware.globalMemory = deviceInfo.GetGlobalMemorySize();
    m_Hardware.sharedPerBlock = deviceInfo.GetLocalMemorySize();
    m_Hardware.computeCapabilityMajor = deviceInfo.GetCudaComputeCapabilityMajor();
    m_Hardware.computeCapabilityMinor = deviceInfo.GetCudaComputeCapabilityMinor();
    ApplyOverrides(m_Hardware, options);
    m_Footprints = ChooseFootprints(m_Hardware, options);
}

void GpuSession::OpenMonitor()
{
    if (m_Options.monitor)
    {
        try
        {
            m_Monitor = std::make_unique<NvmlMonitor>(m_Options.device);
        }
        catch (const std::exception& error)
        {
            std::cout << "WARNING: NVML monitor disabled: " << error.what() << "\n";
        }
    }

    if (m_Monitor != nullptr)
    {
        const auto nvmlIndex = m_Monitor->GetNvmlIndex();
        const std::string smiIndex = nvmlIndex.has_value() ? std::to_string(nvmlIndex.value()) : std::string("?");
        std::cout << "GPU: CUDA device " << m_Options.device << " = nvidia-smi GPU " << smiIndex << " (PCI "
            << m_Monitor->GetBusId() << "), power limit " << Fixed(m_Monitor->GetPowerLimitW(), 0) << " W, "
            << (m_Monitor->HasEnergyCounter() ? "energy counter available" : "no energy counter") << "\n";

        // KTT opens NVML device number <CUDA device number> for its power readings. NVML numbers GPUs in PCI order and
        // ignores CUDA_VISIBLE_DEVICES, so a different CUDA numbering makes KTT read the power of another GPU.
        if (m_Monitor->KttReadsSameGpu() == false)
        {
            const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
            throw std::runtime_error("CUDA device " + std::to_string(m_Options.device) + " is nvidia-smi GPU " + smiIndex
                + ", but KTT would read the power of nvidia-smi GPU " + std::to_string(m_Options.device) + ". "
                + (visible != nullptr ? "Do not select the GPU with CUDA_VISIBLE_DEVICES (NVML ignores it). " : "")
                + "Run with CUDA_DEVICE_ORDER=PCI_BUS_ID and --device <nvidia-smi index>.");
        }
    }
    else
    {
        std::cout << "WARNING: without the NVML monitor the GPU cannot be checked. On machines with several GPUs, run with "
            << "CUDA_DEVICE_ORDER=PCI_BUS_ID so that KTT reads the power of the GPU it runs on.\n";
    }

    if (m_Options.clockLockedExternally)
    {
        std::cout << "SM clock declared as locked (--clock-locked): rows at another clock than the baseline are withheld.\n";
    }

    if (m_Options.lockClockMHz.has_value())
    {
        if (m_Monitor == nullptr)
        {
            throw std::runtime_error("--lock-clock needs NVML, which is not available");
        }

        m_Monitor->LockClocks(m_Options.lockClockMHz.value());
        std::cout << "Locked the SM clock at " << m_Options.lockClockMHz.value() << " MHz (reset at exit; if the program is "
            << "killed, run: sudo nvidia-smi -rgc)\n";
    }
}

ktt::KernelId GpuSession::CreateKernel(const std::string& name, const ktt::KernelDefinitionId definition,
    const Configure& configure, const bool checkPlacement, const ktt::ArgumentId counts)
{
    return CreateProbeKernel(m_Tuner, name, definition, m_Options, counts, GetBlocks(), configure, checkPlacement, m_Stats,
        m_Monitor.get());
}

void GpuSession::MeasureIdle(const std::string& label)
{
    if (m_Monitor != nullptr && m_Options.idleS > 0)
    {
        m_Rows.push_back(IdleRow(*m_Monitor, label, m_Options.idleS, m_Options.traceIntervalMs));
    }
}

void GpuSession::WarmUp(const ktt::KernelId kernel)
{
    if (m_Options.warmupS > 0)
    {
        // A tolerance of 1.0 accepts any power reading, so the kernel runs for exactly the minimum time.
        const ktt::PreciseMeasurementParameters warmup(m_Options.warmupS * 1000, m_Options.warmupS * 1000, 1.0,
            ktt::DurationCalculationMethod::Average);
        RunProbe(m_Tuner, kernel, "warmup", m_Options.outputPrefix, warmup);
    }
}

void GpuSession::RunFor(const ktt::KernelId kernel, const uint64_t seconds, const std::string& label, const std::string& probe,
    const std::string& operation, const std::string& level)
{
    const ktt::PreciseMeasurementParameters fixed(seconds * 1000, seconds * 1000, 1.0, ktt::DurationCalculationMethod::Average);

    for (const auto& result : RunProbe(m_Tuner, kernel, label, m_Options.outputPrefix, fixed))
    {
        m_Rows.push_back(MakeRow(result, probe, operation, level, m_Stats, m_Hardware, m_Footprints));
    }
}

void GpuSession::Run(const ktt::KernelId kernel, const std::string& label, const std::string& probe,
    const std::string& operation, const std::string& level)
{
    const ktt::PreciseMeasurementParameters precise(m_Options.minTimeMs, m_Options.maxTimeMs, m_Options.maxPowerDiff,
        ktt::DurationCalculationMethod::Average);

    for (const auto& result : RunProbe(m_Tuner, kernel, label, m_Options.outputPrefix, precise))
    {
        Row row = MakeRow(result, probe, operation, level, m_Stats, m_Hardware, m_Footprints);
        row.blocksPerSmFit = ResidentBlocks(row.registers, m_Options.blockSize, m_Hardware);

        // A kernel with more registers than BLOCKS_PER_SM blocks allow runs fewer blocks per SM at once than the probes
        // assume, so the blocks of an SM share stripes (the placement check) and its rate is lower than designed.
        if (row.blocksPerSmFit > 0 && row.blocksPerSmFit < m_Options.blocksPerSm)
        {
            // No comma in a note: the CSV cells are not quoted (CsvText).
            AppendNote(row.note, "registers: " + std::to_string(row.registers) + " per thread (" + std::to_string(row.blocksPerSmFit)
                + " of " + std::to_string(m_Options.blocksPerSm) + " blocks fit per SM)");
        }

        m_Rows.push_back(row);
    }
}

void GpuSession::WriteTrace() const
{
    if (m_Monitor != nullptr)
    {
        m_Monitor->WriteTraces(m_Options.outputPrefix + "_trace.csv");
    }
}

ktt::Tuner& GpuSession::GetTuner()
{
    return m_Tuner;
}

const Hardware& GpuSession::GetHardware() const
{
    return m_Hardware;
}

const Footprints& GpuSession::GetFootprints() const
{
    return m_Footprints;
}

const NvmlMonitor* GpuSession::GetMonitor() const
{
    return m_Monitor.get();
}

NvmlMonitor* GpuSession::GetMonitor()
{
    return m_Monitor.get();
}

uint64_t GpuSession::GetBlocks() const
{
    return static_cast<uint64_t>(m_Hardware.multiprocessors) * m_Options.blocksPerSm;
}

std::vector<Row>& GpuSession::GetRows()
{
    return m_Rows;
}

SleepBaselines::SleepBaselines(GpuSession& session, const ktt::KernelDefinitionId sleep, const Configure& configure,
    const ktt::ArgumentId counts, const size_t count) :
    m_Session(session)
{
    for (size_t i = 0; i < count; ++i)
    {
        m_Kernels.push_back(session.CreateKernel("Sleep" + std::to_string(i), sleep, configure, false, counts));
    }
}

void SleepBaselines::RunNext()
{
    const std::string number = std::to_string(m_Next);
    m_Session.Run(m_Kernels.at(m_Next), "sleep_" + number, "sleep-" + number, "sleep", "baseline");
    ++m_Next;
}
