// Direct NVML access, independent of KTT's own power measurement: finds the GPU by PCI bus id, locks its clock on request,
// records a trace of power, energy counter, clocks, temperature and clock-limiter reasons for every configuration, and
// summarizes it.
#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nvml.h>
#include <json.hpp>

#include <Ktt.h>

#include "Common.h"

// Bits of nvmlDeviceGetCurrentClocksEventReasons (formerly ...ThrottleReasons, same bits). Limiters lower the clock.
struct ClockReason
{
    unsigned long long bit;
    const char* name;
    bool limiter;
};

struct TraceSample
{
    double timeS = NaN;
    double powerW = NaN;        // nvmlDeviceGetPowerUsage, which KTT uses too
    double instantPowerW = NaN; // field NVML_FI_DEV_POWER_INSTANT
    double averagePowerW = NaN; // field NVML_FI_DEV_POWER_AVERAGE (1 s average, Ampere and newer)
    double energyJ = NaN;
    double smClockMHz = NaN;
    double memoryClockMHz = NaN;
    double temperatureC = NaN;
    double fanPct = NaN;        // nvmlDeviceGetFanSpeed: intended fan speed, % of the maximum (the fans draw board power)
    unsigned long long reasons = 0;
};

// Statistics over the second half of one configuration's trace (the first half contains the warm-up of the reading).
struct TraceSummary
{
    size_t samples = 0;
    double sampledPowerW = NaN;
    double instantPowerW = NaN;
    double averagePowerW = NaN;
    double counterPowerW = NaN;
    double minClockMHz = NaN;
    double maxClockMHz = NaN;
    double meanClockMHz = NaN;
    double meanMemoryClockMHz = NaN;
    double meanTemperatureC = NaN;
    double maxTemperatureC = NaN;
    double meanFanPct = NaN;
    // Largest deviation of the sampled power from its mean over the last tailSeconds of Stop(), in % (KTT's precise
    // measurement stops once its own samples of the same reading stay within --max-power-diff over --min-time-ms).
    double tailSpreadPct = NaN;
    unsigned long long reasons = 0;
    // Time NVML's clock policies were active during the configuration, from its cumulative counters per reason (read at
    // Start() and Stop(); NEWER_ARCHITECTURES.md section 12), which can catch events shorter than the sampling interval.
    // Not always a lowered clock: on an RTX 3090 at a locked 1200 MHz, the power-cap counter ran for 98 % of the IMMA
    // probe at 192 W while the clock stayed at the lock and the sampled reasons were clear (run Tensor1, 2026-11-16).
    // NaN where the driver does not report them; limitedBy names the reasons with time.
    double limitedMs = NaN;
    std::string limitedBy;
};

// The reasons set in a bit mask of nvmlDeviceGetCurrentClocksThrottleReasons, joined by '+' (only the limiters if asked).
std::string DescribeReasons(const unsigned long long reasons, const bool limitersOnly);

// Talks to NVML directly, independently of KTT's own power measurement.
// What a model file records about the GPU and its software (tools/build_model.py), so that a model can tell when it was
// made on another card or driver. Empty or NaN where NVML does not report it (GeForce cards have no serial number).
struct DeviceProvenance
{
    std::string driverVersion;     // nvmlSystemGetDriverVersion
    std::string cudaDriverVersion; // the CUDA version the driver supports, e.g. 12.9
    std::string uuid;              // nvmlDeviceGetUUID
    std::string serial;            // nvmlDeviceGetSerial
    double powerLimitW = NaN;      // the enforced power limit
};

// The board's settings that change what a power reading means (NEWER_ARCHITECTURES.md sections 4.5, 4.7 and 12): read
// once per run and written to the energy table. Null where the driver or GPU does not report a value.
nlohmann::json DescribeBoard(nvmlDevice_t device);

// Settings of the current CUDA context that change what the memory probes measure (NEWER_ARCHITECTURES.md sections 4.4
// and 21): whether the GPU supports compressible memory (the probes' default allocations are not compressible), the L2
// fetch granularity hint and the persisting-L2 set-aside. Null where the driver does not report them.
nlohmann::json DescribeCudaContext();

class NvmlMonitor
{
public:
    explicit NvmlMonitor(const ktt::DeviceIndex cudaOrdinal);

    ~NvmlMonitor();

    NvmlMonitor(const NvmlMonitor&) = delete;
    NvmlMonitor& operator=(const NvmlMonitor&) = delete;

    void LockClocks(const uint32_t mhz);

    void Start(const std::string& label, const uint64_t intervalMs);

    TraceSummary Stop(const double tailSeconds = 0.0);

    void WriteTraces(const std::string& file) const;

    const std::string& GetBusId() const
    {
        return m_BusId;
    }

    // Index of this GPU as nvidia-smi shows it.
    std::optional<unsigned int> GetNvmlIndex() const
    {
        return m_NvmlIndex;
    }

    std::optional<unsigned int> GetGpuCount() const
    {
        return m_GpuCount;
    }

    std::optional<bool> KttReadsSameGpu() const
    {
        return m_KttReadsSameGpu;
    }

    DeviceProvenance GetProvenance() const;

    // DescribeBoard() of this GPU, with the clock-limited time per reason since the monitor started.
    nlohmann::json GetBoard() const;

    // NVML's supported memory clocks, and the supported SM clocks at the memory clock `memoryMHz` (empty where NVML does not
    // list them).
    std::vector<double> SupportedMemoryClocks() const;
    std::vector<double> SupportedSmClocks(const double memoryMHz) const;

    double GetPowerLimitW() const
    {
        return m_PowerLimitW;
    }

    bool HasEnergyCounter() const
    {
        return m_HasEnergyCounter;
    }

private:
    nvmlDevice_t m_Device{};
    std::string m_BusId;
    std::optional<bool> m_KttReadsSameGpu;
    std::optional<unsigned int> m_NvmlIndex;
    std::optional<unsigned int> m_GpuCount;
    double m_PowerLimitW = NaN;
    bool m_Initialized = false;
    bool m_HasEnergyCounter = false;
    bool m_ClocksLocked = false;
    std::thread m_Thread;
    std::atomic<bool> m_Stop{true};
    std::string m_Label;
    std::vector<TraceSample> m_Samples;
    std::vector<double> m_LimitedNsAtStart;   // per LimitReasons() entry, at Start()
    std::vector<double> m_LimitedNsAtCreation;
    std::vector<std::pair<std::string, TraceSample>> m_AllSamples;

    void StopSampling();

    TraceSample TakeSample(const double timeS) const;

    static TraceSummary Summarize(const std::vector<TraceSample>& samples, const double tailSeconds);
};
