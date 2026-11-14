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

#include <Ktt.h>

#include "Common.h"

// Bits of nvmlDeviceGetCurrentClocksThrottleReasons ("clocks event reasons" in newer NVML). Limiters lower the clock.
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
    std::vector<std::pair<std::string, TraceSample>> m_AllSamples;

    void StopSampling();

    TraceSample TakeSample(const double timeS) const;

    static TraceSummary Summarize(const std::vector<TraceSample>& samples, const double tailSeconds);
};
