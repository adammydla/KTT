#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

#include <cuda.h>

#include "NvmlMonitor.h"

namespace
{

const std::vector<ClockReason> ClockReasons =
{
    {0x1ULL, "idle", false},
    {0x2ULL, "app-clocks", false},
    {0x4ULL, "sw-power-cap", true},
    {0x8ULL, "hw-slowdown", true},
    {0x10ULL, "sync-boost", false},
    {0x20ULL, "sw-thermal", true},
    {0x40ULL, "hw-thermal", true},
    {0x80ULL, "hw-power-brake", true},
    {0x100ULL, "display", false}
};

} // namespace

std::string DescribeReasons(const unsigned long long reasons, const bool limitersOnly)
{
    std::string result;

    for (const auto& reason : ClockReasons)
    {
        if ((reasons & reason.bit) == 0 || (limitersOnly && !reason.limiter))
        {
            continue;
        }

        result += (result.empty() ? "" : "+") + std::string(reason.name);
    }

    return result;
}

NvmlMonitor::NvmlMonitor(const ktt::DeviceIndex cudaOrdinal)
{
    nvmlReturn_t status = nvmlInit_v2();

    if (status != NVML_SUCCESS)
    {
        throw std::runtime_error(std::string("nvmlInit failed: ") + nvmlErrorString(status));
    }

    m_Initialized = true;

    // KTT has initialized CUDA already. NVML numbers GPUs in PCI order while CUDA (by default) does not, so the
    // matching NVML device is found through the PCI bus id of the CUDA device.
    CUdevice device;
    char busId[64] = {};

    if (cuDeviceGet(&device, static_cast<int>(cudaOrdinal)) != CUDA_SUCCESS
        || cuDeviceGetPCIBusId(busId, static_cast<int>(sizeof(busId)), device) != CUDA_SUCCESS)
    {
        throw std::runtime_error("could not query the PCI bus id of CUDA device " + std::to_string(cudaOrdinal));
    }

    status = nvmlDeviceGetHandleByPciBusId_v2(busId, &m_Device);

    if (status != NVML_SUCCESS)
    {
        throw std::runtime_error(std::string("NVML does not know GPU ") + busId + ": " + nvmlErrorString(status));
    }

    m_BusId = busId;

    unsigned int index = 0;
    unsigned int count = 0;

    if (nvmlDeviceGetIndex(m_Device, &index) == NVML_SUCCESS)
    {
        m_NvmlIndex = index;
    }

    if (nvmlDeviceGetCount_v2(&count) == NVML_SUCCESS)
    {
        m_GpuCount = count;
    }

    // KTT's power measurement opens NVML device number <CUDA ordinal>. Check that it is the same GPU.
    nvmlDevice_t indexDevice;

    if (nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned int>(cudaOrdinal), &indexDevice) == NVML_SUCCESS)
    {
        nvmlPciInfo_t ours{};
        nvmlPciInfo_t ktts{};

        if (nvmlDeviceGetPciInfo_v3(m_Device, &ours) == NVML_SUCCESS
            && nvmlDeviceGetPciInfo_v3(indexDevice, &ktts) == NVML_SUCCESS)
        {
            m_KttReadsSameGpu = std::string(ours.busId) == std::string(ktts.busId);
        }
    }

    unsigned int limit = 0;

    if (nvmlDeviceGetEnforcedPowerLimit(m_Device, &limit) == NVML_SUCCESS)
    {
        m_PowerLimitW = static_cast<double>(limit) / 1000.0;
    }

    unsigned long long energy = 0;
    m_HasEnergyCounter = nvmlDeviceGetTotalEnergyConsumption(m_Device, &energy) == NVML_SUCCESS;
}

NvmlMonitor::~NvmlMonitor()
{
    StopSampling();

    if (m_ClocksLocked)
    {
        const nvmlReturn_t status = nvmlDeviceResetGpuLockedClocks(m_Device);
        std::cout << (status == NVML_SUCCESS ? "Reset the locked SM clock.\n"
            : "WARNING: could not reset the locked SM clock, run: sudo nvidia-smi -rgc\n");
    }

    if (m_Initialized)
    {
        nvmlShutdown();
    }
}

void NvmlMonitor::LockClocks(const uint32_t mhz)
{
    const nvmlReturn_t status = nvmlDeviceSetGpuLockedClocks(m_Device, mhz, mhz);

    if (status == NVML_ERROR_NO_PERMISSION)
    {
        throw std::runtime_error("locking the SM clock needs root. Run the program with sudo, or lock the clock "
            "yourself: sudo nvidia-smi -lgc " + std::to_string(mhz) + "," + std::to_string(mhz));
    }

    if (status != NVML_SUCCESS)
    {
        throw std::runtime_error(std::string("nvmlDeviceSetGpuLockedClocks failed: ") + nvmlErrorString(status));
    }

    m_ClocksLocked = true;
}

void NvmlMonitor::Start(const std::string& label, const uint64_t intervalMs)
{
    StopSampling();
    m_Label = label;
    m_Samples.clear();
    m_Stop = false;

    m_Thread = std::thread([this, intervalMs]()
    {
        const auto begin = std::chrono::steady_clock::now();

        while (!m_Stop.load())
        {
            m_Samples.push_back(TakeSample(std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
                .count()));
            std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs));
        }
    });
}

TraceSummary NvmlMonitor::Stop(const double tailSeconds)
{
    StopSampling();

    for (const auto& sample : m_Samples)
    {
        m_AllSamples.emplace_back(m_Label, sample);
    }

    return Summarize(m_Samples, tailSeconds);
}

void NvmlMonitor::WriteTraces(const std::string& file) const
{
    std::ofstream output(file);
    output << "configuration,time_s,power_w,instant_power_w,average_power_w,energy_j,sm_clock_mhz,memory_clock_mhz,"
        << "temperature_c,fan_pct,reasons\n";

    for (const auto& [label, sample] : m_AllSamples)
    {
        output << label << "," << CsvValue(sample.timeS) << "," << CsvValue(sample.powerW) << ","
            << CsvValue(sample.instantPowerW) << "," << CsvValue(sample.averagePowerW) << ","
            << CsvValue(sample.energyJ, 16) << "," << CsvValue(sample.smClockMHz) << "," << CsvValue(sample.memoryClockMHz)
            << "," << CsvValue(sample.temperatureC) << "," << CsvValue(sample.fanPct) << ","
            << DescribeReasons(sample.reasons, false) << "\n";
    }
}

void NvmlMonitor::StopSampling()
{
    m_Stop = true;

    if (m_Thread.joinable())
    {
        m_Thread.join();
    }
}

TraceSample NvmlMonitor::TakeSample(const double timeS) const
{
    TraceSample sample;
    sample.timeS = timeS;
    unsigned int value = 0;
    unsigned long long large = 0;

    if (nvmlDeviceGetPowerUsage(m_Device, &value) == NVML_SUCCESS)
    {
        sample.powerW = static_cast<double>(value) / 1000.0;
    }

    // nvmlDeviceGetPowerUsage averages over 1 s on Ampere and newer (except GA100); the instant field shows the
    // current power. Both help to understand differences to the energy counter.
    nvmlFieldValue_t fields[2] = {};
    fields[0].fieldId = NVML_FI_DEV_POWER_INSTANT;
    fields[1].fieldId = NVML_FI_DEV_POWER_AVERAGE;

    if (nvmlDeviceGetFieldValues(m_Device, 2, fields) == NVML_SUCCESS)
    {
        if (fields[0].nvmlReturn == NVML_SUCCESS)
        {
            sample.instantPowerW = static_cast<double>(fields[0].value.uiVal) / 1000.0;
        }

        if (fields[1].nvmlReturn == NVML_SUCCESS)
        {
            sample.averagePowerW = static_cast<double>(fields[1].value.uiVal) / 1000.0;
        }
    }

    if (m_HasEnergyCounter && nvmlDeviceGetTotalEnergyConsumption(m_Device, &large) == NVML_SUCCESS)
    {
        sample.energyJ = static_cast<double>(large) / 1000.0;
    }

    if (nvmlDeviceGetClockInfo(m_Device, NVML_CLOCK_SM, &value) == NVML_SUCCESS)
    {
        sample.smClockMHz = value;
    }

    if (nvmlDeviceGetClockInfo(m_Device, NVML_CLOCK_MEM, &value) == NVML_SUCCESS)
    {
        sample.memoryClockMHz = value;
    }

    if (nvmlDeviceGetTemperature(m_Device, NVML_TEMPERATURE_GPU, &value) == NVML_SUCCESS)
    {
        sample.temperatureC = value;
    }

    if (nvmlDeviceGetFanSpeed(m_Device, &value) == NVML_SUCCESS)
    {
        sample.fanPct = value;
    }

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    if (nvmlDeviceGetCurrentClocksThrottleReasons(m_Device, &large) == NVML_SUCCESS)
    {
        sample.reasons = large;
    }
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

    return sample;
}

TraceSummary NvmlMonitor::Summarize(const std::vector<TraceSample>& samples, const double tailSeconds)
{
    TraceSummary summary;

    if (samples.empty())
    {
        return summary;
    }

    const double halfTime = samples.back().timeS / 2.0;
    const TraceSample* firstEnergy = nullptr;
    const TraceSample* lastEnergy = nullptr;

    // Means over the samples that have the value.
    struct Mean
    {
        double sum = 0.0;
        size_t count = 0;

        void Add(const double value)
        {
            if (!std::isnan(value))
            {
                sum += value;
                ++count;
            }
        }

        double Get() const
        {
            return count > 0 ? sum / static_cast<double>(count) : NaN;
        }
    };

    Mean power;
    Mean instant;
    Mean average;
    Mean clock;
    Mean memoryClock;
    Mean temperature;
    Mean fan;

    for (const auto& sample : samples)
    {
        if (sample.timeS < halfTime)
        {
            continue;
        }

        ++summary.samples;
        summary.reasons |= sample.reasons;
        power.Add(sample.powerW);
        instant.Add(sample.instantPowerW);
        average.Add(sample.averagePowerW);
        clock.Add(sample.smClockMHz);
        memoryClock.Add(sample.memoryClockMHz);
        temperature.Add(sample.temperatureC);
        fan.Add(sample.fanPct);

        if (!std::isnan(sample.smClockMHz))
        {
            summary.minClockMHz = std::isnan(summary.minClockMHz) ? sample.smClockMHz
                : std::min(summary.minClockMHz, sample.smClockMHz);
            summary.maxClockMHz = std::isnan(summary.maxClockMHz) ? sample.smClockMHz
                : std::max(summary.maxClockMHz, sample.smClockMHz);
        }

        if (!std::isnan(sample.temperatureC))
        {
            summary.maxTemperatureC = std::isnan(summary.maxTemperatureC) ? sample.temperatureC
                : std::max(summary.maxTemperatureC, sample.temperatureC);
        }

        if (!std::isnan(sample.energyJ))
        {
            firstEnergy = firstEnergy == nullptr ? &sample : firstEnergy;
            lastEnergy = &sample;
        }
    }

    summary.sampledPowerW = power.Get();
    summary.instantPowerW = instant.Get();
    summary.averagePowerW = average.Get();
    summary.meanClockMHz = clock.Get();
    summary.meanMemoryClockMHz = memoryClock.Get();
    summary.meanTemperatureC = temperature.Get();
    summary.meanFanPct = fan.Get();

    if (tailSeconds > 0.0)
    {
        Mean tail;

        for (const auto& sample : samples)
        {
            if (sample.timeS >= samples.back().timeS - tailSeconds)
            {
                tail.Add(sample.powerW);
            }
        }

        const double mean = tail.Get();
        double spread = 0.0;

        for (const auto& sample : samples)
        {
            if (sample.timeS >= samples.back().timeS - tailSeconds && !std::isnan(sample.powerW) && mean > 0.0)
            {
                spread = std::max(spread, std::abs(sample.powerW - mean) / mean * 100.0);
            }
        }

        summary.tailSpreadPct = mean > 0.0 ? spread : NaN;
    }

    if (firstEnergy != nullptr && lastEnergy != nullptr && lastEnergy->timeS > firstEnergy->timeS)
    {
        summary.counterPowerW = (lastEnergy->energyJ - firstEnergy->energyJ) / (lastEnergy->timeS - firstEnergy->timeS);
    }

    return summary;
}
