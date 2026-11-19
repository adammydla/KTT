// What every measurement on the GPU needs, for a probe run and a validation run alike: the idle reading before KTT
// creates its CUDA context, the tuner, the GPU as KTT and MT4G describe it, the NVML monitor, the launcher of the probe
// kernels and the rows measured so far.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Ktt.h>

#include "Hardware.h"
#include "NvmlMonitor.h"
#include "Options.h"
#include "ProbeKernels.h"
#include "Rows.h"

class GpuSession
{
public:
    // Measures the idle power without a CUDA context (--idle-s), then creates the tuner and reads the GPU's sizes.
    explicit GpuSession(const Options& options);

    // Opens the NVML monitor, checks that KTT reads the power of the same GPU, and locks the clock with --lock-clock.
    // Separate from the constructor so that the setup can be printed in between.
    void OpenMonitor();

    // A probe kernel of `definition` whose launcher sets the arguments of a configuration with `configure`, records the
    // kernel's own counts (and with `checkPlacement` the SM slots of its blocks) and the NVML trace.
    ktt::KernelId CreateKernel(const std::string& name, const ktt::KernelDefinitionId definition, const Configure& configure,
        const bool checkPlacement, const ktt::ArgumentId counts);

    // NVML readings without a kernel for --idle-s seconds, as a row of level "idle" (with the monitor only).
    void MeasureIdle(const std::string& label);

    // Compares the clocks of the warm-up under load with the expected ones (--expect-clock, --lock-clock): WarmUp calls it.
    void CheckClocks() const;

    // Runs `kernel` for --warmup-s seconds without keeping its results.
    void WarmUp(const ktt::KernelId kernel);

    // Measures every configuration of `kernel` and adds a row for each; `probe`, `operation` and `level` name the rows
    // whose kernel parameters do not (see MakeRow).
    void Run(const ktt::KernelId kernel, const std::string& label, const std::string& probe, const std::string& operation,
        const std::string& level);

    // Writes the NVML trace of all configurations (with the monitor only).
    void WriteTrace() const;

    ktt::Tuner& GetTuner();
    const Hardware& GetHardware() const;
    const Footprints& GetFootprints() const;
    const NvmlMonitor* GetMonitor() const;
    NvmlMonitor* GetMonitor();
    // Blocks of every probe launch: blocks per SM on every SM.
    uint64_t GetBlocks() const;
    std::vector<Row>& GetRows();

private:
    const Options& m_Options;
    std::vector<Row> m_Rows; // first, so that the idle reading without a context is taken before the tuner exists
    ktt::Tuner m_Tuner;
    Hardware m_Hardware;
    Footprints m_Footprints;
    // After the tuner, so that it is destroyed first: that resets a locked clock before KTT shuts down.
    std::unique_ptr<NvmlMonitor> m_Monitor;
    StatsMap m_Stats;
};

// The sleep baselines of a run, measured in turn between the groups of probes ("sleep_0", "sleep_1", ...). Their
// temperatures span the run, which gives the temperature model of the baseline.
class SleepBaselines
{
public:
    SleepBaselines(GpuSession& session, const ktt::KernelDefinitionId sleep, const ktt::ArgumentId counts,
        const size_t count);

    void RunNext();

private:
    GpuSession& m_Session;
    std::vector<ktt::KernelId> m_Kernels;
    size_t m_Next = 0;
};
