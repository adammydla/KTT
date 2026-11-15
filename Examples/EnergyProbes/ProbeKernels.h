// The probes as KTT kernels: the sources, arguments and definitions, what each probe sets for a configuration, the
// launcher that runs a configuration and collects the kernel's own counts and the NVML trace, and the row of a result.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <Ktt.h>
#include "EnergyTuner.h"
#include "Hardware.h"
#include "NvmlMonitor.h"
#include "Options.h"
#include "ProbePlan.h"
#include "Rows.h"

// Sets the runtime arguments and buffers of a definition for one configuration.
using Configure = std::function<void(ktt::ComputeInterface&, const ktt::KernelConfiguration&)>;

std::optional<uint64_t> ParameterValue(const ktt::KernelConfiguration& configuration, const std::string& name);

// Short, comma-free label of a configuration for the trace file, e.g. "global LOAD_OP=0 FOOTPRINT=32768".
std::string TraceLabel(const std::string& kernelName, const ktt::KernelConfiguration& configuration);

// The program a probe kernel is compiled from: kernels/Common.cuh followed by the probe's file in `directory`, with #line
// directives so that compiler messages name the right file. Every probe file is a program of its own.
std::string KernelSource(const std::string& directory, const std::string& file);

// The sleep kernel (the baseline every energy is measured against) and the spin kernel (the warm-up), with their
// arguments: what a probe run and a validation run both need.
struct BaselineKernels
{
    ktt::KernelDefinitionId sleep;
    ktt::KernelDefinitionId spin;
    ktt::ArgumentId sink;      // one word per thread, so that the compiler keeps the accesses
    ktt::ArgumentId counts;    // per-block accesses and active time, SM slots, launches
    ktt::ArgumentId activeSms; // every SM works (the kernels can run on fewer)
    ktt::ArgumentId duration;  // of one launch
};

BaselineKernels AddBaselineKernels(ktt::Tuner& tuner, const Options& options, const uint64_t blocks,
    const uint32_t multiprocessors);

// A KTT kernel of `definition` with the probes' launch parameters (BLOCK_SIZE, BLOCKS_PER_SM, UNROLL, BURST_ACCESSES;
// the grid is BLOCKS_PER_SM blocks of BLOCK_SIZE threads per SM). Its launcher sets the arguments of a configuration
// with `configure`, records the kernel's own counts (with `checkPlacement` also the SM slots of its blocks) in `stats`
// and traces the first pass with `monitor`.
ktt::KernelId CreateProbeKernel(ktt::Tuner& tuner, const std::string& name, const ktt::KernelDefinitionId definition,
    const Options& options, const ktt::ArgumentId counts, const uint64_t blocks, const Configure& configure,
    const bool checkPlacement, StatsMap& stats, NvmlMonitor* monitor);

// The row of a result: what the configuration's parameters say about the probe (operation, level, footprint, width,
// duty), else `operation` and `level`, and what was measured.
Row MakeRow(const ktt::KernelResult& result, const std::string& probe, const std::string& operation,
    const std::string& level, const StatsMap& stats, const Hardware& hardware, const Footprints& footprints);

std::vector<ktt::KernelResult> RunProbe(ktt::Tuner& tuner, const ktt::KernelId kernel, const std::string& label,
    const std::string& outputPrefix, const ktt::PreciseMeasurementParameters& precise);

// The probe kernels of a run: their data, arguments and definitions. Must outlive the measurements (KTT references the
// payload) and be destroyed before the tuner (the misaligned probes' buffer lives in the tuner's CUDA context).
class ProbeSet
{
public:
    ProbeSet(ktt::Tuner& tuner, const Options& options, const Footprints& footprints, const BaselineKernels& baseline,
        const uint32_t multiprocessors);
    ~ProbeSet();
    ProbeSet(const ProbeSet&) = delete;
    ProbeSet& operator=(const ProbeSet&) = delete;

    ktt::KernelDefinitionId Definition(const ProbeKernel kernel) const;

    // What the launcher of a probe sets for a configuration.
    Configure Settings(const ProbeKernel kernel) const;

private:
    std::vector<uint32_t> m_Payload; // random, so that bits toggle as in real data
    uint64_t m_OffsetBuffer = 0;     // the payload once more, for the misaligned probes (a CUdeviceptr)
    std::map<ProbeKernel, ktt::KernelDefinitionId> m_Definitions;
    std::map<ProbeKernel, Configure> m_Settings;
};
