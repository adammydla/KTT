/******************************************************************************
 * Energy probes: the energy of one memory access (loads, stores and atomics served by L1, L2, DRAM, shared and constant
 * memory) and of one arithmetic operation on an NVIDIA GPU, measured with KTT's NVML power readings, and from them an
 * energy model per CUPTI event that prices a kernel configuration from its counters and its time (EnergyModel.h), so
 * that a tuner can pick the configuration that uses the least energy without measuring power.
 *
 * Every probe keeps one part of the GPU busy on all SMs for a fixed time while KTT measures the board power, and counts
 * the accesses it issued. Energy per access is the slope of the power above a baseline over the access rate, across
 * duties. The baseline is a kernel whose warps sleep: the GPU is awake at the same clock but issues almost nothing.
 *
 * Modes: the probes (default; --profile for their CUPTI counters), --validate (example kernels measured the same way,
 * for the model's uncertainty), --tune and --predict (the model used as a tuner uses it), --analyze, --dry-run.
 * See README.md. Requires KTT built with --power-usage and --profiling=cupti.
 */

#include <iostream>
#include <stdexcept>

#include "Analysis.h"
#include "Hardware.h"
#include "Options.h"
#include "ProbePlan.h"
#include "ProbeRun.h"
#include "Report.h"
#include "Rows.h"
#include "Tune.h"
#include "Validation.h"

namespace
{

// The GPU from the MT4G file alone, for the modes that do not touch one.
Hardware HardwareFromFile(const Options& options, const std::string& mode)
{
    if (options.mt4gFile.empty())
    {
        throw std::runtime_error(mode + " needs --mt4g");
    }

    Hardware hardware;
    ApplyOverrides(hardware, options);
    return hardware;
}

// --analyze: the analysis of an earlier run again, from its CSV.
int AnalyzeRun(const Options& options)
{
    const Hardware hardware = HardwareFromFile(options, "--analyze");
    std::vector<Row> rows = ReadCsv(options.analyzePrefix + ".csv");
    ReportProbeRun(rows, FitBaseline(rows), hardware, options, nullptr);
    return 0;
}

// --dry-run: the plan, without touching a GPU.
int DryRun(const Options& options)
{
    const Hardware hardware = HardwareFromFile(options, "--dry-run");
    const Footprints footprints = ChooseFootprints(hardware, options);
    DescribeSetup(hardware, options, footprints, ProbeGroups(options, hardware, footprints));
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;

    try
    {
        options = ParseOptions(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << "\n";
        PrintUsage();
        return 1;
    }

    try
    {
        if (!options.analyzePrefix.empty())
        {
            return AnalyzeRun(options);
        }

        if (!options.predictCsv.empty())
        {
            return RunPredict(options);
        }

        if (!options.tune.empty())
        {
            return RunTune(options);
        }

        if (!options.recountCsv.empty())
        {
            return RunRecount(options);
        }

        if (options.dryRun)
        {
            return DryRun(options);
        }

        return options.validate.empty() ? RunProbes(options) : RunValidation(options);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
