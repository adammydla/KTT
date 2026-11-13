/******************************************************************************
 * Energy probes -- prototype for estimating the energy of a single access served
 * by L1, L2, DRAM or shared memory (loads and stores), and of one arithmetic
 * operation (FFMA, DFMA, IMAD, HFMA2, HMMA).
 *
 * Every probe keeps one part of the GPU busy on all SMs for a fixed time. KTT
 * measures board power (NVML) while the probe is repeated until the reading is
 * stable, and the kernel counts the accesses it issued. Energy per access is
 * (P_probe - P_baseline) * t / accesses. The baseline is a kernel whose warps
 * sleep: the GPU is awake at the same clock but issues almost nothing.
 *
 * Optional studies, following published GPU energy models:
 *   - zero vs random data: control vs datapath energy (Antepara et al., SC '25)
 *   - access widths of 4, 8 and 16 bytes: energy per request vs per byte
 *     (Wattchmen, ICS '26)
 *   - number of active SMs: power of awake and active SMs, chip-wide power
 *     (AccelWattch, MICRO '21)
 *   - idle power without any kernel: constant vs static power (Wattchmen)
 *
 * The program also talks to NVML directly: it can lock the GPU clock, records a
 * power / clock / temperature trace for every configuration, reports why the GPU
 * limited its clock, and cross-checks KTT's power with NVML's energy counter and
 * its instant and 1 s average power fields.
 *
 * Cache sizes are taken from an MT4G result file (--mt4g), otherwise from
 * conservative defaults. Requires KTT built with --power-usage.
 * See README.md in this folder.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Analysis.h"
#include "Hardware.h"
#include "Options.h"
#include "ProbePlan.h"
#include "ProbeRun.h"
#include "Report.h"
#include "Rows.h"
#include "Studies.h"
#include "Tune.h"
#include "Validation.h"

namespace
{

// --analyze: the analysis of an earlier run again, from its CSV. Cache sizes and the SM count come from the MT4G file as in
// --dry-run; the payloads, widths and duties of the studies from the rows themselves.
int AnalyzeRun(const Options& options)
{
    if (options.mt4gFile.empty())
    {
        throw std::runtime_error("--analyze needs --mt4g");
    }

    Hardware hardware;
    ApplyOverrides(hardware, options);
    const Footprints footprints = ChooseFootprints(hardware, options);
    std::vector<Row> rows = ReadCsv(options.analyzePrefix + ".csv");

    Options measured = options;
    std::set<uint64_t> payloads;
    std::set<uint64_t> widths;
    std::set<uint64_t> duties;

    for (const auto& row : rows)
    {
        if (!IsProbeRow(row))
        {
            continue;
        }

        payloads.insert(row.payload == PayloadName(PayloadZero) ? PayloadZero : PayloadRandom);
        duties.insert(row.duty);

        if (row.probe == "global" && row.operation == "ld")
        {
            widths.insert(row.width);
        }
    }

    // In the order ParseOptions leaves them: payloads random first, widths and duties from the largest.
    measured.payloads.assign(payloads.cbegin(), payloads.cend());
    measured.widths.assign(widths.crbegin(), widths.crend());
    measured.duties.assign(duties.crbegin(), duties.crend());

    const BaselineModel baseline = FitBaseline(rows);
    ReportProbeRun(rows, baseline, footprints, hardware, measured, nullptr);
    std::cout << "\nWrote " << measured.outputPrefix << ".csv, " << measured.outputPrefix << "_fits.csv and "
        << measured.outputPrefix << "_energy_table.json from " << options.analyzePrefix << ".csv\n";
    return 0;
}

// Prints the plan from an MT4G file without touching a GPU.
int DryRun(const Options& options)
{
    if (options.mt4gFile.empty())
    {
        throw std::runtime_error("--dry-run needs --mt4g");
    }

    Hardware hardware;
    ApplyOverrides(hardware, options);
    const Footprints footprints = ChooseFootprints(hardware, options);
    DescribeSetup(hardware, options, footprints, BuildPlan(options, footprints));
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

        if (options.dryRun && options.thermalCurve)
        {
            std::cout << "Thermal curve (--thermal-curve): sleep 120 s; spin 120 s, sleep 180 s; DRAM loads 120 s, sleep "
                << "180 s; 6 x (DRAM loads 20 s, sleep 10 s); plus 3 x --idle-s (no warm-up: the first sleep settles). About 15 min.\n";
            return 0;
        }

        if (options.dryRun)
        {
            return DryRun(options);
        }

        if (options.thermalCurve)
        {
            return RunThermalCurve(options);
        }

        return options.validate.empty() ? RunProbes(options) : RunValidation(options);
    }
    catch (const std::exception& error)
    {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
