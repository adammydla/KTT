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

#include <memory>

#include <Ktt.h>

#include "Analysis.h"
#include "GpuSession.h"
#include "NvmlMonitor.h"
#include "CounterTable.h"
#include "ProbeKernels.h"
#include "Report.h"
#include "Rows.h"
#include "Validation.h"
#include "ValidationKernels.h"

namespace
{

// What the launcher of an example kernel measured in the power pass.
struct ValidationStats
{
    uint64_t launches = 0; // timed launches
    double wallS = 0.0;    // host time of all timed launches
    double accesses = NaN; // control kernel: accesses per launch, as the probe counted them
    std::optional<TraceSummary> trace;
};

using ValidationStatsMap = std::map<std::string, ValidationStats>;

struct ValidationRow
{
    std::string kernel;
    uint64_t repeat = 0; // --validate-repeat: which power measurement of the configuration
    ktt::KernelConfiguration configuration;
    bool valid = false;
    uint64_t launches = 0;
    double accesses = NaN; // control kernel: accesses per launch (the probe's own count in the power pass)
    double wallS = NaN;   // host time per launch: kernel plus KTT's synchronisation and NVML queries
    double kernelS = NaN; // GPU time per launch (KTT)
    double powerW = NaN;  // nvmlDeviceGetPowerUsage over the second half of the timed launches, as for the probes
    double instantPowerW = NaN;
    double counterPowerW = NaN;
    double smClockMHz = NaN;
    double memoryClockMHz = NaN;
    double temperatureC = NaN;
    double fanPct = NaN;
    std::string limiters;
    double baselineW = NaN;
    double dynamicJ = NaN; // per launch
    double energyJ = NaN;  // per launch: baseline over the kernel time plus the dynamic energy
    std::string note;
    std::map<std::string, double> counters; // per launch, from the counter pass
};

std::string ConfigurationText(const ktt::KernelConfiguration& configuration)
{
    std::string text;

    for (const auto& pair : configuration.GetPairs())
    {
        text += (text.empty() ? "" : " ") + pair.GetName() + "=" + pair.GetValueString();
    }

    return text;
}

// The baseline model is written with every row (mean power, its temperature, slope; slope 0 = no temperature model), so
// that energy_model.py can move the baseline with another slope when this run's sleeps had too narrow a temperature range.
void WriteValidationCsv(const std::vector<ValidationRow>& rows, const BaselineModel& baseline, const std::string& file)
{
    std::set<std::string> counterNames;

    for (const auto& row : rows)
    {
        for (const auto& counter : row.counters)
        {
            counterNames.insert(counter.first);
        }
    }

    std::ofstream output(file);
    output << "kernel,configuration,valid,launches,accesses_per_launch,wall_s,kernel_s,busy,power_w,instant_power_w,counter_power_w,sm_clock_mhz,"
        << "memory_clock_mhz,temperature_c,fan_pct,clock_limiters,baseline_w,baseline_mean_w,baseline_temperature_c,"
        << "baseline_slope_w_per_c,dynamic_energy_j,energy_j,repeat,note";

    for (const auto& name : counterNames)
    {
        output << "," << name;
    }

    output << "\n";

    for (const auto& row : rows)
    {
        output << CsvText(row.kernel) << "," << CsvText(ConfigurationText(row.configuration)) << "," << row.valid << "," << row.launches << ","
            << CsvValue(row.accesses) << "," << CsvValue(row.wallS) << "," << CsvValue(row.kernelS) << "," << CsvValue(row.kernelS / row.wallS) << ","
            << CsvValue(row.powerW) << "," << CsvValue(row.instantPowerW) << "," << CsvValue(row.counterPowerW) << ","
            << CsvValue(row.smClockMHz) << "," << CsvValue(row.memoryClockMHz) << "," << CsvValue(row.temperatureC) << ","
            << CsvValue(row.fanPct) << "," << CsvText(row.limiters) << "," << CsvValue(row.baselineW) << "," << CsvValue(baseline.meanW) << ","
            << CsvValue(baseline.meanTemperatureC) << "," << CsvValue(baseline.temperatureModel ? baseline.slopeWPerC : 0.0)
            << "," << CsvValue(row.dynamicJ) << "," << CsvValue(row.energyJ) << "," << row.repeat << "," << CsvText(row.note);

        for (const auto& name : counterNames)
        {
            const auto iterator = row.counters.find(name);
            output << "," << (iterator != row.counters.cend() ? CsvValue(iterator->second) : "");
        }

        output << "\n";
    }
}

// The power pass runs the kernel once untimed (compilation, first touch of the data), then back to back for --validate-ms
// while the NVML trace runs. Every launch through KTT ends with a synchronisation and NVML queries, so the GPU idles between
// launches; the dynamic energy therefore uses the host time per launch (the idle gaps are at about baseline power), and the
// busy fraction (GPU time / host time) is reported. The counter pass runs one launch per profiling pass. The control kernel
// gets its arguments before every launch, and its own counts are read after the timed launches.
void SetValidationLauncher(ktt::Tuner& tuner, const ValidationKernel& kernel, const Options& options, ValidationStatsMap& stats,
    NvmlMonitor* monitor)
{
    const ktt::KernelDefinitionId definition = kernel.definition;
    const std::string name = kernel.name;
    const double minimumS = static_cast<double>(options.validateMs) * 1e-3;
    const uint64_t traceIntervalMs = options.traceIntervalMs;
    const auto configure = kernel.configure;
    const auto countsId = kernel.counts;
    const uint64_t blocks = kernel.blocks;

    tuner.SetLauncher(kernel.kernel, [definition, name, minimumS, traceIntervalMs, configure, countsId, blocks, &stats,
        monitor](ktt::ComputeInterface& interface)
    {
        const auto& configuration = interface.GetCurrentConfiguration();

        if (configure)
        {
            configure(interface, configuration);
        }

        if (interface.GetProfiling(definition))
        {
            interface.RunKernelWithProfiling(definition);
            return;
        }

        interface.RunKernel(definition);
        ValidationStats launch;
        // The probe adds its accesses and launches to the buffer on every launch, so it is cleared after the untimed one.
        std::vector<uint64_t> counts(countsId.has_value() ? 3 * blocks + 1 : 0, 0);

        if (countsId.has_value())
        {
            interface.UpdateBuffer(countsId.value(), counts.data(), counts.size() * sizeof(uint64_t));
        }

        if (monitor != nullptr)
        {
            monitor->Start(TraceLabel(name, configuration), traceIntervalMs);
        }

        const auto begin = std::chrono::steady_clock::now();

        do
        {
            interface.RunKernel(definition);
            ++launch.launches;
            launch.wallS = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        }
        while (launch.wallS < minimumS);

        if (monitor != nullptr)
        {
            launch.trace = monitor->Stop();
        }

        if (countsId.has_value())
        {
            interface.DownloadBuffer(countsId.value(), counts.data(), counts.size() * sizeof(uint64_t));
            uint64_t accesses = 0;

            for (uint64_t i = 0; i < blocks; ++i)
            {
                accesses += counts[i];
            }

            // The probe's own launch count must be the launcher's.
            if (counts[3 * blocks] == launch.launches)
            {
                launch.accesses = static_cast<double>(accesses) / static_cast<double>(launch.launches);
            }
        }

        stats[name + "|" + configuration.GetString()] = launch;
    });
}

ValidationRow MakeValidationRow(const ktt::KernelResult& result, const ValidationKernel& kernel,
    const ValidationStatsMap& stats)
{
    ValidationRow row;
    row.kernel = kernel.name;
    row.configuration = result.GetConfiguration();
    row.valid = result.IsValid();

    const auto iterator = stats.find(kernel.name + "|" + row.configuration.GetString());
    const auto& computations = result.GetResults();

    if (!row.valid || iterator == stats.cend() || computations.size() < 2)
    {
        row.valid = false;
        AppendNote(row.note, "run failed");
        return row;
    }

    // KTT keeps one result per launch: the untimed launch first, then the timed ones.
    const ValidationStats& launch = iterator->second;
    double kernelNs = 0.0;

    for (size_t i = 1; i < computations.size(); ++i)
    {
        kernelNs += static_cast<double>(computations[i].GetDuration());
    }

    if (computations.size() - 1 != launch.launches)
    {
        AppendNote(row.note, "launch count mismatch");
    }

    row.launches = launch.launches;
    row.accesses = launch.accesses;
    row.wallS = launch.wallS / static_cast<double>(launch.launches);
    row.kernelS = kernelNs * 1e-9 / static_cast<double>(computations.size() - 1);

    if (launch.trace.has_value())
    {
        row.powerW = launch.trace->sampledPowerW;
        row.instantPowerW = launch.trace->instantPowerW;
        row.counterPowerW = launch.trace->counterPowerW;
        row.smClockMHz = launch.trace->meanClockMHz;
        row.memoryClockMHz = launch.trace->meanMemoryClockMHz;
        row.temperatureC = launch.trace->meanTemperatureC;
        row.fanPct = launch.trace->meanFanPct;
        row.limiters = DescribeReasons(launch.trace->reasons, true);
    }
    else
    {
        AppendNote(row.note, "no NVML trace");
    }

    return row;
}

// Counter pass, energies per launch and output of --validate. The counter pass is separate from the power pass because
// profiling replays the kernel and raises power.
int FinishValidation(ktt::Tuner& tuner, const std::vector<ValidationKernel>& kernels, std::vector<ValidationRow>& rows,
    const std::vector<Row>& baselineRows, const BaselineModel& baseline, const Options& options)
{
    tuner.SetProfiling(true);

    if (tuner.GetProfiling())
    {
        tuner.SetProfilingCounters(ModelCounters(options.counterTable, options.studyCounters));

        for (auto& row : rows)
        {
            // A configuration's counters do not change between its power measurements: collected once.
            if (!row.valid || row.repeat > 0)
            {
                continue;
            }

            const auto kernel = std::find_if(kernels.cbegin(), kernels.cend(), [&row](const ValidationKernel& candidate)
            {
                return candidate.name == row.kernel;
            });

            // Tune repeats a configuration until all profiling passes ran; Run does one pass, so it is repeated here (an
            // unfinished profiling session makes the next kernel's session fail). The last pass carries the counters.
            tuner.SetCompilerOptions(kernel->compilerOptions);
            ktt::KernelResult result = tuner.Run(kernel->kernel, row.configuration, {});

            while (result.IsValid() && result.HasRemainingProfilingRuns())
            {
                result = tuner.Run(kernel->kernel, row.configuration, {});
            }

            if (result.IsValid() && !result.GetResults().empty() && result.GetResults()[0].HasProfilingData())
            {
                for (const auto& counter : result.GetResults()[0].GetProfilingData().GetCounters())
                {
                    row.counters[CanonicalCounter(counter.GetName())] = CounterValue(counter);
                }
            }
            else
            {
                AppendNote(row.note, "no counters");
            }
        }

        tuner.SetCompilerOptions("");

        for (auto& row : rows)
        {
            if (row.repeat == 0)
            {
                continue;
            }

            const auto first = std::find_if(rows.cbegin(), rows.cend(), [&row](const ValidationRow& other)
            {
                return other.repeat == 0 && other.kernel == row.kernel
                    && other.configuration.GetString() == row.configuration.GetString();
            });

            if (first != rows.cend())
            {
                row.counters = first->counters;
            }
        }
    }
    else
    {
        std::cout << "WARNING: profiling could not be enabled (build KTT with --profiling=cupti); the validation CSV has no "
            << "counters\n";
    }

    std::cout << "\nValidation kernels (per launch; baseline " << Fixed(baseline.meanW, 1) << " W at "
        << Fixed(baseline.meanTemperatureC, 1) << " C, " << Fixed(baseline.slopeWPerC, 2) << " W/C):\n"
        << "  " << std::left << std::setw(10) << "kernel" << std::right << std::setw(10) << "t [ms]" << std::setw(7) << "busy"
        << std::setw(9) << "P [W]" << std::setw(9) << "P-base" << std::setw(10) << "E [mJ]" << std::setw(7) << "T [C]"
        << "  configuration\n";

    for (auto& row : rows)
    {
        if (row.valid && !std::isnan(row.powerW))
        {
            row.baselineW = baseline.At(row.temperatureC);
            row.dynamicJ = (row.powerW - row.baselineW) * row.wallS;
            row.energyJ = row.baselineW * row.kernelS + row.dynamicJ;

            if (!std::isnan(row.smClockMHz) && !std::isnan(baseline.clockMHz)
                && std::abs(row.smClockMHz - baseline.clockMHz) > ClockTolerance * baseline.clockMHz)
            {
                AppendNote(row.note, "SM clock " + Fixed(row.smClockMHz, 0) + " MHz differs from the baseline's");
            }
        }

        std::cout << "  " << std::left << std::setw(10) << row.kernel << std::right << std::setw(10)
            << Fixed(row.kernelS * 1e3, 2) << std::setw(6) << Fixed(100.0 * row.kernelS / row.wallS, 0) << "%"
            << std::setw(9) << Fixed(row.powerW, 1) << std::setw(9) << Fixed(row.powerW - row.baselineW, 1) << std::setw(10)
            << Fixed(row.energyJ * 1e3, 1) << std::setw(7) << Fixed(row.temperatureC, 0) << "  "
            << ConfigurationText(row.configuration) << (row.note.empty() ? "" : "  [" + row.note + "]") << "\n";
    }

    WriteValidationCsv(rows, baseline, options.outputPrefix + "_validation.csv");
    WriteCsv(baselineRows, options.outputPrefix + ".csv");
    std::cout << "\nWrote " << options.outputPrefix << "_validation.csv (one row per configuration), " << options.outputPrefix
        << ".csv (idle and sleep rows) and " << options.outputPrefix << "_trace.csv. Predict and compare with energy_model.py "
        << "(probe power run, profiled probe run, this file).\n";
    return 0;
}

// The sleep and spin kernels of a validation run, with the arguments they need (the probes' other arguments and their
// 256 MiB payload are not needed).
struct BaselineKernels
{
    ktt::KernelDefinitionId sleep;
    ktt::KernelDefinitionId spin;
    ktt::ArgumentId counts;
    Configure configure;
};

BaselineKernels AddBaselineKernels(ktt::Tuner& tuner, const Options& options, const uint64_t blocks,
    const uint32_t multiprocessors)
{
    const std::vector<uint32_t> sink(blocks * options.blockSize, 0);
    const std::vector<uint64_t> counts(3 * blocks + 1, 0);
    const ktt::ArgumentId sinkId = tuner.AddArgumentVector(sink, ktt::ArgumentAccessType::WriteOnly);
    const ktt::ArgumentId countsId = tuner.AddArgumentVector(counts, ktt::ArgumentAccessType::ReadWrite);
    const ktt::ArgumentId activeSmsId = tuner.AddArgumentScalar(AllSms);
    const ktt::ArgumentId durationId = tuner.AddArgumentScalar(static_cast<uint64_t>(options.launchMs * 1'000'000));
    const std::string source = KernelSource(options.kernelDirectory, "Baseline.cu");
    const ktt::DimensionVector grid(multiprocessors);
    const ktt::DimensionVector block(1);

    BaselineKernels kernels;
    kernels.sleep = tuner.AddKernelDefinition("probeSleep", source, grid, block);
    kernels.spin = tuner.AddKernelDefinition("probeSpin", source, grid, block);
    kernels.counts = countsId;
    tuner.SetArguments(kernels.sleep, {sinkId, countsId, activeSmsId, durationId});
    tuner.SetArguments(kernels.spin, {sinkId, countsId, durationId});
    kernels.configure = [activeSmsId](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        const uint32_t activeSms = ActiveSms(configuration);
        interface.UpdateScalarArgument(activeSmsId, &activeSms);
    };
    return kernels;
}

// The control kernel (--validate mixed): the mixed-intensity probe with 16-byte loads at the three footprints and 0 to 1024
// FFMAs per load, in launches of 5 and 50 ms, measured like the example kernels (back-to-back launches through KTT, host
// time, this run's sleep baseline, counters from the counter pass). A probe run with --mixed measured the same kernel the
// probes' way (200 ms launches inside KTT's precise measurement, GPU time, that run's baseline), and the model predicts the
// probes within a few percent. So if the model misses the control here as it misses the example kernels, the measurement
// of the validation run is the cause; energy_model.py compares the two measurements of every configuration.
ValidationKernel AddMixedControl(GpuSession& session, const Options& options, std::vector<uint32_t>& payload)
{
    ktt::Tuner& tuner = session.GetTuner();
    const Footprints& footprints = session.GetFootprints();
    // The probes' arguments (random data only, no constant array) and the mixed probe's definition. Of the settings only
    // the mixed probe's are used, so the other definitions are not needed.
    Options probeOptions = options;
    probeOptions.payloads = {PayloadRandom};
    probeOptions.constant = false;
    payload = RandomPayload(footprints.dram);
    const ProbeArguments arguments = AddProbeArguments(tuner, payload, probeOptions, session.GetBlocks());
    ProbeDefinitions definitions{};
    definitions.mixed = AddMixedDefinition(tuner, arguments, probeOptions, session.GetHardware().multiprocessors);
    const ProbeSettings settings = MakeProbeSettings(arguments, definitions, footprints, ProbeTextures{});

    const ktt::KernelDefinitionId definition = definitions.mixed.value();
    const ktt::KernelId kernel = tuner.CreateSimpleKernel("MixedControl", definition);
    AddProbeLaunchParameters(tuner, kernel, definition, probeOptions);
    tuner.AddParameter(kernel, "FOOTPRINT", std::vector<uint64_t>{footprints.l1, footprints.l2, footprints.dram});
    tuner.AddParameter(kernel, "WIDTH", std::vector<uint64_t>{DefaultWidth});
    tuner.AddParameter(kernel, "GRID_STRIPES", std::vector<uint64_t>{footprints.gridStripes});
    tuner.AddParameter(kernel, "MIX_FMAS", MixedControlFmas);
    tuner.AddParameter(kernel, "LAUNCH_MS", MixedControlLaunchMs);

    ValidationKernel control{MixedControlName, kernel, definition, ""};
    const Configure mixed = settings.mixed;
    const ktt::ArgumentId duration = arguments.duration;
    control.configure = [mixed, duration](ktt::ComputeInterface& interface, const ktt::KernelConfiguration& configuration)
    {
        mixed(interface, configuration);
        const uint64_t durationNs = ParameterValue(configuration, "LAUNCH_MS").value_or(MixedControlLaunchMs.back())
            * 1'000'000;
        interface.UpdateScalarArgument(duration, &durationNs);
    };
    control.counts = arguments.counts;
    control.blocks = session.GetBlocks();
    return control;
}

} // namespace

int RunValidation(const Options& options)
{
    GpuSession session(options);
    std::cout << "Validation: " << options.validateConfigs << " random configurations each of";

    for (const auto& name : options.validate)
    {
        std::cout << " " << name;
    }

    std::cout << ", " << options.validateMs << " ms per configuration, a sleep baseline before and after every kernel, "
        << "then the same configurations once more with CUPTI counters\n";
    session.OpenMonitor();

    ktt::Tuner& tuner = session.GetTuner();
    const BaselineKernels baselines = AddBaselineKernels(tuner, options, session.GetBlocks(),
        session.GetHardware().multiprocessors);
    const ktt::KernelId warmup = session.CreateKernel("Warmup", baselines.spin, Configure{}, false, baselines.counts);
    // A baseline before the first example kernel and one after every one (in every round of --validate-repeat).
    SleepBaselines sleeps(session, baselines.sleep, baselines.configure, baselines.counts,
        options.validate.size() * options.validateRepeat + 1);
    std::vector<ValidationKernel> kernels;
    ValidationStatsMap stats;
    std::vector<uint32_t> payload; // the control kernel's data; KTT keeps a reference to it

    SetValidationInputs({options.validateData, options.validateSizeScale, options.accelwattchDirectory});

    for (const auto& name : options.validate)
    {
        kernels.push_back(name == MixedControlName ? AddMixedControl(session, options, payload)
            : AddValidationKernel(tuner, name, kernelPrefix + "../Examples/"));
        SetValidationLauncher(tuner, kernels.back(), options, stats, session.GetMonitor());
    }

    session.MeasureIdle("idle-start");
    session.WarmUp(warmup);
    sleeps.RunNext();
    std::vector<ValidationRow> validationRows;

    // Power pass: random configurations of every example kernel, then a baseline. KTT's compiler options are per tuner, so
    // they are switched for every kernel (and back for the sleep kernel).
    for (const auto& kernel : kernels)
    {
        std::cout << "\n=== Validation kernel " << kernel.name << " ===" << std::endl;
        tuner.SetCompilerOptions(kernel.compilerOptions);
        if (options.validateSeed.has_value())
        {
            tuner.SetSearcher(kernel.kernel, std::make_unique<SeededSearcher>(options.validateSeed.value()));
        }
        else
        {
            tuner.SetSearcher(kernel.kernel, std::make_unique<ktt::RandomSearcher>());
        }

        const auto results = tuner.Tune(kernel.kernel, std::make_unique<ktt::ConfigurationCount>(options.validateConfigs));
        tuner.SaveResults(results, options.outputPrefix + "_" + kernel.name, ktt::OutputFormat::JSON);
        tuner.SetCompilerOptions("");

        for (const auto& result : results)
        {
            validationRows.push_back(MakeValidationRow(result, kernel, stats));
        }

        sleeps.RunNext();
    }

    // --validate-repeat: the same configurations again, in rounds over all kernels, so that a configuration's repeats are
    // spread over the run (its drift shows as a difference between the first and the last round).
    const size_t firstRound = validationRows.size();

    for (uint64_t repeat = 1; repeat < options.validateRepeat; ++repeat)
    {
        for (const auto& kernel : kernels)
        {
            std::cout << "\n=== Validation kernel " << kernel.name << ", repeat " << repeat << " ===" << std::endl;
            tuner.SetCompilerOptions(kernel.compilerOptions);

            for (size_t i = 0; i < firstRound; ++i)
            {
                if (validationRows[i].kernel != kernel.name || !validationRows[i].valid)
                {
                    continue;
                }

                const ktt::KernelResult result = tuner.Run(kernel.kernel, validationRows[i].configuration, {});
                validationRows.push_back(MakeValidationRow(result, kernel, stats));
                validationRows.back().repeat = repeat;
            }

            tuner.SetCompilerOptions("");
            sleeps.RunNext();
        }
    }

    session.MeasureIdle("idle-end");
    session.WriteTrace();

    std::vector<Row>& rows = session.GetRows();
    const BaselineModel baseline = FitBaseline(rows);

    if (baseline.runs == 0)
    {
        PrintRows(rows);
        WriteCsv(rows, options.outputPrefix + ".csv");
        std::cout << "\nNo power data was collected. Rebuild KTT with --power-usage and run on a GPU that reports power "
            << "through NVML.\n";
        return 1;
    }

    return FinishValidation(tuner, kernels, validationRows, rows, baseline, options);
}
