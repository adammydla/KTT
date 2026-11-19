#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <variant>

#include <Ktt.h>

#include "Analysis.h"
#include "CounterTable.h"
#include "EnergyTuner.h"
#include "GpuSession.h"
#include "ProbeKernels.h"
#include "Validation.h"
#include "ValidationKernels.h"

namespace
{

// The energy per launch of a measured configuration: the baseline over the kernel's time plus the power above it over
// the wall time, plus what the GPU draws below the baseline between launches (gapBelowBaselineW) over that gap. Every
// launch through KTT waits for its NVML hook (a context synchronisation and a sampling thread, about 0.6 ms on an RTX
// 3090), and the GPU idles meanwhile at its context-idle power, about 6 W below the sleep baseline (measured with an
// empty kernel in every run). The kernel time is the counter gpu__time_duration (CounterKernelS); KTT's own time per
// launch, which includes the hook, is written as ktt_kernel_s. Before 2026-11-17 the kernel time was KTT's and the gap
// was taken at the baseline: a 0.2 ms kernel counted 0.8 ms, and its energy above the baseline 6 W x 0.6 ms too little.
// The busy fraction (kernel time / wall time) is written too. The baseline model is
// written with every row (mean power, its temperature, slope; slope 0 = no temperature model), so that the tools can
// move the baseline with another slope when this run's sleeps spanned too few degrees.
void WriteValidationCsv(const std::vector<TunedConfiguration>& rows, const std::vector<TraceSummary>& traces,
    const BaselineModel& baseline, const double gapBelowBaselineW, const std::string& file)
{
    std::set<std::string> counterNames;

    for (const auto& row : rows)
    {
        for (const auto& counter : row.counters)
        {
            counterNames.insert(CanonicalCounter(counter.first));
        }
    }

    std::ofstream output(file);
    output << "kernel,configuration,valid,wall_s,kernel_s,busy,power_w,sm_clock_mhz,memory_clock_mhz,temperature_c,fan_pct,"
        << "clock_limiters,baseline_w,baseline_mean_w,baseline_temperature_c,baseline_slope_w_per_c,dynamic_energy_j,"
        << "energy_j,ktt_kernel_s,gap_below_baseline_w,note";

    for (const auto& name : counterNames)
    {
        output << "," << name;
    }

    output << "\n";
    std::cout << "\nValidation kernels (per launch; baseline " << Fixed(baseline.meanW, 1) << " W at "
        << Fixed(baseline.meanTemperatureC, 1) << " C, " << Fixed(baseline.slopeWPerC, 2) << " W/C):\n  " << std::left
        << std::setw(14) << "kernel" << std::right << std::setw(10) << "t [ms]" << std::setw(7) << "busy" << std::setw(9)
        << "P [W]" << std::setw(9) << "P-base" << std::setw(10) << "E [mJ]" << std::setw(7) << "T [C]" << "  configuration\n";

    for (size_t i = 0; i < rows.size(); ++i)
    {
        const TunedConfiguration& row = rows[i];
        const TraceSummary& trace = traces[i];
        const bool valid = !std::isnan(row.measuredPowerW);
        const double baselineW = valid ? baseline.At(row.measuredTemperatureC) : NaN;
        const double dynamicJ = (row.measuredPowerW - baselineW) * row.measuredWallS
            + gapBelowBaselineW * (row.measuredWallS - row.measuredS);
        const double energyJ = baselineW * row.measuredS + dynamicJ;
        std::string note = row.note;

        if (valid && std::abs(trace.meanClockMHz - baseline.clockMHz) > ClockTolerance * baseline.clockMHz)
        {
            AppendNote(note, "SM clock " + Fixed(trace.meanClockMHz, 0) + " MHz differs from the baseline's");
        }

        output << CsvText(row.kernel) << "," << CsvText(row.configuration) << "," << valid << ","
            << CsvValue(row.measuredWallS) << "," << CsvValue(row.measuredS) << "," << CsvValue(row.busy) << ","
            << CsvValue(row.measuredPowerW) << "," << CsvValue(trace.meanClockMHz) << "," << CsvValue(trace.meanMemoryClockMHz)
            << "," << CsvValue(row.measuredTemperatureC) << "," << CsvValue(trace.meanFanPct) << ","
            << CsvText(DescribeReasons(trace.reasons, true)) << "," << CsvValue(baselineW) << "," << CsvValue(baseline.meanW)
            << "," << CsvValue(baseline.meanTemperatureC) << "," << CsvValue(baseline.temperatureModel ? baseline.slopeWPerC : 0.0)
            << "," << CsvValue(dynamicJ) << "," << CsvValue(energyJ) << "," << CsvValue(row.measuredKttS) << ","
            << CsvValue(gapBelowBaselineW) << "," << CsvText(note);

        for (const auto& name : counterNames)
        {
            double value = NaN;

            for (const auto& counter : row.counters)
            {
                value = CanonicalCounter(counter.first) == name ? counter.second : value;
            }

            output << "," << CsvValue(value);
        }

        output << "\n";
        std::cout << "  " << std::left << std::setw(14) << row.kernel << std::right << std::setw(10)
            << Fixed(row.measuredS * 1e3, 2) << std::setw(6) << Fixed(100.0 * row.busy, 0) << "%" << std::setw(9)
            << Fixed(row.measuredPowerW, 1) << std::setw(9) << Fixed(row.measuredPowerW - baselineW, 1) << std::setw(10)
            << Fixed(energyJ * 1e3, 1) << std::setw(7) << Fixed(row.measuredTemperatureC, 0) << "  " << row.configuration
            << (note.empty() ? "" : "  [" + note + "]") << "\n";
    }
}

// A configuration of `kernel` from its text "NAME=VALUE NAME=VALUE ..." (TunedConfiguration::configuration). KTT wants
// each value in the type of its parameter, which the text does not say: the types are tried in turn.
ktt::KernelConfiguration ParseConfiguration(ktt::Tuner& tuner, const ktt::KernelId kernel, const std::string& text)
{
    ktt::ParameterInput input;
    std::istringstream stream(text);
    std::string pair;

    while (stream >> pair)
    {
        const size_t equals = pair.find('=');

        if (equals == std::string::npos)
        {
            throw std::runtime_error("not a configuration: " + text);
        }

        const std::string name = pair.substr(0, equals);
        const std::string value = pair.substr(equals + 1);
        std::vector<ktt::ParameterValue> candidates;

        try
        {
            candidates.emplace_back(static_cast<uint64_t>(std::stoull(value)));
            candidates.emplace_back(static_cast<int64_t>(std::stoll(value)));
        }
        catch (const std::exception&)
        {
        }

        try
        {
            candidates.emplace_back(std::stod(value));
        }
        catch (const std::exception&)
        {
        }

        if (value == "true" || value == "false")
        {
            candidates.emplace_back(value == "true");
        }

        candidates.emplace_back(value);
        bool found = false;

        for (const auto& candidate : candidates)
        {
            try
            {
                tuner.CreateConfiguration(kernel, {{name, candidate}});
                input.emplace_back(name, candidate);
                found = true;
                break;
            }
            catch (const std::exception&)
            {
            }
        }

        if (!found)
        {
            throw std::runtime_error("no value of parameter " + name + " fits " + value);
        }
    }

    return tuner.CreateConfiguration(kernel, input);
}

} // namespace

int RunRecount(const Options& options)
{
    // The kernels and configurations of the earlier run, in its order: valid rows, the first measurement of each.
    std::ifstream input(options.recountCsv);

    if (!input)
    {
        throw std::runtime_error("Unable to open " + options.recountCsv);
    }

    std::string line;
    std::getline(input, line);

    if (line.rfind("kernel,configuration,valid,", 0) != 0)
    {
        throw std::runtime_error(options.recountCsv + " is not the CSV of a --validate run");
    }

    std::vector<std::string> kernels;
    std::map<std::string, std::vector<std::string>> configurations;

    while (std::getline(input, line))
    {
        std::istringstream cells(line);
        std::string kernel;
        std::string configuration;
        std::string valid;
        std::getline(cells, kernel, ',');
        std::getline(cells, configuration, ',');
        std::getline(cells, valid, ',');

        if (valid != "1" || kernel.empty())
        {
            continue;
        }

        auto& list = configurations[kernel];

        if (list.empty())
        {
            kernels.push_back(kernel);
        }

        if (std::find(list.cbegin(), list.cend(), configuration) == list.cend())
        {
            list.push_back(configuration);
        }
    }

    GpuSession session(options);
    ktt::Tuner& tuner = session.GetTuner();
    EnergyTuner energy(tuner);
    SetValidationData(options.validateData);
    std::cout << "Counting " << options.recountCsv << " again: no power measurement\n";

    for (const auto& name : kernels)
    {
        const ValidationKernel kernel = AddValidationKernel(tuner, name, kernelPrefix + "../Examples/");
        energy.AddKernel(name, kernel.kernel, kernel.definition, kernel.compilerOptions);
        std::vector<ktt::KernelConfiguration> parsed;

        for (const auto& text : configurations.at(name))
        {
            parsed.push_back(ParseConfiguration(tuner, kernel.kernel, text));
        }

        std::cout << "  " << name << ": " << parsed.size() << " configurations" << std::endl;
        energy.TimeConfigurations(name, parsed);
    }

    const std::vector<TunedConfiguration> rows = energy.CollectCounters(ModelCounters(options.counterTable,
        session.GetHardware().ComputeCapability()), options.outputPrefix);
    std::set<std::string> counterNames;

    for (const auto& row : rows)
    {
        for (const auto& counter : row.counters)
        {
            counterNames.insert(CanonicalCounter(counter.first));
        }
    }

    const std::string file = options.outputPrefix + "_recount.csv";
    std::ofstream output(file);
    output << "kernel,configuration,kernel_s,ktt_kernel_s,note";

    for (const auto& name : counterNames)
    {
        output << "," << name;
    }

    output << "\n";
    size_t counted = 0;

    for (const auto& row : rows)
    {
        output << CsvText(row.kernel) << "," << CsvText(row.configuration) << "," << CsvValue(row.kernelS) << ","
            << CsvValue(row.kttKernelS) << "," << CsvText(row.note);

        for (const auto& name : counterNames)
        {
            double value = NaN;

            for (const auto& counter : row.counters)
            {
                value = CanonicalCounter(counter.first) == name ? counter.second : value;
            }

            output << "," << CsvValue(value);
        }

        output << "\n";
        counted += row.counters.empty() ? 0 : 1;
    }

    std::cout << "Wrote " << file << " (" << counted << " of " << rows.size() << " configurations counted)\n";
    return counted == rows.size() ? 0 : 1;
}

// The passes are EnergyTuner's (the time and the CUPTI counters of random configurations, then NVML's power of each over
// back-to-back launches), with a sleep baseline before the first kernel's power measurements and after every kernel's.
// The counters come first: profiling replays the kernel and raises power, so it must not run between the power
// measurements of a kernel and their baseline.
int RunValidation(const Options& options)
{
    GpuSession session(options);
    std::cout << "Validation: " << options.validateConfigs << " random configurations each of";

    for (const auto& name : options.validate)
    {
        std::cout << " " << name;
    }

    std::cout << ": time and CUPTI counters, then " << options.validateMs << " ms of power measurement per configuration, "
        << "with a sleep baseline before and after every kernel\n";
    session.OpenMonitor();
    NvmlMonitor* const monitor = session.GetMonitor();

    if (monitor == nullptr)
    {
        throw std::runtime_error("--validate needs the NVML monitor");
    }

    ktt::Tuner& tuner = session.GetTuner();
    const BaselineKernels baseline = AddBaselineKernels(tuner, options, session.GetBlocks(),
        session.GetHardware().multiprocessors);
    const ktt::KernelId warmup = session.CreateKernel("warmup", baseline.spin, Configure{}, false, baseline.counts);
    SleepBaselines sleeps(session, baseline.sleep, baseline.counts, options.validate.size() + 1);

    // The meter keeps the whole trace summary of the last measurement (clocks, fan, limiters) for the CSV.
    std::string label;
    TraceSummary trace;
    PowerMeter meter;
    meter.start = [&]() { monitor->Start(label, options.traceIntervalMs); };
    meter.stop = [&]()
    {
        trace = monitor->Stop();
        return MeterReading{trace.sampledPowerW, trace.meanTemperatureC};
    };
    EnergyTuner energy(tuner, meter);

    SetValidationData(options.validateData);

    if (options.validateData != "example")
    {
        std::cout << "Input data of the kernels: " << options.validateData << "\n";
    }

    for (const auto& name : options.validate)
    {
        const ValidationKernel kernel = AddValidationKernel(tuner, name, kernelPrefix + "../Examples/");
        energy.AddKernel(name, kernel.kernel, kernel.definition, kernel.compilerOptions);
        tuner.SetSearcher(kernel.kernel, std::make_unique<ktt::RandomSearcher>());
    }

    const ValidationKernel gap = AddValidationKernel(tuner, LaunchGapKernel, kernelPrefix + "../Examples/");
    energy.AddKernel(LaunchGapKernel, gap.kernel, gap.definition, gap.compilerOptions);
    session.MeasureIdle("idle-start");
    energy.TimeConfigurations(LaunchGapKernel, 1);

    for (const auto& name : options.validate)
    {
        std::cout << "\n=== Validation kernel " << name << ": time ===" << std::endl;
        energy.TimeConfigurations(name, options.validateConfigs);
    }

    std::cout << "\n=== Counters ===" << std::endl;
    std::vector<TunedConfiguration> rows = energy.CollectCounters(ModelCounters(options.counterTable,
        session.GetHardware().ComputeCapability()), options.outputPrefix);
    std::vector<TraceSummary> traces(rows.size());
    session.WarmUp(warmup);
    sleeps.RunNext();

    // The power between launches: the empty kernel, right after the first baseline.
    const auto gapRow = std::find_if(rows.begin(), rows.end(), [](const TunedConfiguration& r) { return r.kernel == LaunchGapKernel; });
    double gapPowerW = NaN;
    double gapTemperatureC = NaN;

    if (gapRow != rows.end() && gapRow->kttKernelS > 0.0)
    {
        std::cout << "\n=== Power between launches (empty kernel) ===" << std::endl;
        label = LaunchGapKernel;
        energy.MeasurePower(*gapRow, static_cast<double>(options.validateMs) * 1e-3);
        gapPowerW = gapRow->measuredPowerW;
        gapTemperatureC = gapRow->measuredTemperatureC;
    }

    if (gapRow != rows.end())
    {
        traces.erase(traces.begin() + (gapRow - rows.begin()));
        rows.erase(gapRow);
    }

    for (const auto& name : options.validate)
    {
        std::cout << "\n=== Validation kernel " << name << ": power ===" << std::endl;

        for (size_t i = 0; i < rows.size(); ++i)
        {
            if (rows[i].kernel == name && rows[i].kernelS > 0.0)
            {
                label = name + " " + rows[i].configuration;
                energy.MeasurePower(rows[i], static_cast<double>(options.validateMs) * 1e-3);
                traces[i] = trace;
            }
        }

        sleeps.RunNext();
    }

    session.MeasureIdle("idle-end");
    session.WriteTrace();
    const BaselineModel model = FitBaseline(session.GetRows());
    WriteCsv(session.GetRows(), options.outputPrefix + ".csv");

    if (model.runs == 0)
    {
        std::cout << "\nNo power data was collected. Rebuild KTT with --power-usage and run on a GPU that reports power "
            << "through NVML.\n";
        return 1;
    }

    // How far below the baseline the GPU draws between launches; 0 (the gap at the baseline, as before 2026-11-17) when
    // the empty kernel could not be measured.
    double gapBelowBaselineW = model.At(gapTemperatureC) - gapPowerW;

    if (std::isnan(gapBelowBaselineW))
    {
        std::cout << "WARNING: the power between launches was not measured; it is taken at the baseline.\n";
        gapBelowBaselineW = 0.0;
    }

    std::cout << "\nPower between launches: " << Fixed(gapPowerW, 1) << " W at " << Fixed(gapTemperatureC, 0) << " C, "
        << Fixed(gapBelowBaselineW, 1) << " W below the baseline\n";
    WriteValidationCsv(rows, traces, model, gapBelowBaselineW, options.outputPrefix + "_validation.csv");
    std::cout << "\nWrote " << options.outputPrefix << "_validation.csv (one row per configuration), " << options.outputPrefix
        << ".csv (idle and sleep rows) and " << options.outputPrefix << "_trace.csv\n";
    return 0;
}
