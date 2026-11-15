#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>

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
// the wall time. Every launch through KTT ends with a synchronisation and NVML queries, so the GPU idles between
// launches at about baseline power; the busy fraction (kernel time / wall time) is written too. The baseline model is
// written with every row (mean power, its temperature, slope; slope 0 = no temperature model), so that the tools can
// move the baseline with another slope when this run's sleeps spanned too few degrees.
void WriteValidationCsv(const std::vector<TunedConfiguration>& rows, const std::vector<TraceSummary>& traces,
    const BaselineModel& baseline, const std::string& file)
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
        << "energy_j,note";

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
        const double dynamicJ = (row.measuredPowerW - baselineW) * row.measuredWallS;
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
            << "," << CsvValue(dynamicJ) << "," << CsvValue(energyJ) << "," << CsvText(note);

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

} // namespace

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

    for (const auto& name : options.validate)
    {
        const ValidationKernel kernel = AddValidationKernel(tuner, name, kernelPrefix + "../Examples/");
        energy.AddKernel(name, kernel.kernel, kernel.definition, kernel.compilerOptions);
        tuner.SetSearcher(kernel.kernel, std::make_unique<ktt::RandomSearcher>());
    }

    session.MeasureIdle("idle-start");

    for (const auto& name : options.validate)
    {
        std::cout << "\n=== Validation kernel " << name << ": time ===" << std::endl;
        energy.TimeConfigurations(name, options.validateConfigs);
    }

    std::cout << "\n=== Counters ===" << std::endl;
    std::vector<TunedConfiguration> rows = energy.CollectCounters(ModelCounters(options.counterTable), options.outputPrefix);
    std::vector<TraceSummary> traces(rows.size());
    session.WarmUp(warmup);
    sleeps.RunNext();

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

    WriteValidationCsv(rows, traces, model, options.outputPrefix + "_validation.csv");
    std::cout << "\nWrote " << options.outputPrefix << "_validation.csv (one row per configuration), " << options.outputPrefix
        << ".csv (idle and sleep rows) and " << options.outputPrefix << "_trace.csv\n";
    return 0;
}
