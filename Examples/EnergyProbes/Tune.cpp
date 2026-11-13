#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Ktt.h>

#include "Common.h"
#include "EnergyModel.h"
#include "NvmlMonitor.h"
#include "ProbeKernels.h"
#include "Tune.h"
#include "ValidationKernels.h"

namespace
{

constexpr uint64_t TimedLaunches = 3; // per configuration, after one untimed launch

struct TuneRow
{
    std::string kernel;
    std::string configuration;
    double kernelS = NaN;
    double temperatureC = NaN;
    double smClockMHz = NaN;
    CounterValues counters;
    std::string note;
    bool predicted = false;
    EnergyPrediction prediction;
    size_t source = 0;          // --tune: index of the timed result the row came from
    double measuredPowerW = NaN; // --tune-measure: NVML power over back-to-back launches
    double measuredS = NaN;      // and the kernel's time in those launches
    double measuredJ = NaN;      // their product: energy per launch
    std::string measuredChoice;
};

// Measures the power of a row's configuration and fills its measured fields.
using Measure = std::function<void(TuneRow&)>;

EnergyObjective ObjectiveOf(const std::string& name)
{
    if (name == "energy")
    {
        return EnergyObjective::Energy;
    }

    if (name == "edp")
    {
        return EnergyObjective::Edp;
    }

    if (name == "ed2p")
    {
        return EnergyObjective::Ed2p;
    }

    return EnergyObjective::Time;
}

std::string Text(const double value, const int digits)
{
    std::ostringstream stream;
    stream << std::setprecision(digits) << value;
    return stream.str();
}

// Predicts every row, prints the model's choice per kernel and writes the CSV. With `measure`, the candidates (the choice,
// its contenders and the fastest configuration, or all with --tune-measure all) are then measured and the measured best
// is the final choice.
int Report(const EnergyModel& model, std::vector<TuneRow>& rows, const Options& options, const std::string& file,
    const Measure& measure = {})
{
    const EnergyObjective objective = ObjectiveOf(options.objective);
    const double maxSlowdown = options.maxSlowdownPct.has_value() ? options.maxSlowdownPct.value() / 100.0 : NaN;
    std::cout << "\nModel: " << model.DeviceName() << " at " << Fixed(model.ClockMHz(), 0) << " MHz, relative sd of the "
        << "dynamic energy " << Fixed(100.0 * model.RelativeSd(), 1) << " %\n";
    std::vector<std::string> kernels;

    for (auto& row : rows)
    {
        if (std::find(kernels.cbegin(), kernels.cend(), row.kernel) == kernels.cend())
        {
            kernels.push_back(row.kernel);
        }

        if (!(row.kernelS > 0.0) || row.counters.empty())
        {
            continue;
        }

        try
        {
            row.prediction = model.Predict(row.counters, row.kernelS, row.temperatureC, row.smClockMHz, !options.lenient);
            row.prediction.label = row.configuration;
            row.predicted = true;
        }
        catch (const std::exception& error)
        {
            AppendNote(row.note, error.what());
        }
    }

    std::ofstream output(file);
    output << "kernel,configuration,kernel_s,temperature_c,sm_clock_mhz,predicted_j,predicted_sd_j,predicted_dynamic_j,"
        << "baseline_j,choice,measured_power_w,measured_s,measured_j,measured_choice,note\n";
    bool failed = false;

    for (const auto& kernel : kernels)
    {
        std::vector<EnergyPrediction> predictions;
        std::vector<TuneRow*> predicted;
        size_t all = 0;

        for (auto& row : rows)
        {
            if (row.kernel == kernel)
            {
                ++all;

                if (row.predicted)
                {
                    predictions.push_back(row.prediction);
                    predicted.push_back(&row);
                }
            }
        }

        std::cout << "\n" << kernel << ": " << predictions.size() << " of " << all << " configurations predicted, objective "
            << options.objective << "\n";
        std::map<const TuneRow*, std::string> marks;

        if (predictions.empty())
        {
            failed = true;

            for (const auto& row : rows)
            {
                if (row.kernel == kernel && !row.note.empty())
                {
                    std::cout << "  " << row.note << "\n";
                    break;
                }
            }
        }
        else
        {
            const EnergyRecommendation choice = model.Recommend(predictions, objective, options.contenderSd, maxSlowdown);
            const EnergyPrediction& best = predictions[choice.index];
            marks[predicted[choice.index]] = "best";
            std::cout << "  use: " << best.label << "\n       " << Text(best.energyJ, 4) << " J +- " << Text(best.sdJ, 2)
                << " per launch (" << Text(best.dynamicJ, 4) << " dynamic), " << Text(best.kernelS * 1e3, 4) << " ms\n";

            if (choice.contenders.empty())
            {
                std::cout << "  confident: every other admissible configuration is predicted worse by more than "
                    << Text(options.contenderSd, 3) << " sd\n";
            }
            else
            {
                std::cout << "  " << choice.contenders.size() << " contender(s) within " << Text(options.contenderSd, 3)
                    << " sd (measure them if the difference matters):\n";

                for (const size_t i : choice.contenders)
                {
                    marks[predicted[i]] = "contender";
                    std::cout << "    +" << Fixed(100.0 * (choice.values[i] / choice.values[choice.index] - 1.0), 1) << " %  "
                        << predictions[i].label << "  (" << Text(predictions[i].energyJ, 4) << " J, "
                        << Text(predictions[i].kernelS * 1e3, 4) << " ms)\n";
                }
            }

            const auto fastest = std::min_element(predictions.cbegin(), predictions.cend(),
                [](const EnergyPrediction& a, const EnergyPrediction& b) { return a.kernelS < b.kernelS; });

            if (static_cast<size_t>(fastest - predictions.cbegin()) != choice.index)
            {
                std::cout << "  against the fastest configuration: " << Fixed(100.0 * (best.energyJ / fastest->energyJ - 1.0), 1)
                    << " % energy, +" << Fixed(100.0 * (best.kernelS / fastest->kernelS - 1.0), 1) << " % time\n";
            }

            std::vector<std::string> warnings;
            size_t offClock = 0;

            for (const auto& p : predictions)
            {
                offClock += std::any_of(p.warnings.cbegin(), p.warnings.cend(),
                    [](const std::string& w) { return w.rfind("SM clock", 0) == 0; });
            }

            // Measured on an unlocked RTX 3090 (README, "The model at another operating point").
            if (2 * offClock > predictions.size())
            {
                std::cout << "  NOTICE: " << offClock << " of " << predictions.size() << " configurations ran at another SM "
                    << "clock than the model's. On a GPU that is not\n  clock-locked and runs at its power limit, energy is "
                    << "close to the limit times the time: rank by time there\n  (--objective time).\n";
            }

            for (const auto& p : predictions)
            {
                for (const auto& warning : p.warnings)
                {
                    if (offClock >= 4 && warning.rfind("SM clock", 0) == 0)
                    {
                        continue;
                    }

                    if (std::find(warnings.cbegin(), warnings.cend(), warning) == warnings.cend())
                    {
                        warnings.push_back(warning);
                        std::cout << "  WARNING: " << warning << "\n";
                    }
                }
            }

            if (measure)
            {
                const size_t fastestIndex = static_cast<size_t>(fastest - predictions.cbegin());
                std::vector<size_t> candidates{choice.index};
                candidates.insert(candidates.end(), choice.contenders.cbegin(), choice.contenders.cend());

                if (std::find(candidates.cbegin(), candidates.cend(), fastestIndex) == candidates.cend())
                {
                    candidates.push_back(fastestIndex);
                }

                if (options.tuneMeasure == "all")
                {
                    for (size_t i = 0; i < predictions.size(); ++i)
                    {
                        if (std::find(candidates.cbegin(), candidates.cend(), i) == candidates.cend())
                        {
                            candidates.push_back(i);
                        }
                    }
                }

                const auto measured = [&](const size_t i)
                {
                    EnergyPrediction p;
                    p.energyJ = predicted[i]->measuredJ;
                    p.kernelS = predicted[i]->measuredS;
                    return ObjectiveValue(p, objective);
                };
                size_t final = predictions.size();
                std::cout << "  measured with NVML (" << options.validateMs << " ms each; energy = power x kernel time):\n"
                    << "    " << std::setw(10) << "E [J]" << std::setw(8) << "P [W]" << std::setw(10) << "t [ms]"
                    << std::setw(11) << "predicted" << "  " << std::left << std::setw(10) << "role" << std::right
                    << "configuration\n";

                for (const size_t i : candidates)
                {
                    TuneRow& row = *predicted[i];
                    measure(row);

                    if (std::isnan(row.measuredJ))
                    {
                        continue;
                    }

                    if (final == predictions.size() || measured(i) < measured(final))
                    {
                        final = i;
                    }

                    const std::string role = i == choice.index ? "choice" : std::find(choice.contenders.cbegin(),
                        choice.contenders.cend(), i) != choice.contenders.cend() ? "contender" : i == fastestIndex ? "fastest" : "";
                    std::cout << "    " << std::setw(10) << Text(row.measuredJ, 4) << std::setw(8) << Fixed(row.measuredPowerW, 1)
                        << std::setw(10) << Text(row.measuredS * 1e3, 4) << std::setw(9)
                        << Fixed(100.0 * (predictions[i].energyJ / row.measuredJ - 1.0), 1) << " %  " << std::left
                        << std::setw(10) << role << std::right << row.configuration << "\n";
                }

                if (final != predictions.size())
                {
                    predicted[final]->measuredChoice = "final";
                    std::cout << "  final choice: " << (final == choice.index ? "the model's choice"
                        : std::find(choice.contenders.cbegin(), choice.contenders.cend(), final) != choice.contenders.cend()
                        ? "a contender" : final == fastestIndex ? "the fastest configuration" : "another configuration");

                    if (final != choice.index && !std::isnan(predicted[choice.index]->measuredJ))
                    {
                        std::cout << ", " << Fixed(100.0 * (measured(choice.index) / measured(final) - 1.0), 1)
                            << " % better than the model's choice (measured " << options.objective << ")";
                    }

                    if (final != fastestIndex && !std::isnan(predicted[fastestIndex]->measuredJ))
                    {
                        std::cout << ", " << Fixed(100.0 * (measured(fastestIndex) / measured(final) - 1.0), 1)
                            << " % better than the fastest";
                    }

                    std::cout << "\n  " << predicted[final]->configuration << "\n";
                }
            }
        }

        for (const auto& row : rows)
        {
            if (row.kernel != kernel)
            {
                continue;
            }

            const auto mark = marks.find(&row);
            output << CsvText(row.kernel) << "," << CsvText(row.configuration) << "," << CsvValue(row.kernelS, 12) << ","
                << CsvValue(row.temperatureC) << "," << CsvValue(row.smClockMHz) << ","
                << CsvValue(row.predicted ? row.prediction.energyJ : NaN, 12) << ","
                << CsvValue(row.predicted ? row.prediction.sdJ : NaN, 12) << ","
                << CsvValue(row.predicted ? row.prediction.dynamicJ : NaN, 12) << ","
                << CsvValue(row.predicted ? row.prediction.baselineJ : NaN, 12) << ","
                << (mark != marks.cend() ? mark->second : "") << "," << CsvValue(row.measuredPowerW) << ","
                << CsvValue(row.measuredS, 12) << "," << CsvValue(row.measuredJ, 12) << "," << row.measuredChoice << ","
                << CsvText(row.note) << "\n";
        }
    }

    std::cout << "\nWrote " << file << "\n";
    return failed ? 1 : 0;
}

std::vector<std::string> SplitCsvLine(const std::string& line)
{
    std::vector<std::string> cells;
    std::string cell;
    std::istringstream stream(line);

    while (std::getline(stream, cell, ','))
    {
        cells.push_back(cell);
    }

    if (!line.empty() && line.back() == ',')
    {
        cells.emplace_back();
    }

    return cells;
}

} // namespace

int RunPredict(const Options& options)
{
    const EnergyModel model(options.modelFile);
    std::ifstream input(options.predictCsv);

    if (!input)
    {
        throw std::runtime_error("Unable to open " + options.predictCsv);
    }

    std::string line;
    std::getline(input, line);

    if (!line.empty() && line.back() == '\r')
    {
        line.pop_back();
    }

    const std::vector<std::string> header = SplitCsvLine(line);
    const auto column = [&header](const std::string& name)
    {
        const auto found = std::find(header.cbegin(), header.cend(), name);

        if (found == header.cend())
        {
            throw std::runtime_error("the CSV has no column " + name);
        }

        return static_cast<size_t>(found - header.cbegin());
    };

    const size_t kernel = column("kernel");
    const size_t configuration = column("configuration");
    const size_t valid = column("valid");
    const size_t seconds = column("kernel_s");
    const size_t note = column("note");
    const auto optional = [&header](const std::string& name)
    {
        return static_cast<size_t>(std::find(header.cbegin(), header.cend(), name) - header.cbegin());
    };
    const size_t temperature = optional("temperature_c");
    const size_t clock = optional("sm_clock_mhz");
    std::vector<TuneRow> rows;

    while (std::getline(input, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        std::vector<std::string> cells = SplitCsvLine(line);

        // A note with commas (files from before 2026-11-11): the extra cells belong to the note.
        while (cells.size() > header.size())
        {
            cells[note] += "," + cells[note + 1];
            cells.erase(cells.begin() + static_cast<std::ptrdiff_t>(note) + 1);
        }

        if (cells.size() < header.size() || cells[valid] != "1")
        {
            continue;
        }

        const auto value = [&cells](const size_t index)
        {
            if (index >= cells.size() || cells[index].empty())
            {
                return NaN;
            }

            try
            {
                return std::stod(cells[index]);
            }
            catch (const std::exception&)
            {
                return NaN;
            }
        };

        TuneRow row;
        row.kernel = cells[kernel];
        row.configuration = cells[configuration];
        row.kernelS = value(seconds);
        row.temperatureC = value(temperature);
        row.smClockMHz = value(clock);

        for (size_t i = note + 1; i < header.size(); ++i)
        {
            const double counter = value(i);

            if (!std::isnan(counter))
            {
                row.counters[header[i]] = counter;
            }
        }

        rows.push_back(row);
    }

    if (rows.empty())
    {
        throw std::runtime_error(options.predictCsv + " has no valid rows");
    }

    return Report(model, rows, options, options.outputPrefix + "_predicted.csv");
}

int RunTune(const Options& options)
{
    const EnergyModel model(options.modelFile);
    ktt::Tuner tuner(options.platform, options.device, ktt::ComputeApi::CUDA);
    tuner.SetGlobalSizeType(ktt::GlobalSizeType::CUDA);
    tuner.SetTimeUnit(ktt::TimeUnit::Microseconds);
    tuner.SetLoggingLevel(ktt::LoggingLevel::Warning);
    const std::string device = tuner.GetCurrentDeviceInfo().GetName();
    std::cout << "Tuning for " << options.objective << " with the energy model (no power measurement): "
        << options.validateConfigs << " random configurations each of";

    for (const auto& name : options.tune)
    {
        std::cout << " " << name;
    }

    std::cout << " on " << device << "\n";

    if (device != model.DeviceName())
    {
        std::cout << "WARNING: the model was calibrated on " << model.DeviceName() << ", this GPU is " << device
            << ": check it with tools/check_model.py\n";
    }

    // NVML directly: the GPU's power limit for the model, and the power of the candidates with --tune-measure.
    std::unique_ptr<NvmlMonitor> monitor;

    if (options.monitor)
    {
        try
        {
            monitor = std::make_unique<NvmlMonitor>(options.device);
        }
        catch (const std::exception& error)
        {
            std::cout << "WARNING: NVML monitor disabled: " << error.what() << "\n";
        }
    }

    EnergyModel limited = model;

    if (monitor != nullptr && monitor->GetPowerLimitW() > 0.0)
    {
        limited.SetPowerLimitW(monitor->GetPowerLimitW());
    }

    if (options.tuneMeasure != "none" && monitor == nullptr)
    {
        throw std::runtime_error("--tune-measure needs the NVML monitor");
    }

    // What the launcher does: time (0), or measure power over back-to-back launches for this many seconds.
    double measureS = 0.0;
    std::optional<TraceSummary> trace;
    NvmlMonitor* const nvml = monitor.get();
    const uint64_t traceMs = options.traceIntervalMs;
    std::vector<ValidationKernel> kernels;
    std::vector<std::vector<ktt::KernelResult>> results;

    // Timing pass: one untimed launch (compilation, first touch of the data), then the timed ones.
    for (const auto& name : options.tune)
    {
        kernels.push_back(AddValidationKernel(tuner, name, kernelPrefix + "../Examples/"));
        const ktt::KernelDefinitionId definition = kernels.back().definition;
        tuner.SetLauncher(kernels.back().kernel, [definition, &measureS, &trace, nvml, traceMs](ktt::ComputeInterface& interface)
        {
            if (interface.GetProfiling(definition))
            {
                interface.RunKernelWithProfiling(definition);
                return;
            }

            interface.RunKernel(definition);

            if (!(measureS > 0.0))
            {
                for (uint64_t i = 0; i < TimedLaunches; ++i)
                {
                    interface.RunKernel(definition);
                }

                return;
            }

            // As the validation's power pass: back-to-back launches while the NVML trace runs.
            nvml->Start("tune", traceMs);
            const auto begin = std::chrono::steady_clock::now();

            do
            {
                interface.RunKernel(definition);
            }
            while (std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() < measureS);

            trace = nvml->Stop();
        });
        tuner.SetCompilerOptions(kernels.back().compilerOptions);
        tuner.SetSearcher(kernels.back().kernel, std::make_unique<ktt::RandomSearcher>());
        results.push_back(tuner.Tune(kernels.back().kernel, std::make_unique<ktt::ConfigurationCount>(options.validateConfigs)));
        tuner.SetCompilerOptions("");
    }

    // Counter pass: profiling replays the kernel, so its times are not the kernel's.
    tuner.SetProfiling(true);

    if (!tuner.GetProfiling())
    {
        throw std::runtime_error("profiling could not be enabled (build KTT with --profiling=cupti)");
    }

    tuner.SetProfilingCounters(model.Counters());
    std::vector<TuneRow> rows;

    for (size_t k = 0; k < kernels.size(); ++k)
    {
        tuner.SetCompilerOptions(kernels[k].compilerOptions);
        std::vector<ktt::KernelResult> profiled; // saved as KTT writes them, for tools/energy_predictor.py --ktt

        for (const auto& timed : results[k])
        {
            TuneRow row;
            row.kernel = kernels[k].name;
            row.source = static_cast<size_t>(&timed - results[k].data());

            for (const auto& pair : timed.GetConfiguration().GetPairs())
            {
                row.configuration += (row.configuration.empty() ? "" : " ") + pair.GetName() + "=" + pair.GetValueString();
            }

            const auto& computations = timed.GetResults();

            if (!timed.IsValid() || computations.size() < 2)
            {
                AppendNote(row.note, "run failed");
                rows.push_back(row);
                continue;
            }

            double ns = 0.0;
            double temperature = 0.0;
            double clock = 0.0;

            for (size_t i = 1; i < computations.size(); ++i)
            {
                ns += static_cast<double>(computations[i].GetDuration());
                temperature += computations[i].GetTemperature();
                clock += static_cast<double>(computations[i].GetSMFrequency());
            }

            const double launches = static_cast<double>(computations.size() - 1);
            row.kernelS = ns * 1e-9 / launches;
            // KTT reads both from NVML when it is built with --power-usage; 0 otherwise.
            row.temperatureC = temperature > 0.0 ? temperature / launches : NaN;
            row.smClockMHz = clock > 0.0 ? clock / launches : NaN;

            // One pass per Run; the last pass carries the counters (as in the validation's counter pass).
            ktt::KernelResult result = tuner.Run(kernels[k].kernel, timed.GetConfiguration(), {});

            while (result.IsValid() && result.HasRemainingProfilingRuns())
            {
                result = tuner.Run(kernels[k].kernel, timed.GetConfiguration(), {});
            }

            if (result.IsValid() && !result.GetResults().empty() && result.GetResults()[0].HasProfilingData())
            {
                for (const auto& counter : result.GetResults()[0].GetProfilingData().GetCounters())
                {
                    row.counters[counter.GetName()] = CounterValue(counter);
                }

                profiled.push_back(result);
            }
            else
            {
                AppendNote(row.note, "no counters");
            }

            rows.push_back(row);
        }

        tuner.SaveResults(profiled, options.outputPrefix + "_" + kernels[k].name + "_counters", ktt::OutputFormat::JSON);
    }

    tuner.SetCompilerOptions("");
    tuner.SetProfiling(false);
    Measure measure;

    if (options.tuneMeasure != "none")
    {
        measure = [&](TuneRow& row)
        {
            const size_t k = static_cast<size_t>(std::find_if(kernels.cbegin(), kernels.cend(),
                [&row](const ValidationKernel& kernel) { return kernel.name == row.kernel; }) - kernels.cbegin());
            measureS = static_cast<double>(options.validateMs) * 1e-3;
            trace.reset();
            tuner.SetCompilerOptions(kernels[k].compilerOptions);
            const ktt::KernelResult result = tuner.Run(kernels[k].kernel, results[k][row.source].GetConfiguration(), {});
            tuner.SetCompilerOptions("");
            measureS = 0.0;
            const auto& computations = result.GetResults();

            if (!result.IsValid() || computations.size() < 2 || !trace.has_value())
            {
                AppendNote(row.note, "power measurement failed");
                return;
            }

            double ns = 0.0;

            for (size_t i = 1; i < computations.size(); ++i)
            {
                ns += static_cast<double>(computations[i].GetDuration());
            }

            row.measuredS = ns * 1e-9 / static_cast<double>(computations.size() - 1);
            row.measuredPowerW = trace->sampledPowerW;
            row.measuredJ = row.measuredPowerW * row.measuredS;
        };
    }

    return Report(limited, rows, options, options.outputPrefix + "_tune.csv", measure);
}
