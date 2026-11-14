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
#include "EnergyTuner.h"
#include "NvmlMonitor.h"
#include "ProbeKernels.h"
#include "Tune.h"
#include "ValidationKernels.h"

namespace
{

using TuneRow = TunedConfiguration;

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
int Report(const EnergyModel& fileModel, std::vector<TuneRow>& rows, const Options& options, const std::string& file,
    const Measure& measure = {})
{
    EnergyModel model = fileModel;
    model.SetAccounting(AccountingOf(options.accounting), options.idleW.value_or(NaN));
    const EnergyObjective objective = ObjectiveOf(options.objective);
    const double maxSlowdown = options.maxSlowdownPct.has_value() ? options.maxSlowdownPct.value() / 100.0 : NaN;
    const double maxEnergyIncrease = options.maxEnergyIncreasePct.has_value() ? options.maxEnergyIncreasePct.value() / 100.0
        : NaN;
    std::cout << "\nModel: " << model.DeviceName() << " at " << Fixed(model.ClockMHz(), 0) << " MHz, relative sd of the "
        << "dynamic energy " << Fixed(100.0 * model.RelativeSd(), 1) << " %, accounting " << options.accounting;

    if (model.Accounting() == EnergyAccounting::AboveIdle)
    {
        std::cout << " (idle " << Fixed(model.IdlePowerW(), 1) << " W)";
    }

    std::cout << "\n";
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
            row.prediction = model.Predict(row.counters, row.kernelS, row.temperatureC, row.smClockMHz, !options.lenient,
                row.kernel);
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
        << "baseline_j,accounting,choice,measured_power_w,measured_s,measured_j,measured_temperature_c,measured_choice,"
        << "scaled_predicted_j,measured_wall_s,measured_busy,measured_role,note\n";
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
            EnergyRecommendation choice;

            try
            {
                choice = model.Recommend(predictions, objective, options.contenderSd, maxSlowdown, maxEnergyIncrease);
            }
            catch (const std::exception& error)
            {
                // No admissible configuration under both limits: nothing to recommend for this kernel.
                std::cout << "  " << error.what() << "\n";
                failed = true;

                for (const auto& row : rows)
                {
                    if (row.kernel == kernel)
                    {
                        output << CsvText(row.kernel) << "," << CsvText(row.configuration) << ",,,,,,,," << options.accounting
                            << ",,,,,,,,,,,no admissible configuration\n";
                    }
                }

                continue;
            }

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
                std::cout << "  measured with NVML (" << options.validateMs << " ms each; energy = baseline x kernel time + "
                    << "the power above it x wall time; "
                    << "predicted at the measurement's temperature):\n"
                    << "    " << std::setw(10) << "E [J]" << std::setw(8) << "P [W]" << std::setw(10) << "t [ms]"
                    << std::setw(11) << "predicted" << "  " << std::left << std::setw(10) << "role" << std::right
                    << "configuration\n";

                std::vector<size_t> measuredOnes;

                for (const size_t i : candidates)
                {
                    TuneRow& row = *predicted[i];
                    measure(row);

                    if (std::isnan(row.measuredJ))
                    {
                        continue;
                    }

                    measuredOnes.push_back(i);
                    // The baseline at the temperature of this measurement, not of the timing pass.
                    const double atMeasurement = model.Predict(row.counters, row.kernelS, row.measuredTemperatureC,
                        row.smClockMHz, false, row.kernel).energyJ;
                    const std::string role = i == choice.index ? "choice" : std::find(choice.contenders.cbegin(),
                        choice.contenders.cend(), i) != choice.contenders.cend() ? "contender" : i == fastestIndex ? "fastest" : "";
                    std::cout << "    " << std::setw(10) << Text(row.measuredJ, 4) << std::setw(8) << Fixed(row.measuredPowerW, 1)
                        << std::setw(10) << Text(row.measuredS * 1e3, 4) << std::setw(9)
                        << Fixed(100.0 * (atMeasurement / row.measuredJ - 1.0), 1) << " %  " << std::left
                        << std::setw(10) << role << std::right << row.configuration << "\n";
                }

                // Measurements closer than the margin are not told apart (NVML's repeatability): among those near the
                // measured best, the model's choice if it is one of them, else the fastest of them.
                size_t final = predictions.size();
                const double margin = options.measureMarginPct / 100.0;

                if (!measuredOnes.empty())
                {
                    const size_t lowest = *std::min_element(measuredOnes.cbegin(), measuredOnes.cend(),
                        [&](const size_t a, const size_t b) { return measured(a) < measured(b); });
                    std::vector<size_t> tied;

                    for (const size_t i : measuredOnes)
                    {
                        if (measured(i) <= measured(lowest) * (1.0 + margin))
                        {
                            tied.push_back(i);
                        }
                    }

                    final = std::find(tied.cbegin(), tied.cend(), choice.index) != tied.cend() ? choice.index
                        : *std::min_element(tied.cbegin(), tied.cend(), [&](const size_t a, const size_t b)
                            { return predicted[a]->measuredS < predicted[b]->measuredS; });
                    std::cout << "  " << tied.size() << " measured within " << Fixed(100.0 * margin, 1)
                        << " % of the lowest (--measure-margin, the repeatability of a measurement)\n";
                }

                // The kernel's scale (measured / predicted dynamic energy, the awake baseline at each measurement's
                // temperature taken off): it removes the kernel's bias from the predictions of its other configurations
                // and from comparisons with other kernels. It comes from --tune-scale-configs configurations at evenly
                // spaced quantiles of the predicted dynamic energy, not from the candidates above: those were chosen for
                // a low prediction and gave a scale 5.8 % (rms) off the scale over all configurations, the quantiles
                // 2.0 % (27 kernel runs, NEXT_STEPS_PLAN P1; random picks 2.5 %).
                const uint64_t scaleConfigs = options.tuneScaleConfigs.value_or(options.tune.size() > 1 ? 5 : 0);

                if (scaleConfigs > 0)
                {
                    std::vector<size_t> order(predictions.size());

                    for (size_t i = 0; i < order.size(); ++i)
                    {
                        order[i] = i;
                    }

                    std::sort(order.begin(), order.end(), [&](const size_t a, const size_t b)
                        { return predictions[a].unscaledDynamicJ < predictions[b].unscaledDynamicJ; });
                    std::vector<size_t> scaleSet;

                    for (uint64_t q = 0; q < std::min<uint64_t>(scaleConfigs, order.size()); ++q)
                    {
                        const size_t position = scaleConfigs == 1 ? order.size() / 2
                            : static_cast<size_t>(std::llround(static_cast<double>(q) * static_cast<double>(order.size() - 1)
                                / static_cast<double>(scaleConfigs - 1)));
                        const size_t i = order[position];

                        if (std::find(scaleSet.cbegin(), scaleSet.cend(), i) == scaleSet.cend())
                        {
                            scaleSet.push_back(i);
                        }
                    }

                    std::vector<EnergyPrediction> scalePredictions;
                    std::vector<double> measuredDynamic;

                    for (const size_t i : scaleSet)
                    {
                        TuneRow& row = *predicted[i];

                        if (std::isnan(row.measuredJ))
                        {
                            measure(row);
                        }

                        AppendNote(row.measuredRole, "scale");

                        if (!std::isnan(row.measuredJ))
                        {
                            scalePredictions.push_back(predictions[i]);
                            measuredDynamic.push_back(row.measuredJ - model.BaselineW(row.measuredTemperatureC) * row.measuredS);
                        }
                    }

                    const double scale = model.KernelScale(kernel, scalePredictions, measuredDynamic);

                    if (!std::isnan(scale))
                    {
                        for (TuneRow* row : predicted)
                        {
                            row->scaledPredictedJ = model.Predict(row->counters, row->kernelS, row->temperatureC,
                                row->smClockMHz, false, row->kernel).energyJ;
                        }

                        std::cout << "  the kernel's scale from " << measuredDynamic.size() << " configurations spread over its "
                            << "predicted dynamic energy: x " << Fixed(scale, 3) << " (column scaled_predicted_j)\n";
                    }
                    else
                    {
                        std::cout << "  no kernel scale: " << measuredDynamic.size() << " of the " << scaleSet.size()
                            << " configurations for it were measured (3 needed)\n";
                    }
                }

                if (final != predictions.size())
                {
                    predicted[final]->measuredChoice = "final";
                    AppendNote(predicted[final]->measuredRole, "final");
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
                << CsvValue(row.predicted ? row.prediction.baselineJ : NaN, 12) << "," << options.accounting << ","
                << (mark != marks.cend() ? mark->second : "") << "," << CsvValue(row.measuredPowerW) << ","
                << CsvValue(row.measuredS, 12) << "," << CsvValue(row.measuredJ, 12) << "," << CsvValue(row.measuredTemperatureC)
                << "," << row.measuredChoice << "," << CsvValue(row.scaledPredictedJ, 12) << ","
                << CsvValue(row.measuredWallS, 12) << "," << CsvValue(row.busy, 4) << "," << CsvText(row.measuredRole) << ","
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
    const size_t repeat = optional("repeat");
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

        // --validate-repeat: the first power measurement of each configuration.
        if (cells.size() < header.size() || cells[valid] != "1"
            || (repeat < header.size() && !cells[repeat].empty() && cells[repeat] != "0"))
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

    // The timing and counter passes (EnergyTuner.h), then the model's choice; with --tune-measure, NVML's power of the
    // candidates.
    PowerMeter meter;

    if (monitor != nullptr)
    {
        NvmlMonitor* const nvml = monitor.get();
        const uint64_t traceMs = options.traceIntervalMs;
        meter.start = [nvml, traceMs]() { nvml->Start("tune", traceMs); };
        meter.stop = [nvml]()
        {
            const TraceSummary trace = nvml->Stop();
            return MeterReading{trace.sampledPowerW, trace.meanTemperatureC};
        };
    }

    // The measured energy charges the model's baseline over the kernel time and the rest over the wall time.
    meter.baselineW = [model](const double temperatureC) { return model.BaselineW(temperatureC); };
    EnergyTuner energyTuner(tuner, meter);

    SetValidationInputs({options.validateData, options.validateSizeScale, options.accelwattchDirectory});

    for (const auto& name : options.tune)
    {
        const ValidationKernel kernel = AddValidationKernel(tuner, name, kernelPrefix + "../Examples/");
        energyTuner.AddKernel(name, kernel.kernel, kernel.definition, kernel.compilerOptions);

        if (options.validateSeed.has_value())
        {
            tuner.SetSearcher(kernel.kernel, std::make_unique<SeededSearcher>(options.validateSeed.value()));
        }
        else
        {
            tuner.SetSearcher(kernel.kernel, std::make_unique<ktt::RandomSearcher>());
        }

        energyTuner.TimeConfigurations(name, options.validateConfigs);
    }

    std::vector<TuneRow> rows = energyTuner.CollectCounters(model.Counters(), options.outputPrefix);
    Measure measure;

    if (options.tuneMeasure != "none")
    {
        const double seconds = static_cast<double>(options.validateMs) * 1e-3;
        measure = [&energyTuner, seconds](TuneRow& row) { energyTuner.MeasurePower(row, seconds); };
    }

    return Report(limited, rows, options, options.outputPrefix + "_tune.csv", measure);
}
