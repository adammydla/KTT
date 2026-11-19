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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Ktt.h>

#include "Common.h"
#include "EnergyModel.h"
#include "EnergyTuner.h"
#include "NvmlMonitor.h"
#include "Options.h"
#include "Tune.h"
#include "ValidationKernels.h"

namespace
{

using TuneRow = TunedConfiguration;

// Measures the power of a row's configuration and fills its measured fields.
using Measure = std::function<void(TuneRow&)>;

EnergyObjective ObjectiveOf(const std::string& name)
{
    return name == "energy" ? EnergyObjective::Energy : name == "edp" ? EnergyObjective::Edp
        : name == "ed2p" ? EnergyObjective::Ed2p : EnergyObjective::Time;
}

std::string Text(const double value, const int digits)
{
    std::ostringstream stream;
    stream << std::setprecision(digits) << value;
    return stream.str();
}

// The predicted rows of one kernel, with their predictions in the same order.
struct KernelRows
{
    std::string kernel;
    std::vector<TuneRow*> rows;
    std::vector<EnergyPrediction> predictions;
};

bool Contains(const std::vector<size_t>& indices, const size_t index)
{
    return std::find(indices.cbegin(), indices.cend(), index) != indices.cend();
}

// The energy-time trade-off: the Pareto front's points the model tells apart (ParetoSteps), and the choice.
void PrintFront(const KernelRows& k, const EnergyRecommendation& choice,
    const std::vector<size_t>& front, const std::vector<size_t>& steps, const Options& options)
{
    const auto& predictions = k.predictions;
    std::cout << "  energy-time trade-off: " << front.size() << " configuration(s) on the Pareto front, " << steps.size()
        << " told apart (each saves more than " << Text(options.contenderSd, 3) << " sd of energy\n  against the next "
        << "faster one):\n    " << std::setw(10) << "t [ms]" << std::setw(12) << "E [J]" << std::setw(10) << "+- sd"
        << std::setw(9) << "time" << std::setw(9) << "energy" << "  configuration\n";
    const EnergyPrediction& fastest = predictions[front.front()];

    for (const size_t i : front)
    {
        if (!Contains(steps, i) && i != choice.index)
        {
            continue;
        }

        const EnergyPrediction& p = predictions[i];
        std::cout << "    " << std::setw(10) << Text(p.kernelS * 1e3, 4) << std::setw(12) << Text(p.energyJ, 4)
            << std::setw(10) << Text(p.sdJ, 2) << std::setw(8) << Fixed(100.0 * (p.kernelS / fastest.kernelS - 1.0), 1)
            << "%" << std::setw(8) << Fixed(100.0 * (p.energyJ / fastest.energyJ - 1.0), 1) << "%  " << p.label
            << (i == choice.index ? "  [use]" : "") << (Contains(steps, i) ? "" : "  [not told apart from a faster one]")
            << "\n";
    }

}

// The model's choice, its contenders, how it compares with the fastest configuration, and the predictions' warnings.
void PrintChoice(const KernelRows& k, const EnergyRecommendation& choice, const size_t fastest,
    const std::vector<size_t>& front, const std::vector<size_t>& steps, const Options& options)
{
    const auto& predictions = k.predictions;
    const EnergyPrediction& best = predictions[choice.index];
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
            std::cout << "    +" << Fixed(100.0 * (choice.values[i] / choice.values[choice.index] - 1.0), 1) << " %  "
                << predictions[i].label << "  (" << Text(predictions[i].energyJ, 4) << " J, "
                << Text(predictions[i].kernelS * 1e3, 4) << " ms)\n";
        }
    }

    if (fastest != choice.index)
    {
        std::cout << "  against the fastest configuration: " << Fixed(100.0 * (best.energyJ / predictions[fastest].energyJ - 1.0), 1)
            << " % energy, +" << Fixed(100.0 * (best.kernelS / predictions[fastest].kernelS - 1.0), 1) << " % time\n";
    }

    PrintFront(k, choice, front, steps, options);
    const auto offClock = static_cast<size_t>(std::count_if(predictions.cbegin(), predictions.cend(), [](const auto& p)
    {
        return std::any_of(p.warnings.cbegin(), p.warnings.cend(), [](const std::string& w) { return w.rfind("SM clock", 0) == 0; });
    }));

    // Measured on an unlocked RTX 3090 (README, "Limits").
    if (2 * offClock > predictions.size())
    {
        std::cout << "  NOTICE: " << offClock << " of " << predictions.size() << " configurations ran at another SM "
            << "clock than the model's. On a GPU that is not\n  clock-locked and runs at its power limit, energy is "
            << "close to the limit times the time: rank by time there\n  (--objective time).\n";
    }

    std::vector<std::string> warnings;

    for (const auto& p : predictions)
    {
        for (const auto& warning : p.warnings)
        {
            if ((offClock < 4 || warning.rfind("SM clock", 0) != 0)
                && std::find(warnings.cbegin(), warnings.cend(), warning) == warnings.cend())
            {
                warnings.push_back(warning);
                std::cout << "  WARNING: " << warning << "\n";
            }
        }
    }
}

// --tune-measure: NVML's power of the candidates (the choice, its contenders and the fastest configuration, or all),
// and the final choice among them. Measurements closer than --measure-margin are not told apart (NVML's
// repeatability): among those near the measured best, the model's choice if it is one of them, else the fastest.
void MeasureCandidates(const EnergyModel& model, KernelRows& k, const EnergyRecommendation& choice, const size_t fastest,
    const Options& options, const Measure& measure)
{
    const EnergyObjective objective = ObjectiveOf(options.objective);
    std::vector<size_t> candidates{choice.index};
    // The contenders come best predicted first (Recommend), so a cap keeps the likeliest ones.
    const size_t contenders = options.measureMax > 0 && options.tuneMeasure == "contenders"
        ? std::min<size_t>(choice.contenders.size(), options.measureMax - 1) : choice.contenders.size();
    candidates.insert(candidates.end(), choice.contenders.cbegin(), choice.contenders.cbegin() + contenders);

    if (contenders < choice.contenders.size())
    {
        std::cout << "  measuring the choice and " << contenders << " of its " << choice.contenders.size()
            << " contenders (--measure-max " << options.measureMax << ")\n";
    }

    for (size_t i = 0; i < k.rows.size(); ++i)
    {
        if ((i == fastest || options.tuneMeasure == "all") && !Contains(candidates, i))
        {
            candidates.push_back(i);
        }
    }

    const auto measured = [&](const size_t i)
    {
        EnergyPrediction p;
        p.energyJ = k.rows[i]->measuredJ;
        p.kernelS = k.rows[i]->measuredS;
        return ObjectiveValue(p, objective);
    };
    const auto role = [&](const size_t i)
    {
        return i == choice.index ? "the model's choice" : Contains(choice.contenders, i) ? "a contender"
            : i == fastest ? "the fastest configuration" : "another configuration";
    };
    std::cout << "  measured with NVML (" << options.validateMs << " ms each; energy = baseline x kernel time + "
        << "the power above it x wall time; predicted at the measurement's temperature):\n    " << std::setw(10)
        << "E [J]" << std::setw(8) << "P [W]" << std::setw(10) << "t [ms]" << std::setw(11) << "predicted" << "  "
        << "configuration\n";
    std::vector<size_t> tied;
    size_t lowest = k.rows.size();

    for (const size_t i : candidates)
    {
        TuneRow& row = *k.rows[i];
        measure(row);

        if (std::isnan(row.measuredJ))
        {
            continue;
        }

        // The baseline at the temperature of this measurement, not of the timing pass.
        const double atMeasurement = model.Predict(row.counters, row.kernelS, row.measuredTemperatureC, row.smClockMHz,
            false, row.kernel).energyJ;
        std::cout << "    " << std::setw(10) << Text(row.measuredJ, 4) << std::setw(8) << Fixed(row.measuredPowerW, 1)
            << std::setw(10) << Text(row.measuredS * 1e3, 4) << std::setw(9)
            << Fixed(100.0 * (atMeasurement / row.measuredJ - 1.0), 1) << " %  " << row.configuration << "  (" << role(i)
            << ")\n";
        tied.push_back(i);
        lowest = lowest == k.rows.size() || measured(i) < measured(lowest) ? i : lowest;
    }

    if (tied.empty())
    {
        return;
    }

    const double limit = measured(lowest) * (1.0 + options.measureMarginPct / 100.0);
    tied.erase(std::remove_if(tied.begin(), tied.end(), [&](const size_t i) { return measured(i) > limit; }), tied.end());
    const size_t final = Contains(tied, choice.index) ? choice.index : *std::min_element(tied.cbegin(), tied.cend(),
        [&](const size_t a, const size_t b) { return k.rows[a]->measuredS < k.rows[b]->measuredS; });
    k.rows[final]->measuredChoice = "final";
    AppendNote(k.rows[final]->measuredRole, "final");
    std::cout << "  " << tied.size() << " measured within " << Fixed(options.measureMarginPct, 1)
        << " % of the lowest (--measure-margin, the repeatability of a measurement)\n  final choice: " << role(final);

    if (final != choice.index && !std::isnan(k.rows[choice.index]->measuredJ))
    {
        std::cout << ", " << Fixed(100.0 * (measured(choice.index) / measured(final) - 1.0), 1)
            << " % better than the model's choice (measured " << options.objective << ")";
    }

    if (final != fastest && !std::isnan(k.rows[fastest]->measuredJ))
    {
        std::cout << ", " << Fixed(100.0 * (measured(fastest) / measured(final) - 1.0), 1) << " % better than the fastest";
    }

    std::cout << "\n  " << k.rows[final]->configuration << "\n";
}

// The kernel's scale (measured / predicted dynamic energy): it removes the kernel's bias from the predictions of its
// other configurations and from comparisons with other kernels. It comes from `count` configurations at evenly spaced
// quantiles of the predicted dynamic energy, not from the candidates of the choice: those were picked for a low
// prediction and gave a scale 5.8 % (rms) off the scale over all configurations, the quantiles 2.0 %.
void ScaleKernel(EnergyModel& model, KernelRows& k, const uint64_t count, const Measure& measure)
{
    std::vector<size_t> order(k.rows.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](const size_t a, const size_t b)
        { return k.predictions[a].unscaledDynamicJ < k.predictions[b].unscaledDynamicJ; });
    std::vector<size_t> picked;

    for (uint64_t q = 0; q < std::min<uint64_t>(count, order.size()); ++q)
    {
        const size_t i = order[count == 1 ? order.size() / 2 : static_cast<size_t>(std::llround(static_cast<double>(q)
            * static_cast<double>(order.size() - 1) / static_cast<double>(count - 1)))];

        if (!Contains(picked, i))
        {
            picked.push_back(i);
        }
    }

    std::vector<EnergyPrediction> predictions;
    std::vector<double> measuredDynamic;

    for (const size_t i : picked)
    {
        TuneRow& row = *k.rows[i];

        if (std::isnan(row.measuredJ))
        {
            measure(row);
        }

        AppendNote(row.measuredRole, "scale");

        if (!std::isnan(row.measuredJ))
        {
            predictions.push_back(k.predictions[i]);
            measuredDynamic.push_back(row.measuredJ - model.BaselineW(row.measuredTemperatureC) * row.measuredS);
        }
    }

    if (std::isnan(model.KernelScale(k.kernel, predictions, measuredDynamic)))
    {
        std::cout << "  no kernel scale: " << measuredDynamic.size() << " of the " << picked.size()
            << " configurations for it were measured (3 needed)\n";
        return;
    }

    for (TuneRow* row : k.rows)
    {
        row->scaledPredictedJ = model.Predict(row->counters, row->kernelS, row->temperatureC, row->smClockMHz, false,
            row->kernel).energyJ;
    }

    std::cout << "  the kernel's scale from " << measuredDynamic.size() << " configurations spread over its predicted "
        << "dynamic energy: x " << Fixed(model.KernelScale(k.kernel, predictions, measuredDynamic), 3)
        << " (column scaled_predicted_j)\n";
}

// Predicts every row, prints the model's choice per kernel and writes the CSV. With `measure`, the candidates are then
// measured and the measured best is the final choice.
int Report(const EnergyModel& fileModel, std::vector<TuneRow>& rows, const Options& options, const std::string& file,
    const Measure& measure = {})
{
    EnergyModel model = fileModel;
    model.SetAccounting(AccountingOf(options.accounting), options.idleW.value_or(NaN));
    const double maxSlowdown = options.maxSlowdownPct.has_value() ? options.maxSlowdownPct.value() / 100.0 : NaN;
    const double maxEnergyIncrease = options.maxEnergyIncreasePct.has_value() ? options.maxEnergyIncreasePct.value() / 100.0
        : NaN;
    std::cout << "\nModel: " << model.DeviceName() << " at " << Fixed(model.ClockMHz(), 0) << " MHz, relative sd of the "
        << "dynamic energy " << Fixed(100.0 * model.RelativeSd(), 1) << " %, power limit "
        << (std::isnan(model.PowerLimitW()) ? std::string("none") : Fixed(model.PowerLimitW(), 0) + " W")
        << ", accounting " << options.accounting
        << (model.Accounting() == EnergyAccounting::AboveIdle ? " (idle " + Fixed(model.IdlePowerW(), 1) + " W)" : "") << "\n";
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
    output << "kernel,configuration,kernel_s,ktt_kernel_s,temperature_c,sm_clock_mhz,predicted_j,predicted_sd_j,predicted_dynamic_j,"
        << "baseline_j,accounting,choice,pareto,measured_power_w,measured_s,measured_j,measured_temperature_c,measured_choice,"
        << "scaled_predicted_j,measured_wall_s,measured_busy,measured_role,note\n";
    bool failed = false;

    for (const auto& kernel : kernels)
    {
        KernelRows k{kernel, {}, {}};
        std::vector<TuneRow*> all;

        for (auto& row : rows)
        {
            if (row.kernel == kernel)
            {
                all.push_back(&row);

                if (row.predicted)
                {
                    k.rows.push_back(&row);
                    k.predictions.push_back(row.prediction);
                }
            }
        }

        const auto pruned = std::count_if(all.cbegin(), all.cend(), [](const TuneRow* row)
            { return row->note.find("(pruned)") != std::string::npos; });
        std::cout << "\n" << kernel << ": " << k.rows.size() << " of " << all.size() << " configurations predicted, objective "
            << options.objective << "\n";

        if (pruned > 0)
        {
            std::cout << "  " << pruned << " not profiled: from their time alone above the best by more than the model's "
                << "error (--no-prune profiles them)\n";
        }

        EnergyRecommendation choice;
        std::vector<size_t> front;
        std::vector<size_t> steps;
        std::string problem;
        bool recommended = true;

        try
        {
            if (k.rows.empty())
            {
                const auto noted = std::find_if(all.cbegin(), all.cend(), [](const TuneRow* row) { return !row->note.empty(); });
                throw std::runtime_error(noted != all.cend() ? (*noted)->note : "no configuration has a time and counters");
            }

            choice = model.Recommend(k.predictions, ObjectiveOf(options.objective), options.contenderSd, maxSlowdown,
                maxEnergyIncrease);
        }
        catch (const std::exception& error)
        {
            // Nothing predicted, or no admissible configuration under both limits: nothing to recommend for this kernel.
            std::cout << "  " << error.what() << "\n";
            problem = k.rows.empty() ? "" : "no admissible configuration";
            recommended = false;
            failed = true;
        }

        if (recommended)
        {
            const auto fastest = static_cast<size_t>(std::min_element(k.predictions.cbegin(), k.predictions.cend(),
                [](const EnergyPrediction& a, const EnergyPrediction& b) { return a.kernelS < b.kernelS; })
                - k.predictions.cbegin());
            front = ParetoFront(k.predictions);
            steps = ParetoSteps(model, k.predictions, front, options.contenderSd);
            PrintChoice(k, choice, fastest, front, steps, options);

            if (measure)
            {
                MeasureCandidates(model, k, choice, fastest, options, measure);
                const uint64_t scaleConfigs = options.tuneScaleConfigs.value_or(options.tune.size() > 1 ? 5 : 0);

                if (scaleConfigs > 0)
                {
                    ScaleKernel(model, k, scaleConfigs, measure);
                }
            }
        }

        for (const TuneRow* row : all)
        {
            const auto index = static_cast<size_t>(std::find(k.rows.cbegin(), k.rows.cend(), row) - k.rows.cbegin());
            const std::string mark = !recommended || index >= k.rows.size() ? "" : index == choice.index ? "best" : Contains(choice.contenders, index)
                ? "contender" : "";
            const bool predicted = row->predicted && problem.empty();
            output << CsvText(row->kernel) << "," << CsvText(row->configuration) << "," << CsvValue(row->kernelS, 12) << ","
                << CsvValue(row->kttKernelS, 12) << "," << CsvValue(row->temperatureC) << "," << CsvValue(row->smClockMHz) << ","
                << CsvValue(predicted ? row->prediction.energyJ : NaN, 12) << ","
                << CsvValue(predicted ? row->prediction.sdJ : NaN, 12) << ","
                << CsvValue(predicted ? row->prediction.dynamicJ : NaN, 12) << ","
                << CsvValue(predicted ? row->prediction.baselineJ : NaN, 12) << "," << options.accounting << "," << mark
                << "," << (index >= k.rows.size() || !recommended ? "" : Contains(steps, index) ? "step"
                    : Contains(front, index) ? "front" : "")
                << "," << CsvValue(row->measuredPowerW) << "," << CsvValue(row->measuredS, 12) << ","
                << CsvValue(row->measuredJ, 12) << "," << CsvValue(row->measuredTemperatureC) << "," << row->measuredChoice
                << "," << CsvValue(row->scaledPredictedJ, 12) << "," << CsvValue(row->measuredWallS, 12) << ","
                << CsvValue(row->busy, 4) << "," << CsvText(row->measuredRole) << ","
                << CsvText(problem.empty() ? row->note : problem) << "\n";
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

// The power limit to predict with: --power-limit (0 = none), else `fallback` (NaN = none).
double PowerLimitOf(const Options& options, const double fallback)
{
    return options.powerLimitW.has_value() ? (options.powerLimitW.value() > 0.0 ? options.powerLimitW.value() : NaN)
        : fallback;
}

int RunPredict(const Options& options)
{
    EnergyModel model(options.modelFile);
    model.SetPowerLimitW(PowerLimitOf(options, model.FilePowerLimitW()));
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
    const size_t repeat = optional("repeat"); // runs of earlier versions measured a configuration several times
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

        // Only the first power measurement of each configuration.
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

    // A model belongs to the GPU type it was calibrated on (another card of the type: tools/check_model.py). On another
    // type its energies are off (the RTX 3090's on an RTX 2080 Ti: +19.5 % bias), so using it is an explicit choice. The
    // name is the test because it is what the model file records; whether a model fits is check_model.py's question.
    if (device != model.DeviceName())
    {
        if (!options.allowUncalibratedDevice)
        {
            throw std::runtime_error("the model was calibrated on " + model.DeviceName() + ", this GPU is " + device
                + ": calibrate this GPU (README), or pass --allow-uncalibrated-device to use the model anyway");
        }

        std::cout << "WARNING: the model was calibrated on " << model.DeviceName() << ", this GPU is " << device
            << " (--allow-uncalibrated-device): its energies are not this GPU's; check it with tools/check_model.py\n";
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
    limited.SetPowerLimitW(PowerLimitOf(options, monitor != nullptr && monitor->GetPowerLimitW() > 0.0
        ? monitor->GetPowerLimitW() : NaN));

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

    for (const auto& name : options.tune)
    {
        const ValidationKernel kernel = AddValidationKernel(tuner, name, kernelPrefix + "../Examples/");
        energyTuner.AddKernel(name, kernel.kernel, kernel.definition, kernel.compilerOptions);

        tuner.SetSearcher(kernel.kernel, std::make_unique<ktt::RandomSearcher>());

        energyTuner.TimeConfigurations(name, options.validateConfigs);
    }

    // The empty kernel, for the power between launches (Validation.cpp), when candidates are measured.
    if (options.tuneMeasure != "none")
    {
        const ValidationKernel gap = AddValidationKernel(tuner, LaunchGapKernel, kernelPrefix + "../Examples/");
        energyTuner.AddKernel(LaunchGapKernel, gap.kernel, gap.definition, gap.compilerOptions);
        energyTuner.TimeConfigurations(LaunchGapKernel, 1);
    }

    // The counter pass leaves out configurations that cannot be the choice (CounterPruning), unless --no-prune, every
    // configuration is to be measured, or a limit on energy makes the admissible set depend on all energies.
    EnergyModel choiceModel = limited;
    choiceModel.SetAccounting(AccountingOf(options.accounting), options.idleW.value_or(NaN));
    CounterPruning pruning;

    if (options.prune && options.tuneMeasure != "all"
        && (!options.maxEnergyIncreasePct.has_value() || options.objective == "energy"))
    {
        pruning.model = &choiceModel;
        pruning.objective = ObjectiveOf(options.objective);
        pruning.margin = ContenderMargin(choiceModel, options.contenderSd);
        pruning.maxSlowdown = options.maxSlowdownPct.has_value() ? options.maxSlowdownPct.value() / 100.0 : NaN;

        if (pruning.objective != EnergyObjective::Time && pruning.margin >= 0.0)
        {
            std::cout << "Counter pass: fastest first; a configuration predicted from its time alone to be more than "
                << Fixed(100.0 * pruning.margin, 1) << " % above the best " << options.objective << " so far is not "
                << "profiled (--no-prune)\n";
        }
    }

    std::vector<TuneRow> rows = energyTuner.CollectCounters(model.Counters(), options.outputPrefix, pruning);
    Measure measure;

    if (options.tuneMeasure != "none")
    {
        const double seconds = static_cast<double>(options.validateMs) * 1e-3;
        const auto gapRow = std::find_if(rows.begin(), rows.end(), [](const TuneRow& r) { return r.kernel == LaunchGapKernel; });

        if (gapRow != rows.end())
        {
            energyTuner.MeasurePower(*gapRow, seconds);

            if (gapRow->measuredPowerW > 0.0)
            {
                const double below = model.BaselineW(gapRow->measuredTemperatureC) - gapRow->measuredPowerW;
                energyTuner.SetGapBelowBaselineW(below);
                std::cout << "Power between launches: " << Text(gapRow->measuredPowerW, 4) << " W, " << Text(below, 3)
                    << " W below the model's baseline\n";
            }

            rows.erase(gapRow);
        }

        measure = [&energyTuner, seconds](TuneRow& row) { energyTuner.MeasurePower(row, seconds); };
    }

    return Report(limited, rows, options, options.outputPrefix + "_tune.csv", measure);
}
