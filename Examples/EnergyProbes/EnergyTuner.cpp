#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "EnergyTuner.h"

namespace
{

void Note(std::string& note, const std::string& text)
{
    note += (note.empty() ? "" : "; ") + text;
}

} // namespace

double CounterKernelS(const CounterValues& counters)
{
    const auto duration = counters.find("gpu__time_duration.sum");
    return duration != counters.cend() && duration->second > 0.0 ? duration->second * 1e-9
        : std::numeric_limits<double>::quiet_NaN();
}

double CounterValue(const ktt::KernelProfilingCounter& counter)
{
    switch (counter.GetType())
    {
    case ktt::ProfilingCounterType::Int:
        return static_cast<double>(counter.GetValueInt());
    case ktt::ProfilingCounterType::Double:
    case ktt::ProfilingCounterType::Percent:
        return counter.GetValueDouble();
    default:
        return static_cast<double>(counter.GetValueUint());
    }
}

EnergyTuner::EnergyTuner(ktt::Tuner& tuner, PowerMeter meter) :
    m_Tuner(tuner),
    m_Meter(std::move(meter))
{}

void EnergyTuner::AddKernel(const std::string& name, const ktt::KernelId kernel, const ktt::KernelDefinitionId definition,
    const std::string& compilerOptions)
{
    m_Order.push_back(name);
    m_Kernels[name] = Kernel{kernel, definition, compilerOptions, {}};

    m_Tuner.SetLauncher(kernel, [this, definition](ktt::ComputeInterface& interface)
    {
        // Counter pass: one profiled launch per pass (KTT repeats the launcher until every pass ran).
        if (interface.GetProfiling(definition))
        {
            interface.RunKernelWithProfiling(definition);
            return;
        }

        interface.RunKernel(definition);

        if (!(m_MeasureS > 0.0))
        {
            for (uint64_t i = 0; i < TimedLaunches; ++i)
            {
                interface.RunKernel(definition);
            }

            return;
        }

        // Power measurement: back-to-back launches while the meter runs.
        m_Meter.start();
        const auto begin = std::chrono::steady_clock::now();
        m_Launches = 0;

        do
        {
            interface.RunKernel(definition);
            ++m_Launches;
        }
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() < m_MeasureS);

        m_WallS = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        m_Reading = m_Meter.stop();
    });
}

void EnergyTuner::TimeConfigurations(const std::string& name, const uint64_t count)
{
    Kernel& kernel = m_Kernels.at(name);
    m_MeasureS = 0.0;
    m_Tuner.SetCompilerOptions(kernel.compilerOptions);
    kernel.timed = m_Tuner.Tune(kernel.kernel, std::make_unique<ktt::ConfigurationCount>(count));
    m_Tuner.SetCompilerOptions("");
}

void EnergyTuner::TimeConfigurations(const std::string& name, const std::vector<ktt::KernelConfiguration>& configurations)
{
    Kernel& kernel = m_Kernels.at(name);
    m_MeasureS = 0.0;
    m_Tuner.SetCompilerOptions(kernel.compilerOptions);

    for (const auto& configuration : configurations)
    {
        kernel.timed.push_back(m_Tuner.Run(kernel.kernel, configuration, {}));
    }

    m_Tuner.SetCompilerOptions("");
}

std::vector<TunedConfiguration> EnergyTuner::CollectCounters(const std::vector<std::string>& counterNames,
    const std::string& resultPrefix, const CounterPruning& pruning)
{
    m_Tuner.SetProfiling(true);

    if (!m_Tuner.GetProfiling())
    {
        throw std::runtime_error("profiling could not be enabled (build KTT with --profiling=cupti)");
    }

    m_Tuner.SetProfilingCounters(counterNames);
    const bool prune = pruning.model != nullptr && pruning.objective != EnergyObjective::Time && pruning.margin >= 0.0;
    std::vector<TunedConfiguration> rows;

    for (const auto& name : m_Order)
    {
        Kernel& kernel = m_Kernels.at(name);
        m_Tuner.SetCompilerOptions(kernel.compilerOptions);
        std::vector<ktt::KernelResult> profiled;
        std::vector<size_t> timed; // rows with a time, profiled fastest first

        for (const auto& result : kernel.timed)
        {
            TunedConfiguration row;
            row.kernel = name;
            row.kttConfiguration = result.GetConfiguration();

            for (const auto& pair : result.GetConfiguration().GetPairs())
            {
                row.configuration += (row.configuration.empty() ? "" : " ") + pair.GetName() + "=" + pair.GetValueString();
            }

            const auto& computations = result.GetResults();

            if (!result.IsValid() || computations.size() < 2)
            {
                Note(row.note, "run failed");
                rows.push_back(row);
                continue;
            }

            // The untimed launch is the first result.
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
            row.kttKernelS = row.kernelS;
            row.temperatureC = temperature > 0.0 ? temperature / launches : std::numeric_limits<double>::quiet_NaN();
            row.smClockMHz = clock > 0.0 ? clock / launches : std::numeric_limits<double>::quiet_NaN();
            timed.push_back(rows.size());
            rows.push_back(row);
        }

        if (prune)
        {
            std::stable_sort(timed.begin(), timed.end(), [&rows](const size_t a, const size_t b)
                { return rows[a].kttKernelS < rows[b].kttKernelS; });
        }

        // The profiled configurations' kernel time and predicted objective, for the best value so far.
        std::vector<std::pair<double, double>> values;

        for (const size_t index : timed)
        {
            TunedConfiguration& row = rows[index];

            if (prune && !values.empty())
            {
                double fastest = values.front().first;

                for (const auto& value : values)
                {
                    fastest = std::min(fastest, value.first);
                }

                double best = std::numeric_limits<double>::infinity();

                for (const auto& value : values)
                {
                    if (std::isnan(pruning.maxSlowdown) || value.first <= fastest * (1.0 + pruning.maxSlowdown))
                    {
                        best = std::min(best, value.second);
                    }
                }

                const double lowS = pruning.TimeFactor * row.kttKernelS - pruning.LaunchOverheadS;

                if (lowS > 0.0 && std::isfinite(best))
                {
                    EnergyPrediction bound = pruning.model->Predict({}, lowS, row.temperatureC, row.smClockMHz, false, name);
                    bound.kernelS = lowS;
                    const double low = ObjectiveValue(bound, pruning.objective);

                    if (low > best * (1.0 + pruning.margin))
                    {
                        Note(row.note, "not profiled: from its time alone at least "
                            + std::to_string(static_cast<int>(100.0 * (low / best - 1.0))) + " % above the best predicted "
                            + "value (pruned)");
                        continue;
                    }
                }
            }

            // One pass per Run; the last pass carries the counters.
            ktt::KernelResult result = m_Tuner.Run(kernel.kernel, row.kttConfiguration, {});

            while (result.IsValid() && result.HasRemainingProfilingRuns())
            {
                result = m_Tuner.Run(kernel.kernel, row.kttConfiguration, {});
            }

            if (result.IsValid() && !result.GetResults().empty() && result.GetResults()[0].HasProfilingData())
            {
                for (const auto& counter : result.GetResults()[0].GetProfilingData().GetCounters())
                {
                    row.counters[counter.GetName()] = CounterValue(counter);
                }

                if (CounterKernelS(row.counters) > 0.0)
                {
                    row.kernelS = CounterKernelS(row.counters);
                }

                profiled.push_back(result);

                if (prune)
                {
                    try
                    {
                        const EnergyPrediction p = pruning.model->Predict(row.counters, row.kernelS, row.temperatureC,
                            row.smClockMHz, false, name);
                        values.emplace_back(row.kernelS, ObjectiveValue(p, pruning.objective));
                    }
                    catch (const std::exception&)
                    {
                        // Not predictable: it sets no best value (the choice reports why).
                    }
                }
            }
            else
            {
                Note(row.note, "no counters");
            }
        }

        if (!resultPrefix.empty())
        {
            m_Tuner.SaveResults(profiled, resultPrefix + "_" + name + "_counters", ktt::OutputFormat::JSON);
        }
    }

    m_Tuner.SetCompilerOptions("");
    m_Tuner.SetProfiling(false);
    return rows;
}

void EnergyTuner::MeasurePower(TunedConfiguration& row, const double seconds)
{
    if (!m_Meter.start || !m_Meter.stop)
    {
        throw std::runtime_error("EnergyTuner::MeasurePower needs a power meter");
    }

    const Kernel& kernel = m_Kernels.at(row.kernel);
    m_MeasureS = seconds;
    m_Reading.reset();
    m_Tuner.SetCompilerOptions(kernel.compilerOptions);
    const ktt::KernelResult result = m_Tuner.Run(kernel.kernel, row.kttConfiguration, {});
    m_Tuner.SetCompilerOptions("");
    m_MeasureS = 0.0;
    const auto& computations = result.GetResults();

    if (!result.IsValid() || computations.size() < 2 || !m_Reading.has_value())
    {
        Note(row.note, "power measurement failed");
        return;
    }

    double ns = 0.0;

    for (size_t i = 1; i < computations.size(); ++i)
    {
        ns += static_cast<double>(computations[i].GetDuration());
    }

    row.measuredKttS = ns * 1e-9 / static_cast<double>(computations.size() - 1);
    row.measuredS = CounterKernelS(row.counters) > 0.0 ? CounterKernelS(row.counters) : row.measuredKttS;
    row.measuredPowerW = m_Reading->powerW;
    row.measuredTemperatureC = m_Reading->temperatureC;
    row.measuredWallS = m_Launches > 0 ? m_WallS / static_cast<double>(m_Launches) : row.measuredS;
    // The wall time includes the host's side of each launch; it cannot be shorter than the kernel.
    row.measuredWallS = std::max(row.measuredWallS, row.measuredS);
    row.busy = row.measuredS / row.measuredWallS;

    if (m_Meter.baselineW)
    {
        const double baseline = m_Meter.baselineW(row.measuredTemperatureC);
        row.measuredJ = baseline * row.measuredS + (row.measuredPowerW - baseline) * row.measuredWallS
            + m_Meter.gapBelowBaselineW * (row.measuredWallS - row.measuredS);
    }
    else
    {
        row.measuredJ = row.measuredPowerW * row.measuredWallS;
    }

    if (row.busy < 0.95)
    {
        Note(row.note, "kernel busy " + std::to_string(static_cast<int>(100.0 * row.busy + 0.5)) + " % of the wall time");
    }
}
