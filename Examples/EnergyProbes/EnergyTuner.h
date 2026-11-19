// The two passes a tuner needs to price configurations with the energy model (EnergyModel.h), for any simple KTT kernel:
//
//   1. a timing pass: the kernel's searcher picks configurations, each runs once untimed (compilation, first touch of the
//      data) and then a few times timed; no profiling, so the times are the kernel's;
//   2. a counter pass: every timed configuration once more with the CUPTI counters the model needs (profiling replays the
//      kernel, so its times are not used);
//
// and, optionally, NVML's power of a configuration over back-to-back launches (to measure the model's contenders).
// Prediction and choice are then EnergyModel::Predict and EnergyModel::Recommend. The README ("Using the model in a
// tuner") has a minimal program that tunes a kernel of its own with it; --tune (Tune.cpp) uses it for the example kernels.
//
// Depends on KTT and EnergyModel.h only; the power measurement is a callback (this example's NvmlMonitor in --tune).
#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <Ktt.h>

#include "EnergyModel.h"

// The value of a CUPTI counter as a number, whatever its type.
double CounterValue(const ktt::KernelProfilingCounter& counter);

// What a power meter read over a measurement.
struct MeterReading
{
    double powerW = std::numeric_limits<double>::quiet_NaN();       // mean power
    double temperatureC = std::numeric_limits<double>::quiet_NaN(); // mean GPU temperature (NaN if unknown)
};

// A power meter for MeasurePower: start() begins a measurement, stop() ends it and returns what it read. baselineW
// (optional): the power of the GPU awake without work at a temperature (EnergyModel::BaselineW); with it the measured
// energy charges the baseline over the kernel's time only and the power above it over the wall time, as the validation
// does, so that gaps between launches (KTT's overhead, tens of microseconds) do not make a short kernel look cheaper.
struct PowerMeter
{
    std::function<void()> start;
    std::function<MeterReading()> stop;
    std::function<double(double temperatureC)> baselineW = {};
    // How far below the baseline the GPU draws between two launches (W, 0 = at the baseline). Every launch through KTT
    // waits for its NVML hook (a context synchronisation and a sampling thread, about 0.6 ms on an RTX 3090), and the
    // GPU idles meanwhile at its context-idle power, about 6 W below the sleep baseline (2026-11-17).
    double gapBelowBaselineW = 0.0;
};

// The kernel's own GPU time per launch from the CUPTI counter gpu__time_duration.sum (counter pass), NaN without it.
// KTT's launch time also covers its NVML hook (PowerMeter::gapBelowBaselineW), which made a 0.2 ms kernel 0.8 ms.
double CounterKernelS(const CounterValues& counters);

// One configuration of a kernel: its time and counters, the model's prediction, and NVML's measurement if one was made.
struct TunedConfiguration
{
    std::string kernel;
    std::string configuration;                // "NAME=VALUE NAME=VALUE ..."
    ktt::KernelConfiguration kttConfiguration; // to run it again
    double kernelS = std::numeric_limits<double>::quiet_NaN();       // GPU time per launch (CounterKernelS, else the
                                                                      // timing pass)
    double kttKernelS = std::numeric_limits<double>::quiet_NaN();    // KTT's time per launch (timing pass)
    double temperatureC = std::numeric_limits<double>::quiet_NaN();  // KTT reads both from NVML when built with
    double smClockMHz = std::numeric_limits<double>::quiet_NaN();    // --power-usage
    CounterValues counters;                   // per launch (counter pass)
    std::string note;
    bool predicted = false;
    EnergyPrediction prediction;
    double measuredPowerW = std::numeric_limits<double>::quiet_NaN(); // MeasurePower: NVML power over the launches
    double measuredS = std::numeric_limits<double>::quiet_NaN();      // and the kernel's time in those launches
    double measuredKttS = std::numeric_limits<double>::quiet_NaN();   // (CounterKernelS, else KTT's time; KTT's here)
    double measuredJ = std::numeric_limits<double>::quiet_NaN();      // energy per launch: with the meter's baseline,
                                                                      // baseline x kernel time + (power - baseline) x
                                                                      // wall time + gapBelowBaselineW x (wall - kernel
                                                                      // time) per launch; without, power x wall time
    double measuredWallS = std::numeric_limits<double>::quiet_NaN();  // wall time per launch in those launches
    double busy = std::numeric_limits<double>::quiet_NaN();           // measuredS / measuredWallS (note below 0.95)
    double measuredTemperatureC = std::numeric_limits<double>::quiet_NaN(); // the GPU's temperature meanwhile
    double scaledPredictedJ = std::numeric_limits<double>::quiet_NaN();     // with the kernel's measured scale
    std::string measuredChoice;
    std::string measuredRole; // --tune: why it was measured besides being a candidate ("scale", "final")
};

// What CollectCounters may leave out: a configuration whose objective is sure, from its KTT time alone, to be above the
// best one profiled so far by more than `margin`. The model's energy is at least the baseline over the kernel's time
// (every event's price is >= 0), so such a configuration can be neither the model's choice nor a contender of
// EnergyModel::Recommend at the z of ContenderMargin(model, z). Configurations are profiled fastest first (by KTT's
// time), so the best is found early. Default: no pruning.
struct CounterPruning
{
    const EnergyModel* model = nullptr; // with the accounting and kernel scales the choice uses; null = profile all
    EnergyObjective objective = EnergyObjective::Energy; // Time: nothing is left out
    double margin = std::numeric_limits<double>::quiet_NaN(); // ContenderMargin(); NaN = nothing is left out
    double maxSlowdown = std::numeric_limits<double>::quiet_NaN(); // Recommend's limit: only configurations it admits
                                                                     // set the best value
    // KTT's time of a launch against the kernel's own (gpu__time_duration) on an RTX 3090 over 1,135 configurations of
    // 23 kernels (AllMany2, KttMany1, 2026-11-17): up to 1.39 ms longer (its NVML hook) and up to 8.6 % shorter. The
    // kernel's time is taken as at least TimeFactor x KTT's time - LaunchOverheadS.
    static constexpr double LaunchOverheadS = 1.5e-3;
    static constexpr double TimeFactor = 0.85;
};

// The tuner installs launchers that refer to this object: keep it alive as long as the ktt::Tuner may run its kernels.
class EnergyTuner
{
public:
    // Timed launches per configuration, after one untimed launch.
    static constexpr uint64_t TimedLaunches = 3;

    // meter: for MeasurePower (empty = no power measurement).
    explicit EnergyTuner(ktt::Tuner& tuner, PowerMeter meter = {});

    // Registers a simple kernel (one definition) under a name and installs its launcher. compilerOptions: the options
    // the kernel compiles with (KTT's compiler options are per tuner, so they are switched for every pass).
    void AddKernel(const std::string& name, ktt::KernelId kernel, ktt::KernelDefinitionId definition,
        const std::string& compilerOptions = "");

    // Timing pass: `count` configurations chosen by the kernel's searcher (set it with ktt::Tuner::SetSearcher).
    void TimeConfigurations(const std::string& name, uint64_t count);

    // Timing pass over given configurations (to count an earlier run's configurations again, --recount).
    void TimeConfigurations(const std::string& name, const std::vector<ktt::KernelConfiguration>& configurations);

    // Counter pass over every timed configuration of every kernel, with these counters (EnergyModel::Counters()).
    // resultPrefix: if not empty, the profiled results of each kernel go to PREFIX_<name>_counters.json (KTT's JSON,
    // which tools/energy_predictor.py --ktt reads). Rows without a time or counters carry a note. pruning: leave out
    // configurations that cannot be the model's choice (CounterPruning); their rows note it and have no counters.
    std::vector<TunedConfiguration> CollectCounters(const std::vector<std::string>& counterNames,
        const std::string& resultPrefix = "", const CounterPruning& pruning = {});

    // The power of one configuration over back-to-back launches for `seconds`; fills the measured fields (a note if it
    // fails). Needs the meter.
    // PowerMeter::gapBelowBaselineW, once the power between launches is measured.
    void SetGapBelowBaselineW(const double watts) { m_Meter.gapBelowBaselineW = watts; }

    void MeasurePower(TunedConfiguration& row, double seconds);

private:
    struct Kernel
    {
        ktt::KernelId kernel;
        ktt::KernelDefinitionId definition;
        std::string compilerOptions;
        std::vector<ktt::KernelResult> timed;
    };

    ktt::Tuner& m_Tuner;
    PowerMeter m_Meter;
    std::vector<std::string> m_Order;
    std::map<std::string, Kernel> m_Kernels;
    double m_MeasureS = 0.0; // what the launchers do: time (0), or measure power for this many seconds
    std::optional<MeterReading> m_Reading;
    double m_WallS = 0.0;     // the measured launches' wall time
    uint64_t m_Launches = 0;  // and their number
};
