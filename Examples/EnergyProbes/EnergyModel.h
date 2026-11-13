// Energy of a kernel launch from its CUPTI counters and its time, with a model file (tools/build_model.py). This is the
// part a tuner uses; it needs no power measurement. It depends only on the standard library and nlohmann::json, so it can
// be copied into any KTT program:
//
//     EnergyModel model("rtx3090_1200mhz.model.json");
//     tuner.SetProfiling(true);
//     tuner.SetProfilingCounters(model.Counters());
//     ...
//     EnergyPrediction p = model.Predict(counters, kernelSeconds);        // counters: CUPTI name -> value of one launch
//     EnergyRecommendation r = model.Recommend(predictions, EnergyObjective::Energy);
//
// The events and their counting are the same as in tools/energy_events.py (tests/test_tools.py compares the two on a
// validation run), the predictions the same as tools/energy_predictor.py. See that file for what the uncertainty means.
#pragma once

#include <cstddef>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

using CounterValues = std::map<std::string, double>;

struct EnergyPrediction
{
    std::string label;      // the caller's (a configuration)
    double energyJ = 0.0;   // per launch: baseline over the kernel's time plus the dynamic energy
    double sdJ = 0.0;       // standard deviation of energyJ between configurations of one kernel
    double dynamicJ = 0.0;  // sum of the event counts times the model's energies
    double baselineJ = 0.0; // GPU awake at the model's clock without work, over the kernel's time
    double kernelS = 0.0;
    std::vector<std::pair<std::string, double>> events; // event counts, in the model's order
    std::vector<std::string> warnings;
};

enum class EnergyObjective
{
    Energy, // J per launch
    Edp,    // J s
    Ed2p,   // J s^2
    Time    // s
};

struct EnergyRecommendation
{
    size_t index = 0;               // the configuration with the lowest predicted objective among the admissible ones
    std::vector<size_t> contenders; // the others that cannot be told apart from it, best first
    std::vector<double> values;     // the objective per configuration (NaN where not admissible)
};

class EnergyModel
{
public:
    // Throws std::runtime_error for a file that is not a model of a version this code knows.
    explicit EnergyModel(const std::string& file);

    const std::string& DeviceName() const { return m_Device; }
    double ClockMHz() const { return m_ClockMHz; }
    double RelativeSd() const { return m_RelativeSd; }

    // The GPU's power limit (W; NVML reports it). A configuration whose predicted power is above it will not run at the
    // model's clock: the GPU lowers its clock to hold the limit, and the energy is then about the limit times the time.
    // Predict() takes that value and adds a warning. NaN (the default) = no limit known.
    void SetPowerLimitW(const double watts) { m_PowerLimitW = watts; }

    // The CUPTI counters to collect: every counter the model needs, under all its names on other chips (KTT skips the
    // ones a GPU lacks).
    std::vector<std::string> Counters() const;

    // The counters the model needs that `counters` does not have.
    std::vector<std::string> MissingCounters(const CounterValues& counters) const;

    // Power of the GPU awake at the model's clock without work; temperatureC NaN = at the calibration's mean temperature.
    double BaselineW(double temperatureC = std::numeric_limits<double>::quiet_NaN()) const;

    // Event counts of one launch of kernelS seconds.
    std::vector<std::pair<std::string, double>> Events(const CounterValues& counters, double kernelS) const;

    // Energy of one launch. temperatureC and smClockMHz: NaN = unknown. With strict, counters the model needs and does not
    // find throw std::runtime_error (they would count as 0); without, they become a warning.
    EnergyPrediction Predict(const CounterValues& counters, double kernelS,
        double temperatureC = std::numeric_limits<double>::quiet_NaN(),
        double smClockMHz = std::numeric_limits<double>::quiet_NaN(), bool strict = true) const;

    // The configuration to use. maxSlowdown: only configurations at most this fraction slower than the fastest (NaN = all);
    // maxEnergyIncrease: the same against the lowest predicted energy. z: configurations within z standard deviations of
    // the best are contenders (measure them if the difference matters).
    EnergyRecommendation Recommend(const std::vector<EnergyPrediction>& predictions,
        EnergyObjective objective = EnergyObjective::Energy, double z = 2.0,
        double maxSlowdown = std::numeric_limits<double>::quiet_NaN(),
        double maxEnergyIncrease = std::numeric_limits<double>::quiet_NaN()) const;

private:
    double Value(const CounterValues& counters, const std::string& name) const; // NaN if absent, aliases resolved
    double Count(const CounterValues& counters, const std::string& name) const; // 0 if absent

    std::string m_Device;
    double m_ClockMHz = std::numeric_limits<double>::quiet_NaN();
    std::vector<std::pair<std::string, double>> m_Energies; // pJ per event, in the file's order
    std::vector<std::string> m_Unpriced;                    // events no probe exercised
    std::vector<std::string> m_Required;
    std::vector<std::pair<std::string, std::string>> m_Groups; // optional counter group and the counter that shows it
    std::map<std::string, std::vector<std::string>> m_Aliases; // canonical name -> other names
    std::vector<std::string> m_LsuShared;
    std::vector<std::string> m_LsuGlobal;
    bool m_Busy = true;
    double m_BusyScaleGbs = 95.0;
    bool m_FmaHalves = false;
    double m_BaselineW = 0.0;
    double m_BaselineTemperatureC = 0.0;
    double m_BaselineSlope = 0.0;
    double m_RelativeSd = 0.06;
    double m_BaselineSdW = 0.0;
    double m_UnknownTemperatureSdW = 0.0;
    double m_PowerLimitW = std::numeric_limits<double>::quiet_NaN();
};

double ObjectiveValue(const EnergyPrediction& prediction, EnergyObjective objective);
