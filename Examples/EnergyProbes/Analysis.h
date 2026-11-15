// From rows to energies: the temperature model of the sleep baseline, the energy per access above it, and the fit
// across duties.
#pragma once

#include <cmath>
#include <string>
#include <vector>
#include "Common.h"
#include "Rows.h"

// Baseline power as a function of GPU temperature, fitted over the sleep runs between the probe groups. Leakage grows with
// temperature, so every probe is compared with the baseline at its own temperature.
struct BaselineModel
{
    size_t runs = 0;
    double meanW = NaN;
    double minW = NaN;
    double maxW = NaN;
    double meanTemperatureC = NaN;
    double minTemperatureC = NaN;
    double maxTemperatureC = NaN;
    double slopeWPerC = 0.0;
    bool temperatureModel = false;
    std::string note; // why there is no temperature model
    double clockMHz = NaN;

    double At(const double temperatureC) const
    {
        return temperatureModel && !std::isnan(temperatureC) ? meanW + slopeWPerC * (temperatureC - meanTemperatureC) : meanW;
    }
};

// Power above baseline = fixed + cost per access x access rate, fitted over the clock-matched duties of one probe.
// "Fixed" is power that comes with running the probe at all but does not scale with its access rate.
struct ProbeFit
{
    std::string operation;
    std::string level;
    uint64_t footprint = 0;
    uint64_t width = 0;
    std::string duties;
    size_t points = 0;
    double pJPerAccess = NaN; // per access, or per operation of one thread for compute fits
    double fixedW = NaN;
    double r2 = NaN;          // with three duties or more
};

BaselineModel FitBaseline(const std::vector<Row>& rows);

// Whether a row ran at the baseline's SM clock (within ClockTolerance).
bool ClockMatched(const Row& row);

// Energy per access above the baseline. A probe whose SM clock differs from the baseline's is compared with a baseline power
// measured at another clock and voltage, so its energy is biased. With a locked clock such a difference means the GPU
// lowered the clock below the lock, and the energy is withheld (unless `force`). Without a locked clock it is computed and
// marked as biased.
void ComputeEnergies(std::vector<Row>& rows, const BaselineModel& baseline, const bool clockLocked, const bool force);

// Below about 25 % duty, power is not linear in the access rate for the instruction-heavy probes (L1, shared), so the fit
// only uses duties of at least minDuty. Probes measured at one duty have no fit.
std::vector<ProbeFit> FitProbes(const std::vector<Row>& rows, const uint64_t minDuty);
