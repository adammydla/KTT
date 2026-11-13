// From rows to energies: the temperature model of the sleep baseline, energies per access above it, the fits
// across duties, and the selection of the rows and fits a result is taken from.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
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
    std::string note;
    double clockMHz = NaN;

    double At(const double temperatureC) const
    {
        if (!temperatureModel || std::isnan(temperatureC))
        {
            return meanW;
        }

        return meanW + slopeWPerC * (temperatureC - meanTemperatureC);
    }
};

// Which variant of a probe: access width (0 = any, e.g. compute rows), payload, and all SMs (SM sweep rows are separate).
struct Variant
{
    uint64_t width = DefaultWidth;
    std::string payload = PayloadName(PayloadRandom);
};

// Power above baseline = fixed + cost per access x access rate, fitted over the clock-matched duties of one probe.
// "Fixed" is power that comes with running the probe at all but does not scale with its access rate.
struct ProbeFit
{
    std::string operation;
    std::string level;
    uint64_t footprint = 0;
    uint64_t width = 0;
    std::string payload;
    double flopsPerOp = NaN;
    std::string duties;
    size_t points = 0;
    double pJPerAccess = NaN; // per access, or per operation of one thread for compute fits
    double pJPerByte = NaN;
    double fixedW = NaN;
    double r2 = NaN;
};

// Least-squares line y = intercept + slope * x.
struct Line
{
    double slope = NaN;
    double intercept = NaN;
    double r2 = NaN;
};

// Energy per access (per operation for compute probes) of a probe: the fit across intensities if there is one, otherwise
// the best single row.
struct Estimate
{
    double pJPerAccess = NaN;
    std::string source;
};

BaselineModel FitBaseline(const std::vector<Row>& rows);

bool ClockMatched(const Row& row);

// Energy per access above the baseline. A probe whose SM clock differs from the baseline's is compared with a baseline power
// measured at another clock and voltage, so its energy is biased. With a locked clock such a difference means the GPU
// lowered the clock below the lock, and the energy is withheld. Without a locked clock it is computed and marked as biased.
// Rows of the SM sweep are compared with the sleep baseline on the same number of SMs, measured just before them, moved to
// the row's temperature with the slope of the baseline model (probes on all SMs ran up to 7 C warmer than that baseline).
void ComputeEnergies(std::vector<Row>& rows, const BaselineModel& baseline, const bool clockLocked, const bool force);

// Rows of one probe. Rows with the representative footprint are preferred (sweep mode has several rows per level).
std::vector<const Row*> ProbeRows(const std::vector<Row>& rows, const std::string& operation, const std::string& level,
    const uint64_t footprint, const Variant& variant = {});

// The most trustworthy row of a probe: one with an energy, ideally at the baseline clock (then the highest duty), otherwise
// the one closest to the baseline clock.
const Row* SelectRow(const std::vector<Row>& rows, const std::string& operation, const std::string& level,
    const uint64_t footprint = 0, const Variant& variant = {});

// Rows for the difference of two probes: both at the same duty (identical SASS apart from the footprint) and both at the
// baseline clock, at the highest such duty. Falls back to each probe's best row, which the caller then flags.
std::pair<const Row*, const Row*> PairForDifference(const std::vector<Row>& rows, const std::string& operationA,
    const std::string& levelA, const uint64_t footprintA, const std::string& operationB, const std::string& levelB,
    const uint64_t footprintB, const Variant& variant = {});

// The full-intensity row of a probe, used where bandwidths are compared (a lower duty lowers bandwidth by design).
const Row* FullIntensityRow(const std::vector<Row>& rows, const std::string& operation, const std::string& level,
    const uint64_t footprint);

// Below about 25 % duty, power is not linear in the access rate for the instruction-heavy probes (L1, shared), so the fit
// only uses duties of at least minDuty. Every width and payload is a probe of its own; SM sweep rows are not fitted.
std::vector<ProbeFit> FitProbes(const std::vector<Row>& rows, const uint64_t minDuty);

const ProbeFit* FindFit(const std::vector<ProbeFit>& fits, const std::string& operation, const std::string& level,
    const uint64_t footprint, const Variant& variant = {});

void WriteFitsCsv(const std::vector<ProbeFit>& fits, const std::string& file);

Line FitLine(const std::vector<std::pair<double, double>>& points);

// A probe's energy per access: the slope of its fit across duties, or the energy of its best row. A fit whose fixed power
// exceeds half the dynamic power of the full-rate row is not a line through the rows (on an RTX 3090 the constant L1.5:
// duties 50 and 100 ran at the same rate, while power at duty 25 fell to 28 %, as fewer requests waited and were replayed),
// so the full-rate row is used instead.
Estimate EstimateEnergy(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const std::string& operation,
    const std::string& level, const uint64_t footprint, const Variant& variant);
