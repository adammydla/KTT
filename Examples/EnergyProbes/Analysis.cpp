#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "Analysis.h"

namespace
{

bool Matches(const Row& row, const Variant& variant)
{
    return (variant.width == 0 || row.width == 0 || row.width == variant.width) && row.payload == variant.payload
        && row.activeSms == 0;
}

int LevelRank(const std::string& level)
{
    const std::vector<std::string> order = {"shared", "L1", "L2", "mixed", "DRAM", "none", "compute"};
    const auto iterator = std::find(order.cbegin(), order.cend(), level);
    return static_cast<int>(iterator - order.cbegin());
}

} // namespace

BaselineModel FitBaseline(const std::vector<Row>& rows)
{
    BaselineModel model;
    std::vector<const Row*> runs;

    for (const auto& row : rows)
    {
        if (row.level == "baseline" && row.hasPower && row.activeSms == 0)
        {
            runs.push_back(&row);
        }
    }

    model.runs = runs.size();

    if (runs.empty())
    {
        return model;
    }

    double sumW = 0.0;
    double sumTemperature = 0.0;
    double sumClock = 0.0;
    size_t temperatures = 0;
    size_t clocks = 0;
    model.minW = model.maxW = runs.front()->powerW;

    for (const Row* run : runs)
    {
        sumW += run->powerW;
        model.minW = std::min(model.minW, run->powerW);
        model.maxW = std::max(model.maxW, run->powerW);

        if (!std::isnan(run->temperatureC))
        {
            sumTemperature += run->temperatureC;
            model.minTemperatureC = temperatures == 0 ? run->temperatureC : std::min(model.minTemperatureC, run->temperatureC);
            model.maxTemperatureC = temperatures == 0 ? run->temperatureC : std::max(model.maxTemperatureC, run->temperatureC);
            ++temperatures;
        }

        if (!std::isnan(run->smClockMHz))
        {
            sumClock += run->smClockMHz;
            ++clocks;
        }
    }

    model.meanW = sumW / static_cast<double>(runs.size());
    model.clockMHz = clocks > 0 ? sumClock / static_cast<double>(clocks) : NaN;

    if (temperatures > 0)
    {
        model.meanTemperatureC = sumTemperature / static_cast<double>(temperatures);
    }

    if (temperatures < 3 || temperatures != runs.size())
    {
        model.note = "fewer than 3 baseline runs with a temperature, using the mean";
        return model;
    }

    if (model.maxTemperatureC - model.minTemperatureC < 2.0)
    {
        model.note = "baseline temperatures within 2 C, using the mean";
        return model;
    }

    // Least squares line through the (temperature, power) points; it passes through the means.
    double sxx = 0.0;
    double sxy = 0.0;

    for (const Row* run : runs)
    {
        const double dx = run->temperatureC - model.meanTemperatureC;
        sxx += dx * dx;
        sxy += dx * (run->powerW - model.meanW);
    }

    const double slope = sxy / sxx;

    if (slope < 0.0 || slope > 5.0)
    {
        model.note = "fitted slope of " + Fixed(slope, 2) + " W/C is implausible, using the mean";
        return model;
    }

    model.slopeWPerC = slope;
    model.temperatureModel = true;
    return model;
}

bool ClockMatched(const Row& row)
{
    return !std::isnan(row.clockDifference) && std::abs(row.clockDifference) <= ClockTolerance;
}

void ComputeEnergies(std::vector<Row>& rows, const BaselineModel& baseline, const bool clockLocked, const bool force)
{
    const double baselineClockMHz = baseline.clockMHz;
    std::map<uint64_t, const Row*> sweepBaselines;

    for (const auto& row : rows)
    {
        if (row.level == "baseline" && row.activeSms != 0 && row.hasPower)
        {
            sweepBaselines[row.activeSms] = &row;
        }
    }

    for (auto& row : rows)
    {
        if (!IsProbeRow(row) || !row.hasPower || row.accesses == 0)
        {
            continue;
        }

        bool withhold = false;

        if (row.activeSms == 0)
        {
            row.baselineW = baseline.At(row.temperatureC);
        }
        else if (sweepBaselines.count(row.activeSms) > 0)
        {
            const Row* sleep = sweepBaselines.at(row.activeSms);
            const double temperatureDifference = std::isnan(row.temperatureC) || std::isnan(sleep->temperatureC) ? 0.0
                : row.temperatureC - sleep->temperatureC;
            row.baselineW = sleep->powerW + baseline.slopeWPerC * temperatureDifference;
        }
        else
        {
            AppendNote(row.note, "no baseline on " + std::to_string(row.activeSms) + " SMs");
            continue;
        }

        if (!std::isnan(row.smClockMHz) && !std::isnan(baselineClockMHz) && baselineClockMHz > 0.0)
        {
            row.clockDifference = (row.smClockMHz - baselineClockMHz) / baselineClockMHz;

            if (!ClockMatched(row))
            {
                withhold = clockLocked;
                AppendNote(row.note, "SM clock " + SignedPercent(row.clockDifference) + " vs baseline"
                    + (clockLocked ? "" : " (biased)"));
            }
        }

        if (row.powerW <= row.baselineW)
        {
            withhold = true;
            AppendNote(row.note, "power not above baseline");
        }

        if (!row.limiters.empty())
        {
            AppendNote(row.note, "clock limited by " + row.limiters);
        }

        if (withhold && !force)
        {
            AppendNote(row.note, "energy withheld");
            continue;
        }

        if (withhold)
        {
            AppendNote(row.note, "forced");
        }

        row.dynamicJ = (row.powerW - row.baselineW) * row.durationS;
        row.pJPerAccess = row.dynamicJ / static_cast<double>(row.accesses) * 1e12;

        if (row.width > 0)
        {
            row.pJPerByte = row.pJPerAccess / static_cast<double>(row.width);
        }

        if (!std::isnan(row.flopsPerOp))
        {
            row.pJPerFlop = row.pJPerAccess / row.flopsPerOp;
        }
    }
}

std::vector<const Row*> ProbeRows(const std::vector<Row>& rows, const std::string& operation, const std::string& level,
    const uint64_t footprint, const Variant& variant)
{
    std::vector<const Row*> result;

    for (const bool exactFootprint : {true, false})
    {
        for (const auto& row : rows)
        {
            if (row.operation == operation && row.level == level && row.hasPower && row.accesses > 0 && Matches(row, variant)
                && (!exactFootprint || footprint == 0 || row.footprint == footprint))
            {
                result.push_back(&row);
            }
        }

        if (!result.empty())
        {
            break;
        }
    }

    return result;
}

const Row* SelectRow(const std::vector<Row>& rows, const std::string& operation, const std::string& level,
    const uint64_t footprint, const Variant& variant)
{
    auto distance = [](const Row* row)
    {
        return std::isnan(row->clockDifference) ? std::numeric_limits<double>::max() : std::abs(row->clockDifference);
    };

    auto better = [&distance](const Row* a, const Row* b)
    {
        const bool aEnergy = !std::isnan(a->pJPerAccess);
        const bool bEnergy = !std::isnan(b->pJPerAccess);

        if (aEnergy != bEnergy)
        {
            return aEnergy;
        }

        if (ClockMatched(*a) != ClockMatched(*b))
        {
            return ClockMatched(*a);
        }

        if (ClockMatched(*a))
        {
            return a->duty > b->duty;
        }

        return distance(a) < distance(b);
    };

    const Row* best = nullptr;

    for (const Row* row : ProbeRows(rows, operation, level, footprint, variant))
    {
        if (best == nullptr || better(row, best))
        {
            best = row;
        }
    }

    return best;
}

std::pair<const Row*, const Row*> PairForDifference(const std::vector<Row>& rows, const std::string& operationA,
    const std::string& levelA, const uint64_t footprintA, const std::string& operationB, const std::string& levelB,
    const uint64_t footprintB, const Variant& variant)
{
    const Row* bestA = nullptr;
    const Row* bestB = nullptr;

    for (const Row* a : ProbeRows(rows, operationA, levelA, footprintA, variant))
    {
        for (const Row* b : ProbeRows(rows, operationB, levelB, footprintB, variant))
        {
            const bool usable = a->duty == b->duty && ClockMatched(*a) && ClockMatched(*b) && !std::isnan(a->pJPerAccess)
                && !std::isnan(b->pJPerAccess);

            if (usable && (bestA == nullptr || a->duty > bestA->duty))
            {
                bestA = a;
                bestB = b;
            }
        }
    }

    if (bestA != nullptr)
    {
        return {bestA, bestB};
    }

    return {SelectRow(rows, operationA, levelA, footprintA, variant), SelectRow(rows, operationB, levelB, footprintB, variant)};
}

const Row* FullIntensityRow(const std::vector<Row>& rows, const std::string& operation, const std::string& level,
    const uint64_t footprint)
{
    const Row* best = nullptr;

    for (const Row* row : ProbeRows(rows, operation, level, footprint))
    {
        if (best == nullptr || row->duty > best->duty)
        {
            best = row;
        }
    }

    return best;
}

std::vector<ProbeFit> FitProbes(const std::vector<Row>& rows, const uint64_t minDuty)
{
    std::map<std::tuple<std::string, int, std::string, uint64_t, uint64_t>, std::vector<const Row*>> groups;

    for (const auto& row : rows)
    {
        if (!IsProbeRow(row) || row.activeSms != 0 || std::isnan(row.pJPerAccess) || !ClockMatched(row)
            || std::isnan(row.baselineW) || row.durationS <= 0.0 || row.duty < minDuty)
        {
            continue;
        }

        groups[{row.payload, LevelRank(row.level), row.operation, row.footprint, row.width}].push_back(&row);
    }

    std::vector<ProbeFit> fits;

    for (auto& [key, members] : groups)
    {
        if (members.size() < 2)
        {
            continue;
        }

        std::sort(members.begin(), members.end(), [](const Row* a, const Row* b) { return a->duty > b->duty; });

        ProbeFit fit;
        fit.operation = members.front()->operation;
        fit.level = members.front()->level;
        fit.footprint = members.front()->footprint;
        fit.width = members.front()->width;
        fit.payload = members.front()->payload;
        fit.flopsPerOp = members.front()->flopsPerOp;
        fit.points = members.size();

        double meanX = 0.0;
        double meanY = 0.0;

        for (const Row* row : members)
        {
            fit.duties += (fit.duties.empty() ? "" : "/") + std::to_string(row->duty);
            meanX += static_cast<double>(row->accesses) / row->durationS;
            meanY += row->powerW - row->baselineW;
        }

        meanX /= static_cast<double>(members.size());
        meanY /= static_cast<double>(members.size());
        double sxx = 0.0;
        double sxy = 0.0;
        double syy = 0.0;

        for (const Row* row : members)
        {
            const double dx = static_cast<double>(row->accesses) / row->durationS - meanX;
            const double dy = row->powerW - row->baselineW - meanY;
            sxx += dx * dx;
            sxy += dx * dy;
            syy += dy * dy;
        }

        if (sxx <= 0.0)
        {
            continue;
        }

        const double slope = sxy / sxx;
        fit.pJPerAccess = slope * 1e12;
        fit.pJPerByte = fit.width > 0 ? fit.pJPerAccess / static_cast<double>(fit.width) : NaN;
        fit.fixedW = meanY - slope * meanX;

        if (members.size() >= 3 && syy > 0.0)
        {
            fit.r2 = (sxy * sxy) / (sxx * syy);
        }

        fits.push_back(fit);
    }

    return fits;
}

const ProbeFit* FindFit(const std::vector<ProbeFit>& fits, const std::string& operation, const std::string& level,
    const uint64_t footprint, const Variant& variant)
{
    for (const auto& fit : fits)
    {
        if (fit.operation == operation && fit.level == level && (footprint == 0 || fit.footprint == footprint)
            && (variant.width == 0 || fit.width == 0 || fit.width == variant.width) && fit.payload == variant.payload)
        {
            return &fit;
        }
    }

    return nullptr;
}

void WriteFitsCsv(const std::vector<ProbeFit>& fits, const std::string& file)
{
    std::ofstream output(file);
    output << "operation,level,footprint_bytes,width_bytes,payload,duties,points,pj_per_access,pj_per_byte,pj_per_flop,"
        << "fixed_w,r2\n";

    for (const auto& fit : fits)
    {
        output << fit.operation << "," << fit.level << "," << fit.footprint << "," << fit.width << "," << fit.payload << ","
            << fit.duties << "," << fit.points << "," << CsvValue(fit.pJPerAccess) << "," << CsvValue(fit.pJPerByte) << ","
            << CsvValue(fit.pJPerAccess / fit.flopsPerOp) << "," << CsvValue(fit.fixedW) << "," << CsvValue(fit.r2) << "\n";
    }
}

Line FitLine(const std::vector<std::pair<double, double>>& points)
{
    Line line;

    if (points.size() < 2)
    {
        return line;
    }

    const double n = static_cast<double>(points.size());
    double meanX = 0.0;
    double meanY = 0.0;

    for (const auto& [x, y] : points)
    {
        meanX += x / n;
        meanY += y / n;
    }

    double sxx = 0.0;
    double sxy = 0.0;
    double syy = 0.0;

    for (const auto& [x, y] : points)
    {
        sxx += (x - meanX) * (x - meanX);
        sxy += (x - meanX) * (y - meanY);
        syy += (y - meanY) * (y - meanY);
    }

    if (sxx <= 0.0)
    {
        return line;
    }

    line.slope = sxy / sxx;
    line.intercept = meanY - line.slope * meanX;
    line.r2 = points.size() >= 3 && syy > 0.0 ? sxy * sxy / (sxx * syy) : NaN;
    return line;
}

Estimate EstimateEnergy(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const std::string& operation,
    const std::string& level, const uint64_t footprint, const Variant& variant)
{
    const Row* row = SelectRow(rows, operation, level, footprint, variant);
    const bool rowEnergy = row != nullptr && !std::isnan(row->pJPerAccess);

    if (const ProbeFit* fit = FindFit(fits, operation, level, footprint, variant); fit != nullptr
        && !std::isnan(fit->pJPerAccess))
    {
        if (!rowEnergy || std::abs(fit->fixedW) <= 0.5 * (row->powerW - row->baselineW))
        {
            return {fit->pJPerAccess, "fit " + fit->duties};
        }

        return {row->pJPerAccess, "duty " + std::to_string(row->duty) + " (fit not linear: fixed " + Fixed(fit->fixedW, 1)
            + " W)"};
    }

    if (rowEnergy)
    {
        return {row->pJPerAccess, "duty " + std::to_string(row->duty)};
    }

    return {};
}
