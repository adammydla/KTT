#include <algorithm>
#include <map>
#include <tuple>

#include "Analysis.h"

BaselineModel FitBaseline(const std::vector<Row>& rows)
{
    BaselineModel model;
    std::vector<std::pair<double, double>> points; // temperature, power
    double sumW = 0.0;
    double sumClock = 0.0;
    size_t clocks = 0;

    for (const auto& row : rows)
    {
        if (row.level != "baseline" || !row.hasPower)
        {
            continue;
        }

        ++model.runs;
        sumW += row.powerW;
        model.minW = std::fmin(model.minW, row.powerW);
        model.maxW = std::fmax(model.maxW, row.powerW);

        if (!std::isnan(row.temperatureC))
        {
            points.emplace_back(row.temperatureC, row.powerW);
            model.minTemperatureC = std::fmin(model.minTemperatureC, row.temperatureC);
            model.maxTemperatureC = std::fmax(model.maxTemperatureC, row.temperatureC);
        }

        if (!std::isnan(row.smClockMHz))
        {
            sumClock += row.smClockMHz;
            ++clocks;
        }
    }

    if (model.runs == 0)
    {
        return model;
    }

    model.meanW = sumW / static_cast<double>(model.runs);
    model.clockMHz = clocks > 0 ? sumClock / static_cast<double>(clocks) : NaN;
    double sumTemperature = 0.0;

    for (const auto& point : points)
    {
        sumTemperature += point.first;
    }

    model.meanTemperatureC = points.empty() ? NaN : sumTemperature / static_cast<double>(points.size());
    const Line line = FitLine(points); // passes through the means

    if (points.size() < 3 || points.size() != model.runs)
    {
        model.note = "fewer than 3 baseline runs with a temperature, using the mean";
    }
    else if (model.maxTemperatureC - model.minTemperatureC < 2.0)
    {
        model.note = "baseline temperatures within 2 C, using the mean";
    }
    else if (!(line.slope >= 0.0 && line.slope <= 5.0))
    {
        model.note = "fitted slope of " + Fixed(line.slope, 2) + " W/C is implausible, using the mean";
    }
    else
    {
        model.slopeWPerC = line.slope;
        model.temperatureModel = true;
    }

    return model;
}

bool ClockMatched(const Row& row)
{
    return !std::isnan(row.clockDifference) && std::abs(row.clockDifference) <= ClockTolerance;
}

void ComputeEnergies(std::vector<Row>& rows, const BaselineModel& baseline, const bool clockLocked, const bool force)
{
    for (auto& row : rows)
    {
        if (!IsProbeRow(row) || !row.hasPower || row.accesses == 0)
        {
            continue;
        }

        bool withhold = false;
        row.baselineW = baseline.At(row.temperatureC);

        if (!std::isnan(row.smClockMHz) && baseline.clockMHz > 0.0)
        {
            row.clockDifference = (row.smClockMHz - baseline.clockMHz) / baseline.clockMHz;

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
    }
}

std::vector<ProbeFit> FitProbes(const std::vector<Row>& rows, const uint64_t minDuty)
{
    // In the order of the levels, then by operation, footprint and width.
    const std::vector<std::string> levels = {"shared", "L1", "L2", "mixed", "DRAM", "none", "compute"};
    std::map<std::tuple<size_t, std::string, uint64_t, uint64_t>, std::vector<const Row*>> probes;

    for (const auto& row : rows)
    {
        if (IsProbeRow(row) && !std::isnan(row.pJPerAccess) && ClockMatched(row) && row.durationS > 0.0 && row.duty >= minDuty)
        {
            const size_t level = static_cast<size_t>(std::find(levels.cbegin(), levels.cend(), row.level) - levels.cbegin());
            probes[{level, row.operation, row.footprint, row.width}].push_back(&row);
        }
    }

    std::vector<ProbeFit> fits;

    for (auto& [key, members] : probes)
    {
        std::sort(members.begin(), members.end(), [](const Row* a, const Row* b) { return a->duty > b->duty; });
        std::vector<std::pair<double, double>> points; // access rate, power above the baseline
        ProbeFit fit;

        for (const Row* row : members)
        {
            fit.duties += (fit.duties.empty() ? "" : "/") + std::to_string(row->duty);
            points.emplace_back(static_cast<double>(row->accesses) / row->durationS, row->powerW - row->baselineW);
        }

        const Line line = FitLine(points);

        if (std::isnan(line.slope))
        {
            continue;
        }

        fit.operation = members.front()->operation;
        fit.level = members.front()->level;
        fit.footprint = members.front()->footprint;
        fit.width = members.front()->width;
        fit.points = members.size();
        fit.pJPerAccess = line.slope * 1e12;
        fit.fixedW = line.intercept;
        fit.r2 = line.r2;
        fits.push_back(fit);
    }

    return fits;
}
