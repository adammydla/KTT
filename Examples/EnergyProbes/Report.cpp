#include <fstream>
#include <iomanip>
#include <iostream>

#include "Report.h"

namespace
{

void PrintRows(const std::vector<Row>& rows)
{
    std::cout << "\n" << std::left << std::setw(12) << "probe" << std::setw(26) << "op" << std::setw(10) << "footprint"
        << std::setw(11) << "level" << std::right << std::setw(3) << "W" << std::setw(5) << "duty" << std::setw(8) << "P [W]"
        << std::setw(7) << "MHz" << std::setw(5) << "T[C]" << std::setw(9) << "GB/s" << std::setw(9) << "G/s" << std::setw(11)
        << "pJ/access" << "  note\n";

    for (const auto& row : rows)
    {
        const bool probe = IsProbeRow(row);
        const double rate = row.accesses > 0 && probe ? static_cast<double>(row.accesses) / row.durationS * 1e-9 : NaN;
        std::cout << std::left << std::setw(12) << row.probe << std::setw(26) << row.operation << std::setw(10)
            << (row.footprint > 0 ? FormatBytes(row.footprint) : "-") << std::setw(11) << row.level << std::right
            << std::setw(3) << (row.width > 0 ? std::to_string(row.width) : "-") << std::setw(5)
            << (probe ? std::to_string(row.duty) : "-") << std::setw(8) << Fixed(row.powerW, 1) << std::setw(7)
            << Fixed(row.smClockMHz, 0) << std::setw(5) << Fixed(row.temperatureC, 0) << std::setw(9)
            << Fixed(row.bandwidthGBs, 0) << std::setw(9) << Fixed(rate, 1) << std::setw(11) << Fixed(row.pJPerAccess, 1)
            << "  " << row.note << "\n";
    }
}

// The sleep baseline and its temperature model, and whether the probes ran at its clock and settled.
void PrintBaselineAndClock(const std::vector<Row>& rows, const BaselineModel& baseline, const Options& options)
{
    std::cout << "\nBaseline (sleep kernel): " << baseline.runs << " runs, " << Fixed(baseline.minW, 1) << "-"
        << Fixed(baseline.maxW, 1) << " W at " << Fixed(baseline.minTemperatureC, 1) << "-" << Fixed(baseline.maxTemperatureC, 1)
        << " C, SM clock " << Fixed(baseline.clockMHz, 0) << " MHz\n";

    if (baseline.temperatureModel)
    {
        std::cout << "  temperature model: " << Fixed(baseline.meanW, 1) << " W + " << Fixed(baseline.slopeWPerC, 2)
            << " W/C x (T - " << Fixed(baseline.meanTemperatureC, 1) << " C); each probe uses the baseline at its own temperature\n";
    }
    else
    {
        std::cout << "  " << baseline.note << "\n";
    }

    size_t probes = 0;
    size_t matched = 0;
    size_t unsettled = 0;

    for (const auto& row : rows)
    {
        if (IsProbeRow(row) && row.hasPower && row.accesses > 0)
        {
            ++probes;
            matched += ClockMatched(row) ? 1 : 0;
            unsettled += row.powerSettled == 0 ? 1 : 0;
        }
    }

    std::cout << matched << " of " << probes << " probe rows ran within " << Fixed(ClockTolerance * 100.0, 0)
        << " % of the baseline SM clock";

    if (!ClockIsLocked(options))
    {
        std::cout << ". The SM clock was not locked: a row at another clock is compared with a baseline at a different clock "
            << "and voltage, so its energy is biased. Lock the clock for final numbers";
    }

    std::cout << ".\n";

    // KTT's precise measurement ends at --max-time-ms when the power did not settle within --max-power-diff.
    if (unsettled > 0)
    {
        std::cout << "NOTE: in " << unsettled << " configurations the power reading did not settle before --max-time-ms "
            << "(power_settled 0 in the CSV).\n";
    }
}

void PrintFits(const std::vector<ProbeFit>& fits, const Options& options)
{
    std::cout << "\nFit across duties (clock-matched rows, duties >= " << options.fitMinDuty
        << " %): power above baseline = fixed + energy per access x access rate\n  " << std::left << std::setw(26) << "op"
        << std::setw(10) << "footprint" << std::setw(11) << "level" << std::right << std::setw(3) << "W" << std::setw(11)
        << "duties" << std::setw(12) << "pJ/access" << std::setw(11) << "fixed [W]" << "\n";

    for (const auto& fit : fits)
    {
        std::cout << "  " << std::left << std::setw(26) << fit.operation << std::setw(10)
            << (fit.footprint > 0 ? FormatBytes(fit.footprint) : "-") << std::setw(11) << fit.level << std::right
            << std::setw(3) << (fit.width > 0 ? std::to_string(fit.width) : "-") << std::setw(11) << fit.duties
            << std::setw(12) << Fixed(fit.pJPerAccess, 1) << std::setw(11) << Fixed(fit.fixedW, 1)
            << (fit.pJPerAccess < 0.0 ? "  negative: rows not comparable" : "") << "\n";
    }

    if (fits.empty())
    {
        std::cout << "  none: a fit needs two clock-matched duties of a probe (a run at one duty is a counter run or a check)\n";
    }
}

nlohmann::json EnergyTable(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const BaselineModel& baseline,
    const Hardware& hardware, const Options& options, const NvmlMonitor* monitor)
{
    nlohmann::json table;
    table["device"] = hardware.name;
    table["compute_capability"] = std::to_string(hardware.computeCapabilityMajor) + "."
        + std::to_string(hardware.computeCapabilityMinor);
    table["sm_clock_mhz"] = ToJson(baseline.clockMHz);
    table["fit_min_duty_pct"] = options.fitMinDuty;
    table["baseline"] = {{"runs", baseline.runs}, {"mean_w", ToJson(baseline.meanW)},
        {"slope_w_per_c", baseline.slopeWPerC}, {"mean_temperature_c", ToJson(baseline.meanTemperatureC)}};
    // The idle GPU without a kernel: "idle-noctx" before any CUDA context exists (its lowest power state), the others
    // with a context. The predictor's accountings use them.
    table["idle"] = nlohmann::json::array();
    table["fits"] = nlohmann::json::array();

    for (const auto& row : rows)
    {
        if (row.level == "idle" && row.hasPower)
        {
            table["idle"].push_back({{"label", row.probe}, {"power_w", ToJson(row.powerW)},
                {"temperature_c", ToJson(row.temperatureC)}, {"sm_clock_mhz", ToJson(row.smClockMHz)},
                {"memory_clock_mhz", ToJson(row.memoryClockMHz)}});
        }
    }

    for (const auto& fit : fits)
    {
        table["fits"].push_back({{"operation", fit.operation}, {"level", fit.level}, {"footprint_bytes", fit.footprint},
            {"width_bytes", fit.width}, {"duties", fit.duties}, {"pj_per_access", ToJson(fit.pJPerAccess)},
            {"fixed_w", ToJson(fit.fixedW)}, {"r2", ToJson(fit.r2)}});
    }

    // The card and the software the run was made with.
    if (monitor != nullptr)
    {
        const DeviceProvenance provenance = monitor->GetProvenance();
        const auto textOrNull = [](const std::string& text) { return text.empty() ? nlohmann::json() : nlohmann::json(text); };
        table["provenance"] = {{"driver_version", textOrNull(provenance.driverVersion)},
            {"cuda_driver_version", textOrNull(provenance.cudaDriverVersion)}, {"uuid", textOrNull(provenance.uuid)},
            {"serial", textOrNull(provenance.serial)}, {"power_limit_w", ToJson(provenance.powerLimitW)}};
        table["board"] = monitor->GetBoard();
    }

    table["cuda_context"] = DescribeCudaContext();

    return table;
}

// One line per entry: the top-level values, and the elements of the lists (an idle reading, a fit).
void WriteJson(const nlohmann::json& table, const std::string& file)
{
    std::ofstream output(file);
    std::string separator = "{\n";

    for (const auto& [key, value] : table.items())
    {
        output << separator << " " << nlohmann::json(key).dump() << ": ";
        separator = ",\n";

        if (!value.is_array() || value.empty())
        {
            output << value.dump();
            continue;
        }

        std::string inner = "[\n";

        for (const auto& element : value)
        {
            output << inner << "  " << element.dump();
            inner = ",\n";
        }

        output << "\n ]";
    }

    output << "\n}\n";
}

} // namespace

void ReportProbeRun(std::vector<Row>& rows, const BaselineModel& baseline, const Hardware& hardware, const Options& options,
    const NvmlMonitor* monitor)
{
    ComputeEnergies(rows, baseline, ClockIsLocked(options), options.forceEnergy);
    const std::vector<ProbeFit> fits = FitProbes(rows, options.fitMinDuty);
    PrintRows(rows);
    PrintBaselineAndClock(rows, baseline, options);
    PrintFits(fits, options);
    WriteCsv(rows, options.outputPrefix + ".csv");
    WriteJson(EnergyTable(rows, fits, baseline, hardware, options, monitor), options.outputPrefix + "_energy_table.json");
    std::cout << "\nWrote " << options.outputPrefix << ".csv and " << options.outputPrefix << "_energy_table.json; check the "
        << "run with tools/check_run.py " << options.outputPrefix << "\n";
}
