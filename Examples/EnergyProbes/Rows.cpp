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

#include "Rows.h"

bool IsProbeRow(const Row& row)
{
    return row.level != "baseline" && row.level != "reference" && row.level != "idle";
}

std::string DutyText(const Row& row)
{
    return IsProbeRow(row) ? std::to_string(row.duty) : "-";
}

void WriteCsv(const std::vector<Row>& rows, const std::string& file)
{
    std::set<std::string> counterNames;

    for (const auto& row : rows)
    {
        for (const auto& counter : row.counters)
        {
            counterNames.insert(counter.first);
        }
    }

    std::ofstream output(file);
    output << "probe,operation,level,footprint_bytes,width_bytes,payload,active_sms,duty_pct,valid,power_w,baseline_w,"
        << "duration_s,energy_j,sm_clock_mhz,clock_difference_pct,memory_clock_mhz,temperature_c,fan_pct,accesses_per_launch,"
        << "launches,"
        << "active_s,bandwidth_gbs,counter_power_w,trace_power_w,instant_power_w,average_power_w,trace_sm_clock_min_mhz,"
        << "trace_sm_clock_max_mhz,clock_limiters,dynamic_energy_j,pj_per_access,pj_per_byte,flops_per_op,pj_per_flop,"
        << "registers,blocks_per_sm_fit,measure_s,power_settled,power_spread_pct,note";

    for (const auto& name : counterNames)
    {
        output << "," << name;
    }

    output << "\n";

    for (const auto& row : rows)
    {
        output << CsvText(row.probe) << "," << CsvText(row.operation) << "," << CsvText(row.level) << "," << row.footprint
            << "," << row.width << "," << (IsProbeRow(row) ? CsvText(row.payload) : "") << "," << (row.activeSms > 0 ? std::to_string(row.activeSms) : "all") << ","
            << DutyText(row) << ","
            << row.valid << "," << CsvValue(row.powerW) << "," << CsvValue(row.baselineW) << "," << CsvValue(row.durationS) << ","
            << CsvValue(row.energyJ) << ","
            << CsvValue(row.smClockMHz) << "," << CsvValue(row.clockDifference * 100.0) << ","
            << CsvValue(row.memoryClockMHz) << "," << CsvValue(row.temperatureC) << "," << CsvValue(row.fanPct) << ","
            << row.accesses << "," << row.launches << "," << CsvValue(row.activeS) << "," << CsvValue(row.bandwidthGBs) << ","
            << CsvValue(row.counterPowerW) << "," << CsvValue(row.tracePowerW) << "," << CsvValue(row.instantPowerW) << ","
            << CsvValue(row.averagePowerW) << "," << CsvValue(row.minClockMHz) << ","
            << CsvValue(row.maxClockMHz) << "," << CsvText(row.limiters) << "," << CsvValue(row.dynamicJ) << ","
            << CsvValue(row.pJPerAccess) << "," << CsvValue(row.pJPerByte) << "," << CsvValue(row.flopsPerOp) << ","
            << CsvValue(row.pJPerFlop) << "," << (row.registers > 0 ? std::to_string(row.registers) : "") << ","
            << (row.blocksPerSmFit > 0 ? std::to_string(row.blocksPerSmFit) : "") << "," << CsvValue(row.measureS) << ","
            << (row.powerSettled >= 0 ? std::to_string(row.powerSettled) : "") << "," << CsvValue(row.powerSpreadPct) << ","
            << CsvText(row.note);

        for (const auto& name : counterNames)
        {
            const auto iterator = row.counters.find(name);
            output << "," << (iterator != row.counters.cend() ? CsvValue(iterator->second) : "");
        }

        output << "\n";
    }
}

std::vector<Row> ReadCsv(const std::string& file)
{
    std::ifstream input(file);

    if (!input)
    {
        throw std::runtime_error("Unable to open " + file);
    }

    auto split = [](const std::string& line)
    {
        std::vector<std::string> fields;
        std::stringstream stream(line);
        std::string field;

        while (std::getline(stream, field, ','))
        {
            fields.push_back(field);
        }

        if (!line.empty() && line.back() == ',')
        {
            fields.emplace_back();
        }

        return fields;
    };

    std::string line;
    std::getline(input, line);
    const std::vector<std::string> header = split(line);
    std::map<std::string, size_t> column;

    for (size_t i = 0; i < header.size(); ++i)
    {
        column[header[i]] = i;
    }

    const size_t firstCounter = column.at("note") + 1;
    std::vector<Row> rows;

    while (std::getline(input, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::vector<std::string> fields = split(line);

        // Runs from before 2026-11-11 wrote notes with a comma (a placement or register note) unquoted: such a row has
        // more cells than the header, and the cells after the note (the counters) are shifted. The extra cells belong to
        // the note.
        if (fields.size() > header.size())
        {
            const size_t note = column.at("note");
            const size_t extra = fields.size() - header.size();

            for (size_t i = 1; i <= extra; ++i)
            {
                fields[note] += "," + fields[note + i];
            }

            fields.erase(fields.begin() + static_cast<std::ptrdiff_t>(note + 1),
                fields.begin() + static_cast<std::ptrdiff_t>(note + 1 + extra));
        }

        fields.resize(header.size());
        auto text = [&](const std::string& name) { return fields[column.at(name)]; };
        auto value = [&](const std::string& name)
        {
            const std::string& field = fields[column.at(name)];
            return field.empty() ? NaN : std::stod(field);
        };

        Row row;
        row.probe = text("probe");
        row.operation = text("operation");
        row.level = text("level");
        row.footprint = std::stoull(text("footprint_bytes"));
        row.width = std::stoull(text("width_bytes"));
        row.payload = text("payload").empty() ? PayloadName(PayloadRandom) : text("payload");
        row.activeSms = text("active_sms") == "all" ? 0 : std::stoull(text("active_sms"));
        row.duty = text("duty_pct") == "-" ? 100 : std::stoull(text("duty_pct"));
        row.valid = text("valid") == "1";
        row.powerW = value("power_w");
        row.hasPower = !std::isnan(row.powerW);
        row.durationS = value("duration_s");
        row.energyJ = value("energy_j");
        row.smClockMHz = value("sm_clock_mhz");
        row.memoryClockMHz = value("memory_clock_mhz");
        row.temperatureC = value("temperature_c");
        row.fanPct = column.count("fan_pct") > 0 ? value("fan_pct") : NaN; // CSVs from before 2026-11-06 lack it
        row.accesses = std::stoull(text("accesses_per_launch"));
        row.launches = std::stoull(text("launches"));
        row.activeS = value("active_s");
        row.bandwidthGBs = value("bandwidth_gbs");
        row.counterPowerW = value("counter_power_w");
        row.tracePowerW = value("trace_power_w");
        row.instantPowerW = value("instant_power_w");
        row.averagePowerW = value("average_power_w");
        row.minClockMHz = value("trace_sm_clock_min_mhz");
        row.maxClockMHz = value("trace_sm_clock_max_mhz");
        row.limiters = text("clock_limiters");
        row.flopsPerOp = value("flops_per_op");
        // Columns of runs since 2026-11-06.
        auto optional = [&](const std::string& name) { return column.count(name) > 0 ? value(name) : NaN; };
        row.registers = std::isnan(optional("registers")) ? 0 : static_cast<uint64_t>(optional("registers"));
        row.blocksPerSmFit = std::isnan(optional("blocks_per_sm_fit")) ? 0 : static_cast<uint64_t>(optional("blocks_per_sm_fit"));
        row.measureS = optional("measure_s");
        row.powerSettled = std::isnan(optional("power_settled")) ? -1 : static_cast<int>(optional("power_settled"));
        row.powerSpreadPct = optional("power_spread_pct");

        // Notes of the measurement (placement, failed runs); the others come from the analysis.
        std::stringstream notes(text("note"));
        std::string note;

        while (std::getline(notes, note, ';'))
        {
            note.erase(0, note.find_first_not_of(' '));

            if (note.rfind("placement: ", 0) == 0)
            {
                row.placement = note.substr(11);
                AppendNote(row.note, note);
            }
            else if (note == "run failed")
            {
                AppendNote(row.note, note);
            }
        }

        const bool checksPlacement = row.probe == "global" || row.probe == "constant" || row.probe == "texture"
            || row.probe == "mixed" || row.probe == "write" || (row.probe == "sm-sweep" && row.level != "shared" && row.level != "compute");

        if (row.placement.empty() && checksPlacement)
        {
            row.placement = "ok";
        }

        for (size_t i = firstCounter; i < header.size(); ++i)
        {
            if (!fields[i].empty())
            {
                row.counters[header[i]] = std::stod(fields[i]);
            }
        }

        rows.push_back(row);
    }

    return rows;
}
