#include <cmath>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <stdexcept>

#include "Rows.h"

namespace
{

// A column of the CSV: its name, and how a row's cell is written and read (text; empty = missing).
struct Column
{
    std::string name;
    std::function<std::string(const Row&)> write;
    std::function<void(Row&, const std::string&)> read; // empty: computed by the analysis, not read back
};

Column Number(const std::string& name, double Row::* member, const double scale = 1.0, const bool measured = true)
{
    return {name, [=](const Row& row) { return CsvValue(row.*member * scale); },
        measured ? [=](Row& row, const std::string& cell) { row.*member = cell.empty() ? NaN : std::stod(cell) / scale; }
        : std::function<void(Row&, const std::string&)>()};
}

// 0 is written as an empty cell when it means "unknown".
Column Count(const std::string& name, uint64_t Row::* member, const bool emptyIfZero = false)
{
    return {name, [=](const Row& row) { return emptyIfZero && row.*member == 0 ? "" : std::to_string(row.*member); },
        [=](Row& row, const std::string& cell) { row.*member = cell.empty() ? 0 : static_cast<uint64_t>(std::stod(cell)); }};
}

Column Text(const std::string& name, std::string Row::* member)
{
    return {name, [=](const Row& row) { return CsvText(row.*member); },
        [=](Row& row, const std::string& cell) { row.*member = cell; }};
}

const std::vector<Column>& Columns()
{
    // payload and active_sms are constants: every probe runs on random data on all SMs (earlier versions had studies
    // on zeros and on fewer SMs, and the tools select by these columns).
    static const std::vector<Column> columns = {
        Text("probe", &Row::probe), Text("operation", &Row::operation), Text("level", &Row::level),
        Count("footprint_bytes", &Row::footprint), Count("width_bytes", &Row::width),
        {"payload", [](const Row& row) { return IsProbeRow(row) ? "random" : ""; }, {}},
        {"active_sms", [](const Row&) { return "all"; }, {}},
        {"duty_pct", [](const Row& row) { return IsProbeRow(row) ? std::to_string(row.duty) : "-"; },
            [](Row& row, const std::string& cell) { row.duty = cell == "-" || cell.empty() ? 100 : std::stoull(cell); }},
        {"valid", [](const Row& row) { return row.valid ? "1" : "0"; },
            [](Row& row, const std::string& cell) { row.valid = cell == "1"; }},
        Number("power_w", &Row::powerW), Number("baseline_w", &Row::baselineW, 1.0, false),
        Number("duration_s", &Row::durationS), Number("energy_j", &Row::energyJ), Number("sm_clock_mhz", &Row::smClockMHz),
        Number("clock_difference_pct", &Row::clockDifference, 100.0, false),
        Number("memory_clock_mhz", &Row::memoryClockMHz), Number("temperature_c", &Row::temperatureC),
        Number("fan_pct", &Row::fanPct), Count("accesses_per_launch", &Row::accesses), Count("launches", &Row::launches),
        Number("active_s", &Row::activeS), Number("bandwidth_gbs", &Row::bandwidthGBs),
        Number("counter_power_w", &Row::counterPowerW), Number("trace_power_w", &Row::tracePowerW),
        Number("instant_power_w", &Row::instantPowerW), Number("average_power_w", &Row::averagePowerW),
        Number("trace_sm_clock_min_mhz", &Row::minClockMHz), Number("trace_sm_clock_max_mhz", &Row::maxClockMHz),
        Text("clock_limiters", &Row::limiters), Number("dynamic_energy_j", &Row::dynamicJ, 1.0, false),
        Number("pj_per_access", &Row::pJPerAccess, 1.0, false),
        Count("registers", &Row::registers, true), Count("blocks_per_sm_fit", &Row::blocksPerSmFit, true),
        Number("measure_s", &Row::measureS),
        {"power_settled", [](const Row& row) { return row.powerSettled >= 0 ? std::to_string(row.powerSettled) : ""; },
            [](Row& row, const std::string& cell) { row.powerSettled = cell.empty() ? -1 : std::stoi(cell); }},
        Number("power_spread_pct", &Row::powerSpreadPct),
        // Read back: only the notes of the measurement (placement, failed runs); the others come from the analysis.
        {"note", [](const Row& row) { return CsvText(row.note); }, [](Row& row, const std::string& cell)
        {
            std::stringstream notes(cell);
            std::string note;

            while (std::getline(notes, note, ';'))
            {
                note.erase(0, note.find_first_not_of(' '));

                if (note.rfind("placement: ", 0) == 0 || note == "run failed")
                {
                    AppendNote(row.note, note);
                }
            }
        }},
    };
    return columns;
}

std::vector<std::string> Split(const std::string& line)
{
    std::vector<std::string> cells;
    std::stringstream stream(line);
    std::string cell;

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

bool IsProbeRow(const Row& row)
{
    return row.level != "baseline" && row.level != "reference" && row.level != "idle";
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
    std::string separator;

    for (const Column& column : Columns())
    {
        output << separator << column.name;
        separator = ",";
    }

    for (const auto& name : counterNames)
    {
        output << "," << name;
    }

    output << "\n";

    for (const auto& row : rows)
    {
        separator.clear();

        for (const Column& column : Columns())
        {
            output << separator << column.write(row);
            separator = ",";
        }

        for (const auto& name : counterNames)
        {
            const auto counter = row.counters.find(name);
            output << "," << (counter != row.counters.cend() ? CsvValue(counter->second) : "");
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

    std::string line;
    std::getline(input, line);
    const std::vector<std::string> header = Split(line);
    std::map<std::string, size_t> position;

    for (size_t i = 0; i < header.size(); ++i)
    {
        position[header[i]] = i;
    }

    const size_t note = position.at("note");
    std::vector<Row> rows;

    while (std::getline(input, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::vector<std::string> cells = Split(line);

        // Runs from before 2026-11-11 wrote notes with a comma unquoted: such a row has more cells than the header, and
        // the cells after the note (the counters) are shifted. The extra cells belong to the note.
        for (; cells.size() > header.size(); cells.erase(cells.begin() + static_cast<std::ptrdiff_t>(note) + 1))
        {
            cells[note] += "," + cells[note + 1];
        }

        cells.resize(header.size());
        const auto cell = [&](const std::string& name) { return position.count(name) > 0 ? cells[position.at(name)] : ""; };

        if ((cell("payload") != "" && cell("payload") != "random") || (cell("active_sms") != "" && cell("active_sms") != "all"))
        {
            continue;
        }

        Row row;

        for (const Column& column : Columns())
        {
            if (column.read)
            {
                column.read(row, cell(column.name));
            }
        }

        row.hasPower = !std::isnan(row.powerW);

        for (size_t i = note + 1; i < header.size(); ++i)
        {
            if (!cells[i].empty())
            {
                row.counters[header[i]] = std::stod(cells[i]);
            }
        }

        rows.push_back(row);
    }

    return rows;
}
