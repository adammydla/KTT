// One row per measured configuration (probe, baseline, reference or idle reading), and its CSV format.
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include "Common.h"
#include "NvmlMonitor.h"

// Numbers reported by the kernel itself, averaged over the launches of a configuration, plus the NVML trace summary.
struct LaunchStats
{
    uint64_t accesses = 0; // per launch
    uint64_t launches = 0;
    double meanActiveSeconds = 0.0; // per launch and working block
    std::string placement; // blocks per SM and their stripes (global probe only, last launch): "ok" or the problem
    double measureS = NaN;  // host time of KTT's measurement of the configuration (its repeated launches)
    int powerSettled = -1;  // 1: KTT's power reading settled before --max-time-ms, 0: it ran out of time, -1: unknown
    std::optional<TraceSummary> trace;
};

using StatsMap = std::map<std::string, LaunchStats>;

struct Row
{
    std::string probe;
    std::string operation;
    std::string level; // L1, L2, DRAM, mixed, shared, none (no-load), compute, baseline, reference, idle
    uint64_t footprint = 0;
    uint64_t width = 0; // bytes per access and thread (0 for compute, baselines and references)
    std::string payload = PayloadName(PayloadRandom);
    uint64_t activeSms = 0; // 0 = all SMs work; otherwise a row of the SM sweep
    double flopsPerOp = NaN; // compute rows
    bool valid = false;
    bool hasPower = false;
    double powerW = NaN;
    double durationS = NaN;
    double energyJ = NaN;
    double smClockMHz = NaN;
    double memoryClockMHz = NaN;
    double temperatureC = NaN;
    double fanPct = NaN; // mean of the NVML trace (second half)
    uint64_t accesses = 0; // per launch
    uint64_t launches = 0;
    double activeS = NaN;
    double bandwidthGBs = NaN;
    double counterPowerW = NaN;
    double tracePowerW = NaN;
    double instantPowerW = NaN;
    double averagePowerW = NaN;
    double minClockMHz = NaN;
    double maxClockMHz = NaN;
    std::string limiters;
    std::string placement;
    uint64_t registers = 0;       // per thread, from KTT's compilation data (0 = unknown)
    uint64_t blocksPerSmFit = 0;  // blocks of the kernel that fit on an SM at once by its registers (0 = unknown)
    double measureS = NaN;        // host time of KTT's measurement (LaunchStats)
    int powerSettled = -1;        // LaunchStats
    double powerSpreadPct = NaN;  // largest deviation of the power over the last --min-time-ms (TraceSummary)
    uint64_t duty = 100;
    double clockDifference = NaN; // relative to the baseline's SM clock
    double baselineW = NaN;       // baseline power at this row's temperature
    double dynamicJ = NaN;
    double pJPerAccess = NaN;     // per access or, for compute rows, per operation of one thread
    double pJPerByte = NaN;
    double pJPerFlop = NaN;
    std::string note;
    std::map<std::string, double> counters;
};

// Rows of probes, as opposed to baselines, the spin reference and idle readings.
bool IsProbeRow(const Row& row);

std::string DutyText(const Row& row);

void WriteCsv(const std::vector<Row>& rows, const std::string& file);

// Reads the rows of an earlier run (WriteCsv's format) for --analyze. Only what was measured is kept; the baseline, the
// energies and their notes are computed again. The placement check is kept in the notes only: rows of the probes that
// check it and have no placement note were "ok".
std::vector<Row> ReadCsv(const std::string& file);
