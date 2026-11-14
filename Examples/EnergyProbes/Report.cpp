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

#include "Report.h"
#include "Studies.h"

void PrintRows(const std::vector<Row>& rows)
{
    std::cout << "\n" << std::left
        << std::setw(14) << "probe" << std::setw(17) << "op" << std::setw(10) << "footprint" << std::setw(9) << "level"
        << std::right << std::setw(3) << "W" << std::setw(5) << "data" << std::setw(4) << "SMs" << std::setw(5) << "duty"
        << std::setw(8) << "P [W]" << std::setw(9) << "ctr [W]" << std::setw(7) << "MHz" << std::setw(5) << "T[C]"
        << std::setw(9) << "GB/s" << std::setw(9) << "G/s" << std::setw(11) << "pJ/access" << std::setw(8) << "pJ/B"
        << "  note\n";

    for (const auto& row : rows)
    {
        const double rate = row.accesses > 0 && IsProbeRow(row) ? static_cast<double>(row.accesses) / row.durationS * 1e-9
            : NaN;

        std::cout << std::left
            << std::setw(14) << row.probe << std::setw(17) << row.operation
            << std::setw(10) << (row.footprint > 0 ? FormatBytes(row.footprint) : "-") << std::setw(9) << row.level
            << std::right << std::setw(3) << (row.width > 0 ? std::to_string(row.width) : "-")
            << std::setw(5) << (IsProbeRow(row) ? row.payload.substr(0, 4) : "-")
            << std::setw(4) << (row.activeSms > 0 ? std::to_string(row.activeSms) : "all")
            << std::setw(5) << DutyText(row) << std::setw(8) << Fixed(row.powerW, 1)
            << std::setw(9) << Fixed(row.counterPowerW, 1)
            << std::setw(7) << Fixed(row.smClockMHz, 0) << std::setw(5) << Fixed(row.temperatureC, 0)
            << std::setw(9) << Fixed(row.bandwidthGBs, 0) << std::setw(9) << Fixed(rate, 1)
            << std::setw(11) << Fixed(row.pJPerAccess, 1) << std::setw(8) << Fixed(row.pJPerByte, 2)
            << "  " << row.note << "\n";
    }
}

void PrintCounters(const std::vector<Row>& rows)
{
    const bool anyCounters = std::any_of(rows.cbegin(), rows.cend(), [](const Row& row) { return !row.counters.empty(); });

    if (!anyCounters)
    {
        return;
    }

    auto get = [](const Row& row, const std::string& name)
    {
        const auto iterator = row.counters.find(name);
        return iterator != row.counters.cend() ? iterator->second : NaN;
    };

    // Sectors (or shared wavefronts) per requested sector, and warp instructions per warp access (per warp instruction of
    // the operation for compute rows). Counters come from one replay launch, accesses are the mean per launch. For the
    // constant probe: constant-cache (IDC) requests per warp access, and the share of them sent to the L1.5. sec/req: L2
    // sectors per L2 request from L1 (4 for coalesced 16-byte accesses, 1 when every sector is a request of its own).
    // (dram__cycles_active is not used: on an RTX 3090 it was exactly 4 per DRAM sector for every access pattern.)
    std::cout << "\nTraffic from CUPTI counters (profiled replays, approximate; ratios per requested sector):\n" << std::left
        << std::setw(14) << "probe" << std::setw(17) << "op" << std::setw(10) << "footprint" << std::setw(9) << "level"
        << std::right << std::setw(3) << "W" << std::setw(5) << "duty" << std::setw(9) << "L1 hit%" << std::setw(9) << "L2 hit%"
        << std::setw(8) << "L1" << std::setw(8) << "L2" << std::setw(8) << "DRAM" << std::setw(8) << "smem" << std::setw(10)
        << "inst/acc" << std::setw(8) << "IDC" << std::setw(9) << "to L1.5" << std::setw(9) << "sec/req" << "\n";

    for (const auto& row : rows)
    {
        if (row.counters.empty())
        {
            continue;
        }

        const double warpAccesses = static_cast<double>(row.accesses) / 32.0;
        const double requestedSectors = static_cast<double>(row.accesses * row.width) / SectorBytes;
        auto ratio = [&](const double value)
        {
            return requestedSectors > 0.0 ? value / requestedSectors : NaN;
        };

        const double l1 = get(row, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")
            + get(row, "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")
            + get(row, "l1tex__t_sectors_pipe_tex_mem_texture.sum");
        // Atomics are operations of their own in L2, neither reads nor writes.
        const double l2 = get(row, "lts__t_sectors_op_read.sum") + get(row, "lts__t_sectors_op_write.sum")
            + get(row, "lts__t_sectors_srcunit_tex_op_atom.sum") + get(row, "lts__t_sectors_srcunit_tex_op_red.sum");
        const double dram = get(row, "dram__sectors_read.sum") + get(row, "dram__sectors_write.sum");
        // A conflict-free 512-byte warp access is 4 wavefronts, the same number as its 16 sectors / 4.
        const double shared = 4.0 * (get(row, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum")
            + get(row, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum")
            + get(row, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom.sum"));
        const double constantRequests = get(row, "idc__requests.sum");
        const double l2Requests = get(row, "lts__t_requests_srcunit_tex_op_read.sum")
            + get(row, "lts__t_requests_srcunit_tex_op_write.sum");

        std::cout << std::left
            << std::setw(14) << row.probe << std::setw(17) << row.operation
            << std::setw(10) << (row.footprint > 0 ? FormatBytes(row.footprint) : "-") << std::setw(9) << row.level
            << std::right << std::setw(3) << (row.width > 0 ? std::to_string(row.width) : "-") << std::setw(5) << DutyText(row)
            << std::setw(9) << Fixed(get(row, "l1tex__t_sector_hit_rate.pct"), 1)
            << std::setw(9) << Fixed(get(row, "lts__t_sector_hit_rate.pct"), 1)
            << std::setw(8) << Fixed(ratio(l1), 3) << std::setw(8) << Fixed(ratio(l2), 3) << std::setw(8) << Fixed(ratio(dram), 3)
            << std::setw(8) << Fixed(ratio(shared), 3)
            << std::setw(10) << Fixed(warpAccesses > 0.0 ? get(row, "smsp__inst_executed.sum") / warpAccesses : NaN, 2)
            << std::setw(8) << Fixed(warpAccesses > 0.0 ? constantRequests / warpAccesses : NaN, 3)
            << std::setw(9) << Fixed(constantRequests > 0.0 ? get(row, "idc__requests_lookup_miss_tag_miss.sum")
                / constantRequests : NaN, 3)
            << std::setw(9) << Fixed(l2Requests > 0.0 ? l2 / l2Requests : NaN, 2) << "\n";
    }
}

namespace
{

// "clock-matched", or how far the row's SM clock was from the baseline's.
std::string ClockText(const Row& row)
{
    if (std::isnan(row.clockDifference))
    {
        return std::string("clock unknown");
    }

    return ClockMatched(row) ? std::string("clock-matched")
        : "clock " + SignedPercent(row.clockDifference) + " vs baseline: BIASED";
}

// One line of the path energies: the row's energy per byte, or why there is none.
void PrintPathRow(const std::string& label, const Row* row)
{
    std::cout << "  " << std::left << std::setw(40) << label << std::right;

    if (row == nullptr)
    {
        std::cout << "not measured\n";
    }
    else if (std::isnan(row->pJPerByte))
    {
        std::cout << "withheld (" << row->note << ")\n";
    }
    else
    {
        std::cout << std::setw(8) << Fixed(row->pJPerByte, 2) << " pJ/B   duty " << std::setw(3) << row->duty << " %, "
            << ClockText(*row) << "\n";
    }
}

bool HasEnergy(const Row* row)
{
    return row != nullptr && !std::isnan(row->pJPerByte);
}

// A difference is only as good as both rows: flag different duties (slightly different instruction mix) and clocks.
std::string PairNote(const Row* a, const Row* b)
{
    std::string note;

    if (a->duty != b->duty)
    {
        note += "  (rows at duty " + std::to_string(a->duty) + " % and " + std::to_string(b->duty) + " %)";
    }

    if (!ClockMatched(*a) || !ClockMatched(*b))
    {
        note += "  BIASED: clock differs from baseline";
    }

    return note;
}

// The sleep baseline and its temperature model, and whether the probes ran at its clock.
void PrintBaselineAndClock(const std::vector<Row>& rows, const BaselineModel& baseline, const Options& options,
    const NvmlMonitor* monitor)
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

        if (baseline.meanW > 0.0 && (baseline.maxW - baseline.minW) / baseline.meanW > 0.02)
        {
            std::cout << "WARNING: baseline runs differ by more than 2 %, the GPU was probably not in a steady thermal state.\n";
        }
    }

    const bool clockLocked = ClockIsLocked(options);
    std::string clockState = "SM clock not locked";

    if (options.lockClockMHz.has_value())
    {
        clockState = "SM clock locked at " + std::to_string(options.lockClockMHz.value()) + " MHz";
    }
    else if (options.clockLockedExternally)
    {
        clockState = "SM clock locked externally (--clock-locked)";
    }

    std::cout << (monitor != nullptr && !std::isnan(monitor->GetPowerLimitW())
        ? "Power limit: " + Fixed(monitor->GetPowerLimitW(), 0) + " W, " : std::string("")) << clockState << "\n";
    size_t probeRows = 0;
    size_t matchedRows = 0;

    for (const auto& row : rows)
    {
        if (IsProbeRow(row) && row.hasPower && row.accesses > 0)
        {
            ++probeRows;
            matchedRows += ClockMatched(row) ? 1 : 0;
        }
    }

    std::cout << matchedRows << " of " << probeRows << " probe rows ran within " << Fixed(ClockTolerance * 100.0, 0)
        << " % of the baseline SM clock.\n";

    // KTT's precise measurement ends at --max-time-ms when the power did not settle within --max-power-diff; it only
    // warns on the console. The rows record it (power_settled, power_spread_pct), and tools/check_run.py checks it.
    size_t judged = 0;
    size_t unsettled = 0;
    double worstSpread = 0.0;

    for (const auto& row : rows)
    {
        if (row.hasPower && row.powerSettled >= 0)
        {
            ++judged;
            unsettled += row.powerSettled == 0 ? 1 : 0;
            worstSpread = row.powerSettled == 0 && !std::isnan(row.powerSpreadPct) ? std::max(worstSpread, row.powerSpreadPct)
                : worstSpread;
        }
    }

    if (unsettled > 0)
    {
        std::cout << "NOTE: in " << unsettled << " of " << judged << " configurations KTT's power reading did not settle within "
            << "the precision target before --max-time-ms (largest spread over the last --min-time-ms: "
            << Fixed(worstSpread, 1) << " %); their rows have power_settled 0. A queue timeout must allow every "
            << "configuration --max-time-ms (the dry run's upper bound).\n";
    }

    if (!clockLocked)
    {
        std::cout << "NOTE: the SM clock was not locked. A row at another clock than the baseline is compared with a baseline\n"
            << "power measured at a different clock and voltage, so its energy is biased (marked below). Lower duty rows\n"
            << "draw less power and are more likely to run at the baseline clock. Lock the clock for final numbers.\n";
    }
}

// The best row of every load and store probe, per byte.
void PrintPathEnergies(const std::vector<Row>& rows, const Footprints& footprints)
{
    const Row* l1 = SelectRow(rows, "ld", "L1", footprints.l1);
    const Row* l2 = SelectRow(rows, "ld", "L2", footprints.l2);
    const Row* dram = SelectRow(rows, "ld", "DRAM", footprints.dram);
    const Row* l1Strong = SelectRow(rows, "ld.ca", "L1", footprints.l1);
    const Row* l2Direct = SelectRow(rows, "ld.cg", "L2", footprints.l2);
    const Row* reference = SelectRow(rows, "no-load", "none");
    const Row* shared = SelectRow(rows, "ld.shared", "shared");

    std::cout << "\nPath energies (dynamic energy above baseline, per byte read; best row per probe):\n";

    PrintPathRow("shared memory hit", shared);
    PrintPathRow("L1 hit", l1);
    PrintPathRow("L1 hit, ld.ca (MT4G-style strong load)", l1Strong);
    PrintPathRow("L1 hit, ld.nc (read-only path)", SelectRow(rows, "ld.nc", "L1", footprints.l1));
    PrintPathRow("L2 hit (L1 thrashed)", l2);
    PrintPathRow("L2 hit (L1 bypassed, ld.cg)", l2Direct);
    PrintPathRow("L2 hit, ld.nc (read-only path)", SelectRow(rows, "ld.nc", "L2", footprints.l2));
    PrintPathRow("DRAM", dram);
    PrintPathRow("shared memory store", SelectRow(rows, "st.shared", "shared"));
    PrintPathRow("store, L2 footprint", SelectRow(rows, "st", "L2", footprints.l2));
    PrintPathRow("store, DRAM footprint", SelectRow(rows, "st", "DRAM", footprints.dram));
    PrintPathRow("loop without memory access", reference);
}

// The fits across duties of the main probes: cost per access and fixed power.
void PrintFitSummary(const std::vector<ProbeFit>& fits, const Footprints& footprints, const Options& options)
{
    std::cout << "\nFit across intensities (clock-matched rows, duties >= " << options.fitMinDuty
        << " %, random data, " << DefaultWidth << "-byte accesses): power above baseline = fixed + cost per access x rate\n";

    std::vector<ProbeFit> mainFits;

    for (const auto& fit : fits)
    {
        if (fit.payload == PayloadName(PayloadRandom) && fit.width == DefaultWidth)
        {
            mainFits.push_back(fit);
        }
    }

    if (mainFits.empty())
    {
        std::cout << "  not available: needs at least two clock-matched intensities of at least " << options.fitMinDuty
            << " % of the same probe\n";
    }
    else
    {
        std::cout << "  " << std::left << std::setw(11) << "op" << std::setw(10) << "footprint" << std::setw(8) << "level"
            << std::setw(12) << "duties" << std::right << std::setw(9) << "pJ/B" << std::setw(12) << "pJ/access"
            << std::setw(11) << "fixed [W]" << std::setw(7) << "R^2" << "\n";

        for (const auto& fit : mainFits)
        {
            std::cout << "  " << std::left << std::setw(11) << fit.operation
                << std::setw(10) << (fit.footprint > 0 ? FormatBytes(fit.footprint) : "-") << std::setw(8) << fit.level
                << std::setw(12) << fit.duties << std::right << std::setw(9) << Fixed(fit.pJPerByte, 2)
                << std::setw(12) << Fixed(fit.pJPerAccess, 1) << std::setw(11) << Fixed(fit.fixedW, 1)
                << std::setw(7) << Fixed(fit.r2, 3) << (fit.pJPerByte < 0.0 ? "  negative: rows not comparable" : "") << "\n";
        }

        const ProbeFit* l1Fit = FindFit(fits, "ld", "L1", footprints.l1);
        const ProbeFit* l2Fit = FindFit(fits, "ld", "L2", footprints.l2);
        const ProbeFit* dramFit = FindFit(fits, "ld", "DRAM", footprints.dram);

        if (l1Fit != nullptr && l2Fit != nullptr)
        {
            std::cout << "  marginal L2 - L1  : " << Fixed(l2Fit->pJPerByte - l1Fit->pJPerByte, 2) << " pJ/B\n";
        }

        if (l2Fit != nullptr && dramFit != nullptr)
        {
            std::cout << "  marginal DRAM - L2: " << Fixed(dramFit->pJPerByte - l2Fit->pJPerByte, 2) << " pJ/B\n";
        }

        std::cout << "  'fixed' is power that comes with running the probe but does not scale with its access rate.\n";
    }
}

// Differences between levels at equal duty, where only the footprint differs.
void PrintIncrementalEnergies(const std::vector<Row>& rows, const Footprints& footprints)
{
    std::cout << "\nIncremental energies (same SASS at equal duty, only the footprint differs):\n";

    // Each difference uses its own pair of rows at a common duty (see PairForDifference), not the rows printed above.
    auto difference = [&](const std::string& label, const std::pair<const Row*, const Row*>& pair, const bool perAccess,
        const std::string& remark)
    {
        const auto [lower, upper] = pair;

        if (!HasEnergy(lower) || !HasEnergy(upper))
        {
            return;
        }

        std::cout << "  " << std::left << std::setw(17) << label << std::right << ": ";

        if (perAccess)
        {
            std::cout << Fixed(upper->pJPerAccess - lower->pJPerAccess, 1) << " pJ/access";
        }
        else
        {
            std::cout << Fixed(upper->pJPerByte - lower->pJPerByte, 2) << " pJ/B";
        }

        std::cout << "   duty " << upper->duty << " %" << remark << PairNote(lower, upper) << "\n";
    };

    difference("L2 - L1", PairForDifference(rows, "ld", "L1", footprints.l1, "ld", "L2", footprints.l2), false, "");
    difference("DRAM - L2", PairForDifference(rows, "ld", "L2", footprints.l2, "ld", "DRAM", footprints.dram), false, "");
    difference("L1 - no-load", PairForDifference(rows, "no-load", "none", 0, "ld", "L1", footprints.l1), true,
        "  (removes loop and address arithmetic, approximate)");
    difference("shared - no-load", PairForDifference(rows, "no-load", "none", 0, "ld.shared", "shared", 0), true,
        "  (approximate)");
}

// Bandwidths, clocks and limiters, block placement, the NVML readings against each other, the spin reference.
void PrintSanityChecks(const std::vector<Row>& rows, const Footprints& footprints, const BaselineModel& baseline,
    const Options& options, const NvmlMonitor* monitor)
{
    const bool clockLocked = ClockIsLocked(options);
    const Row* l1 = SelectRow(rows, "ld", "L1", footprints.l1);
    const Row* l2 = SelectRow(rows, "ld", "L2", footprints.l2);
    const Row* dram = SelectRow(rows, "ld", "DRAM", footprints.dram);

    std::cout << "\nSanity checks:\n";

    if (HasEnergy(l1) && HasEnergy(l2) && HasEnergy(dram))
    {
        const bool ordered = l1->pJPerByte < l2->pJPerByte && l2->pJPerByte < dram->pJPerByte;
        std::cout << "  energy  L1 < L2 < DRAM     : " << (ordered ? "ok" : "VIOLATED") << "\n";
    }
    else
    {
        std::cout << "  energy  L1 < L2 < DRAM     : not available (energies missing or withheld)\n";
    }

    const Row* l1Full = FullIntensityRow(rows, "ld", "L1", footprints.l1);
    const Row* l2Full = FullIntensityRow(rows, "ld", "L2", footprints.l2);
    const Row* dramFull = FullIntensityRow(rows, "ld", "DRAM", footprints.dram);

    if (l1Full != nullptr && l2Full != nullptr && dramFull != nullptr)
    {
        const bool ordered = l1Full->bandwidthGBs > l2Full->bandwidthGBs && l2Full->bandwidthGBs > dramFull->bandwidthGBs;
        std::cout << "  bandwidth L1 > L2 > DRAM   : " << (ordered ? "ok" : "VIOLATED") << " (rows at duty "
            << l1Full->duty << " %)\n";
    }

    double minClock = std::numeric_limits<double>::max();
    double maxClock = 0.0;

    for (const auto& row : rows)
    {
        // Without a kernel the GPU may lower its clock; that is not a problem of the measurement.
        if (!std::isnan(row.smClockMHz) && row.level != "idle")
        {
            minClock = std::min(minClock, row.smClockMHz);
            maxClock = std::max(maxClock, row.smClockMHz);
        }
    }

    if (maxClock > 0.0)
    {
        const bool stable = (maxClock - minClock) / maxClock <= ClockTolerance;
        std::cout << "  SM clock range " << Fixed(minClock, 0) << " - " << Fixed(maxClock, 0) << " MHz : "
            << (stable ? "ok" : "VARIES") << "\n";

        if (!stable)
        {
            const uint64_t suggestion = static_cast<uint64_t>(minClock / 50.0) * 50 - 50;
            std::cout << "    " << (clockLocked ? "Some rows ran below the locked clock (throttled). Lock it lower"
                : "Lock the clock below the lowest value") << ", e.g. run with sudo and --lock-clock " << suggestion
                << ", or: sudo nvidia-smi -i <gpu> -lgc " << suggestion << "," << suggestion << "\n";
        }
    }

    std::string limited;

    for (const auto& row : rows)
    {
        if (!row.limiters.empty())
        {
            limited += "\n    " + row.probe + " " + row.operation + (row.footprint > 0 ? " " + FormatBytes(row.footprint) : "")
                + ": " + row.limiters;
        }
    }

    std::cout << "  clock limiters (NVML)      : " << (monitor == nullptr ? "not monitored" : (limited.empty() ? "none" : limited))
        << "\n";

    size_t placedRows = 0;
    size_t misplacedRows = 0;

    for (const auto& row : rows)
    {
        if (!row.placement.empty())
        {
            ++placedRows;
            misplacedRows += row.placement != "ok" ? 1 : 0;
        }
    }

    if (placedRows > 0)
    {
        std::cout << "  blocks per SM (global)     : " << (misplacedRows == 0 ? "ok, " + std::to_string(options.blocksPerSm)
            + " per SM on distinct stripes" : std::to_string(misplacedRows) + " of " + std::to_string(placedRows)
            + " rows UNEVEN (see notes)") << "\n";
    }

    // KTT reads nvmlDeviceGetPowerUsage. The other NVML readings of the same configurations (second half of the trace) show
    // whether a difference comes from averaging (instant vs 1 s average) or from what the counter measures.
    const std::vector<std::pair<std::string, double Row::*>> readings = {
        {"energy counter     ", &Row::counterPowerW},
        {"instant power field", &Row::instantPowerW},
        {"1 s average field  ", &Row::averagePowerW}};

    for (const auto& [label, member] : readings)
    {
        double sumRatio = 0.0;
        double maxDifference = 0.0;
        size_t count = 0;

        for (const auto& row : rows)
        {
            if (row.hasPower && row.level != "idle" && !std::isnan(row.*member) && row.powerW > 0.0)
            {
                sumRatio += row.*member / row.powerW;
                maxDifference = std::max(maxDifference, std::abs(row.*member - row.powerW) / row.powerW);
                ++count;
            }
        }

        if (count == 0)
        {
            std::cout << "  KTT power vs " << label << ": not available\n";
            continue;
        }

        std::cout << "  KTT power vs " << label << ": mean ratio " << Fixed(sumRatio / static_cast<double>(count), 3)
            << ", max difference " << Fixed(maxDifference * 100.0, 1) << " % "
            << (maxDifference <= 0.05 ? "(ok)" : "(LARGE: inspect the trace file)") << "\n";
    }

    // On an RTX 3090 the energy counter reads far less than the power readings at idle and about 20 % less under load, so it
    // does not measure the same rails. Its share of the dynamic power (above the sleep baseline) shows how much of a probe's
    // energy it sees.
    double baselineCounterW = 0.0;
    size_t baselineCounters = 0;

    for (const auto& row : rows)
    {
        if (row.level == "baseline" && row.activeSms == 0 && !std::isnan(row.counterPowerW))
        {
            baselineCounterW += row.counterPowerW;
            ++baselineCounters;
        }
    }

    if (baselineCounters > 0)
    {
        baselineCounterW /= static_cast<double>(baselineCounters);
        std::vector<double> shares;

        for (const auto& row : rows)
        {
            if (IsProbeRow(row) && row.activeSms == 0 && row.duty == 100 && !std::isnan(row.counterPowerW)
                && !std::isnan(row.baselineW) && row.powerW - row.baselineW > 10.0)
            {
                shares.push_back((row.counterPowerW - baselineCounterW) / (row.powerW - baseline.At(row.temperatureC)));
            }
        }

        if (!shares.empty())
        {
            std::sort(shares.begin(), shares.end());
            std::cout << "  energy counter's share of dynamic power (duty-100 rows more than 10 W above the baseline): median "
                << Fixed(shares[shares.size() / 2], 2) << ", range " << Fixed(shares.front(), 2) << "-"
                << Fixed(shares.back(), 2) << "\n";
        }
    }

    const Row* spin = nullptr;

    for (const auto& row : rows)
    {
        if (row.level == "reference" && row.hasPower)
        {
            spin = &row;
        }
    }

    if (spin != nullptr)
    {
        std::cout << "  spin reference (heavy issue, not a baseline): " << Fixed(spin->powerW, 1) << " W at "
            << Fixed(spin->smClockMHz, 0) << " MHz\n";
    }
}

} // namespace

void PrintSummary(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const Footprints& footprints,
    const BaselineModel& baseline, const Options& options, const NvmlMonitor* monitor)
{
    PrintBaselineAndClock(rows, baseline, options, monitor);
    PrintPathEnergies(rows, footprints);
    PrintFitSummary(fits, footprints, options);
    PrintIncrementalEnergies(rows, footprints);
    PrintSanityChecks(rows, footprints, baseline, options, monitor);
}

void ReportProbeRun(std::vector<Row>& rows, const BaselineModel& baseline, const Footprints& footprints,
    const Hardware& hardware, const Options& options, const NvmlMonitor* monitor)
{
    ComputeEnergies(rows, baseline, ClockIsLocked(options), options.forceEnergy);
    const std::vector<ProbeFit> fits = FitProbes(rows, options.fitMinDuty);

    PrintRows(rows);
    PrintCounters(rows);
    PrintSummary(rows, fits, footprints, baseline, options, monitor);
    nlohmann::json table = PrintStudies(rows, fits, footprints, baseline, hardware, options);

    // For tools/build_model.py: the card and the software the run was made with.
    if (monitor != nullptr)
    {
        const DeviceProvenance provenance = monitor->GetProvenance();
        const auto textOrNull = [](const std::string& text) { return text.empty() ? nlohmann::json() : nlohmann::json(text); };
        table["provenance"] = {{"driver_version", textOrNull(provenance.driverVersion)},
            {"cuda_driver_version", textOrNull(provenance.cudaDriverVersion)}, {"uuid", textOrNull(provenance.uuid)},
            {"serial", textOrNull(provenance.serial)}, {"power_limit_w", ToJson(provenance.powerLimitW)}};
    }

    WriteCsv(rows, options.outputPrefix + ".csv");
    WriteFitsCsv(fits, options.outputPrefix + "_fits.csv");
    std::ofstream(options.outputPrefix + "_energy_table.json") << table.dump(2) << "\n";
}
