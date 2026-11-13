#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdio>
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

#include "Studies.h"

namespace
{

// A probe as the studies name it: its label, and the operation, level, footprint and width of its rows.
struct Probe
{
    std::string label;
    std::string operation;
    std::string level;
    uint64_t footprint;
    uint64_t width;
};

// The idle readings (no kernel) of a run, in their order.
std::vector<const Row*> IdleRows(const std::vector<Row>& rows)
{
    std::vector<const Row*> idle;

    for (const auto& row : rows)
    {
        if (row.level == "idle" && row.hasPower)
        {
            idle.push_back(&row);
        }
    }

    return idle;
}

// The arithmetic operations a run measured (random operands).
std::vector<uint64_t> MeasuredComputeOps(const std::vector<Row>& rows)
{
    const Variant anyWidth{0, PayloadName(PayloadRandom)};
    std::vector<uint64_t> computeOps;

    for (uint64_t op = ComputeFfma; op <= ComputeMufuSin; ++op)
    {
        const bool indexing = op == ComputeStaticIndex || op == ComputeDynamicIndex; // a study of their own

        if (!indexing && !ProbeRows(rows, ComputeName(op), "compute", 0, anyWidth).empty())
        {
            computeOps.push_back(op);
        }
    }

    return computeOps;
}

// The idle power (no CUDA context, with a context, and the sleep baseline) and the split of the static power.
void PrintConstantAndStaticPower(const std::vector<Row>& rows, const BaselineModel& baseline, nlohmann::json& table)
{
    // Constant power: the idle GPU without a CUDA context, in its lowest power state (Berkeley's P_const, Wattchmen's
    // constant power). Static power: the sleep baseline above it (an awake GPU at the probe clocks, Wattchmen's static
    // power), split into what a CUDA context alone costs (the driver keeps the memory clock up) and what the SMs with
    // sleeping warps add. Energies per access are above all of it.
    const std::vector<const Row*> idle = IdleRows(rows);

    if (!idle.empty())
    {
        std::cout << "\nConstant and static power:\n";
        table["idle"] = nlohmann::json::array();

        for (const Row* row : idle)
        {
            std::cout << "  " << std::left << std::setw(11) << row->probe << std::right << " (no kernel): "
                << Fixed(row->powerW, 1) << " W at " << Fixed(row->temperatureC, 0) << " C, SM " << Fixed(row->smClockMHz, 0)
                << " MHz, memory " << Fixed(row->memoryClockMHz, 0) << " MHz\n";
            table["idle"].push_back({{"label", row->probe}, {"power_w", ToJson(row->powerW)},
                {"temperature_c", ToJson(row->temperatureC)}, {"sm_clock_mhz", ToJson(row->smClockMHz)},
                {"memory_clock_mhz", ToJson(row->memoryClockMHz)}});
        }

        // The reading without a context, and the context reading closest to the baseline's temperature.
        const Row* withoutContext = nullptr;
        const Row* withContext = nullptr;

        for (const Row* row : idle)
        {
            if (row->probe == "idle-noctx")
            {
                withoutContext = row;
            }
            else if (withContext == nullptr || std::abs(row->temperatureC - baseline.meanTemperatureC)
                < std::abs(withContext->temperatureC - baseline.meanTemperatureC))
            {
                withContext = row;
            }
        }

        std::cout << "  sleep baseline: " << Fixed(baseline.meanW, 1) << " W at " << Fixed(baseline.meanTemperatureC, 0)
            << " C\n";

        if (withoutContext != nullptr)
        {
            const double staticW = baseline.meanW - withoutContext->powerW;
            std::cout << "  constant (no CUDA context): " << Fixed(withoutContext->powerW, 1) << " W; static = sleep baseline - "
                << "constant: " << Fixed(staticW, 1) << " W\n";
            table["constant_w"] = ToJson(withoutContext->powerW);
            table["static_w"] = ToJson(staticW);
        }

        double sleepMemoryClockMHz = NaN;

        for (const auto& row : rows)
        {
            if (row.level == "baseline" && row.activeSms == 0 && !std::isnan(row.memoryClockMHz))
            {
                sleepMemoryClockMHz = row.memoryClockMHz;
            }
        }

        // With a context the driver lowers the memory clock after a while; the split is only meaningful while the memory
        // still runs at the baseline's clock.
        if (withContext != nullptr && std::abs(withContext->memoryClockMHz - sleepMemoryClockMHz) > 1.0)
        {
            std::cout << "  (no split of the static power: with a context but no kernel the memory ran at "
                << Fixed(withContext->memoryClockMHz, 0) << " MHz on average, the baseline at " << Fixed(sleepMemoryClockMHz, 0)
                << " MHz)\n";
        }
        else if (withContext != nullptr)
        {
            const double awakeW = baseline.meanW - withContext->powerW;
            std::cout << "  of the static power, SMs with sleeping warps (sleep baseline - " << withContext->probe << "): "
                << Fixed(awakeW, 1) << " W";

            if (withoutContext != nullptr)
            {
                std::cout << "; a CUDA context without a kernel: " << Fixed(withContext->powerW - withoutContext->powerW, 1)
                    << " W (memory clock " << Fixed(withoutContext->memoryClockMHz, 0) << " -> "
                    << Fixed(withContext->memoryClockMHz, 0) << " MHz)";
                table["context_w"] = ToJson(withContext->powerW - withoutContext->powerW);
            }

            std::cout << "\n";
            table["awake_sms_w"] = ToJson(awakeW);
        }
    }
}

// Every fit across duties, for the table.
void AddFits(const std::vector<ProbeFit>& fits, nlohmann::json& table)
{
    std::vector<nlohmann::json> fitEntries;

    for (const auto& fit : fits)
    {
        fitEntries.push_back({{"operation", fit.operation}, {"level", fit.level}, {"footprint_bytes", fit.footprint},
            {"width_bytes", fit.width}, {"payload", fit.payload}, {"duties", fit.duties},
            {"pj_per_access", ToJson(fit.pJPerAccess)}, {"pj_per_byte", ToJson(fit.pJPerByte)},
            {"pj_per_flop", ToJson(fit.pJPerAccess / fit.flopsPerOp)}, {"fixed_w", ToJson(fit.fixedW)},
            {"r2", ToJson(fit.r2)}});
    }

    table["fits"] = fitEntries;
}

void PrintArithmetic(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, nlohmann::json& table)
{
    const std::string random = PayloadName(PayloadRandom);

    // Arithmetic: energy per operation of one thread, minus the loop overhead per counted operation (the empty loop counts
    // the same number of operations per iteration).
    const Variant anyWidth{0, random};
    const Estimate loop = EstimateEnergy(rows, fits, ComputeName(ComputeLoop), "compute", 0, anyWidth);
    const std::vector<uint64_t> computeOps = MeasuredComputeOps(rows);

    if (!computeOps.empty())
    {
        std::cout << "\nArithmetic (random operands; per operation of one thread, a warp instruction is 32 of them):\n"
            << "  " << std::left << std::setw(7) << "op" << std::right << std::setw(11) << "pJ/op" << std::setw(12)
            << "net of loop" << std::setw(14) << "pJ/warp instr" << std::setw(9) << "pJ/FLOP" << "  source\n";
        table["compute"] = nlohmann::json::array();

        for (const uint64_t op : computeOps)
        {
            const Estimate estimate = EstimateEnergy(rows, fits, ComputeName(op), "compute", 0, anyWidth);
            const double net = estimate.pJPerAccess - (std::isnan(loop.pJPerAccess) ? 0.0 : loop.pJPerAccess);
            // The rows carry the FLOPs of their GPU's shape (HMMA: m8n8k4 / m16n8k8 on sm_7x, m16n8k16 from sm_80).
            const auto withFlops = std::find_if(rows.cbegin(), rows.cend(), [&](const Row& row)
                { return row.operation == ComputeName(op) && row.level == "compute" && !std::isnan(row.flopsPerOp); });
            const double perFlop = net / (withFlops != rows.cend() ? withFlops->flopsPerOp : FlopsPerOperation(op));
            std::cout << "  " << std::left << std::setw(7) << ComputeName(op) << std::right << std::setw(11)
                << Fixed(estimate.pJPerAccess, 2) << std::setw(12) << Fixed(net, 2) << std::setw(14) << Fixed(net * 32.0, 1)
                << std::setw(9) << Fixed(perFlop, 3) << "  " << estimate.source << "\n";
            table["compute"].push_back({{"operation", ComputeName(op)}, {"pj_per_op", ToJson(estimate.pJPerAccess)},
                {"net_pj_per_op", ToJson(net)}, {"pj_per_warp_instruction", ToJson(net * 32.0)},
                {"pj_per_flop", ToJson(perFlop)}, {"source", estimate.source}});
        }

        std::cout << "  loop overhead: " << Fixed(loop.pJPerAccess, 2) << " pJ per counted operation (" << loop.source
            << "); 'pJ/FLOP' counts an FMA as 2 FLOPs (IMAD: 2 integer operations)\n";
        table["compute_loop_pj_per_op"] = ToJson(loop.pJPerAccess);

        // The FFMA probe reads one register twice and takes c from the constant bank (x * x + c); ffma3 reads three
        // different registers, as the FMAs of real kernels.
        const Estimate ffma = EstimateEnergy(rows, fits, ComputeName(ComputeFfma), "compute", 0, anyWidth);
        const Estimate ffma3 = EstimateEnergy(rows, fits, ComputeName(ComputeFfma3), "compute", 0, anyWidth);

        if (!std::isnan(ffma.pJPerAccess) && !std::isnan(ffma3.pJPerAccess))
        {
            const double extra = ffma3.pJPerAccess - ffma.pJPerAccess;
            std::cout << "  three register operands (ffma3) vs x * x + c (ffma): " << (extra >= 0 ? "+" : "") << Fixed(extra, 2)
                << " pJ per FFMA (" << SignedPercent(extra / ffma.pJPerAccess) << "), " << Fixed(extra * 32.0, 0)
                << " pJ per warp instruction\n";
            table["ffma3_extra_pj_per_op"] = ToJson(extra);
        }
    }
}

void PrintIndexing(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, nlohmann::json& table)
{
    const std::string random = PayloadName(PayloadRandom);
    const Variant anyWidth{0, random};

    // Register-array indexing: the same FFMA updates of 32 accumulators with static indices and with a runtime index (compares
    // and guarded moves around every update). The difference is what the dynamic indexing costs per update.
    const Estimate staticIndex = EstimateEnergy(rows, fits, ComputeName(ComputeStaticIndex), "compute", 0, anyWidth);
    const Estimate dynamicIndex = EstimateEnergy(rows, fits, ComputeName(ComputeDynamicIndex), "compute", 0, anyWidth);

    if (!std::isnan(staticIndex.pJPerAccess) && !std::isnan(dynamicIndex.pJPerAccess))
    {
        const double extra = dynamicIndex.pJPerAccess - staticIndex.pJPerAccess;
        std::cout << "\nRegister-array indexing (32 FFMA accumulators, per update of one thread):\n"
            << "  static indices " << Fixed(staticIndex.pJPerAccess, 2) << " pJ (" << staticIndex.source << "), runtime index "
            << Fixed(dynamicIndex.pJPerAccess, 2) << " pJ (" << dynamicIndex.source << "): the compares and guarded moves add "
            << Fixed(extra, 2) << " pJ per update, " << Fixed(extra * 32.0, 0) << " pJ per warp update\n";
        table["indexing"] = {{"static_pj_per_update", ToJson(staticIndex.pJPerAccess)},
            {"indexed_pj_per_update", ToJson(dynamicIndex.pJPerAccess)}, {"extra_pj_per_update", ToJson(extra)}};
    }
}

void PrintAtomics(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const Footprints& footprints,
    nlohmann::json& table)
{
    const std::string random = PayloadName(PayloadRandom);

    // Atomics (32-bit adds, every thread its own word): red runs the same code as the 4-byte store, atom the same as the
    // 4-byte load plus the operand recurrence of the store, so the differences are what the atomic operation in L2 adds.
    const Variant atomicWidth{AtomicWidth, random};

    if (!ProbeRows(rows, "red", "L2", footprints.l2, atomicWidth).empty()
        || !ProbeRows(rows, "red.shared", "shared", 0, atomicWidth).empty())
    {
        struct AtomicProbe
        {
            std::string label;
            std::string operation;
            std::string level;
            uint64_t footprint;
            std::string reference;
        };

        const std::vector<AtomicProbe> atomicProbes = {
            {"L2 red", "red", "L2", footprints.l2, "st"},
            {"L2 atom", "atom", "L2", footprints.l2, "ld"},
            {"DRAM red", "red", "DRAM", footprints.dram, "st"},
            {"DRAM atom", "atom", "DRAM", footprints.dram, "ld"},
            {"shared red", "red.shared", "shared", 0, ""}};
        const Estimate loopW4 = EstimateEnergy(rows, fits, "no-load", "none", 0, atomicWidth);

        std::cout << "\nAtomics (32-bit add per thread, random operands; pJ per access of one thread, a warp instruction is 32):\n"
            << "  " << std::left << std::setw(11) << "probe" << std::right << std::setw(10) << "pJ/access" << std::setw(13)
            << "net of loop" << std::setw(11) << "reference" << std::setw(10) << "ref pJ" << std::setw(12) << "atomic adds"
            << std::setw(16) << "adds/warp instr" << "  source\n";
        table["atomics"] = nlohmann::json::array();

        for (const auto& probe : atomicProbes)
        {
            const Estimate estimate = EstimateEnergy(rows, fits, probe.operation, probe.level, probe.footprint, atomicWidth);

            if (std::isnan(estimate.pJPerAccess))
            {
                continue;
            }

            const Estimate reference = probe.reference.empty() ? Estimate{} : EstimateEnergy(rows, fits, probe.reference,
                probe.level, probe.footprint, atomicWidth);
            const double extra = estimate.pJPerAccess - reference.pJPerAccess;
            std::cout << "  " << std::left << std::setw(11) << probe.label << std::right << std::setw(10)
                << Fixed(estimate.pJPerAccess, 1) << std::setw(13) << Fixed(estimate.pJPerAccess - loopW4.pJPerAccess, 1)
                << std::setw(11) << (probe.reference.empty() ? "-" : probe.reference) << std::setw(10)
                << Fixed(reference.pJPerAccess, 1) << std::setw(12) << Fixed(extra, 1) << std::setw(16) << Fixed(extra * 32.0, 0)
                << "  " << estimate.source << "\n";
            table["atomics"].push_back({{"probe", probe.label}, {"operation", probe.operation}, {"level", probe.level},
                {"pj_per_access", ToJson(estimate.pJPerAccess)},
                {"net_pj_per_access", ToJson(estimate.pJPerAccess - loopW4.pJPerAccess)},
                {"reference", probe.reference}, {"reference_pj_per_access", ToJson(reference.pJPerAccess)},
                {"extra_pj_per_access", ToJson(extra)}, {"source", estimate.source}});
        }

        std::cout << "  'net of loop' subtracts the 4-byte no-load loop (" << Fixed(loopW4.pJPerAccess, 1) << " pJ), which lacks the "
            << "operand recurrence; 'atomic adds' is the energy above the store (red) or load (atom) of the same level\n";
    }
}

void PrintConstantMemory(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const Footprints& footprints,
    nlohmann::json& table)
{
    const std::string random = PayloadName(PayloadRandom);
    const Variant anyWidth{0, random};

    // Constant memory: one request to the constant cache per warp access. The L1.5 footprint misses the constant L1 on
    // every access with the same SASS, so the difference is the energy of a constant L1 miss served by the L1.5.
    const Estimate constantL1 = EstimateEnergy(rows, fits, "ld.const", "const-L1", footprints.constantL1, anyWidth);
    const Estimate constantL15 = EstimateEnergy(rows, fits, "ld.const", "const-L1.5", footprints.constantL15, anyWidth);

    if (!std::isnan(constantL1.pJPerAccess) || !std::isnan(constantL15.pJPerAccess))
    {
        std::cout << "\nConstant memory (all threads of a warp read one word; one constant-cache request per warp access):\n";
        table["constant"] = nlohmann::json::array();

        for (const auto& [level, footprint, estimate] : std::vector<std::tuple<std::string, uint64_t, Estimate>>{
            {"const-L1", footprints.constantL1, constantL1}, {"const-L1.5", footprints.constantL15, constantL15}})
        {
            std::cout << "  " << std::left << std::setw(11) << level << std::right << std::setw(9) << FormatBytes(footprint)
                << std::setw(9) << Fixed(estimate.pJPerAccess, 2) << " pJ per thread access, " << std::setw(7)
                << Fixed(estimate.pJPerAccess * 32.0, 1) << " pJ per warp instruction  (" << estimate.source << ")\n";
            table["constant"].push_back({{"level", level}, {"footprint_bytes", footprint},
                {"pj_per_access", ToJson(estimate.pJPerAccess)}, {"pj_per_warp_instruction", ToJson(estimate.pJPerAccess * 32.0)},
                {"source", estimate.source}});
        }

        const double miss = (constantL15.pJPerAccess - constantL1.pJPerAccess) * 32.0;
        std::cout << "  constant L1 miss served by the L1.5 (difference, same SASS): " << Fixed(miss, 1) << " pJ per warp request\n";
        table["constant_l1_miss_pj_per_request"] = ToJson(miss);
    }
}

void PrintDramLocality(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const Footprints& footprints,
    nlohmann::json& table)
{
    const std::string random = PayloadName(PayloadRandom);

    // Granularity and DRAM row locality: the same SASS streams through the footprint or accesses runs of consecutive sectors
    // placed apart. Runs of 128 bytes and more are whole lines (4 sectors per L2 request, as the stream) and lose only row
    // locality: if every run costs a row activation (and what comes with it), pJ/B = per byte + per run / run bytes, so a
    // line over 1 / run bytes gives both. Runs of 32 and 64 bytes also send 1 or 2 sectors per L2 request. The L2 control
    // runs the stream and 32-byte runs at the L2 footprint: what differs there is not DRAM.
    if (!ProbeRows(rows, "ld-stream", "DRAM", footprints.dram).empty())
    {
        std::cout << "\nGranularity and DRAM row locality (16-byte accesses, random data; runs of consecutive sectors placed "
            << "apart):\n"
            << "  " << std::left << std::setw(4) << "op" << std::setw(9) << "run" << std::right << std::setw(9) << "pJ/B"
            << std::setw(15) << "GB/s at 100 %" << std::setw(11) << "vs stream" << "  source\n";
        table["dram_locality"] = nlohmann::json::array();

        for (const std::string operation : {"ld", "st"})
        {
            const Estimate stream = EstimateEnergy(rows, fits, operation + "-stream", "DRAM", footprints.dram,
                {DefaultWidth, random});
            std::vector<std::pair<double, double>> points;

            for (const uint64_t run : LocalityRuns)
            {
                const std::string name = operation + (run == 0 ? "-stream" : "-run" + std::to_string(run));
                const Estimate estimate = EstimateEnergy(rows, fits, name, "DRAM", footprints.dram, {DefaultWidth, random});
                const Row* full = FullIntensityRow(rows, name, "DRAM", footprints.dram);
                const double perByte = estimate.pJPerAccess / DefaultWidth;

                if (std::isnan(perByte))
                {
                    continue;
                }

                std::cout << "  " << std::left << std::setw(4) << operation << std::setw(9)
                    << (run == 0 ? std::string("stream") : FormatBytes(run)) << std::right << std::setw(9) << Fixed(perByte, 1)
                    << std::setw(15) << Fixed(full != nullptr ? full->bandwidthGBs : NaN, 0) << std::setw(10)
                    << Fixed(estimate.pJPerAccess / stream.pJPerAccess, 2) << "x  " << estimate.source << "\n";
                table["dram_locality"].push_back({{"operation", operation}, {"run_bytes", run},
                    {"pj_per_byte", ToJson(perByte)}, {"bandwidth_gbs", ToJson(full != nullptr ? full->bandwidthGBs : NaN)},
                    {"source", estimate.source}});

                if (run >= 128)
                {
                    points.emplace_back(1.0 / static_cast<double>(run), perByte);
                }
            }

            const Line line = FitLine(points);
            std::cout << "  " << operation << ", runs of whole lines: pJ/B = " << Fixed(line.intercept, 1) << " + "
                << Fixed(line.slope, 0) << " pJ per run / run bytes (R^2 " << Fixed(line.r2, 3) << ")\n";
            table["dram_locality_fit"][operation] = {{"pj_per_byte", ToJson(line.intercept)},
                {"pj_per_run", ToJson(line.slope)}, {"r2", ToJson(line.r2)}};
        }

        for (const std::string operation : {"ld", "st"})
        {
            const Estimate l2Stream = EstimateEnergy(rows, fits, operation + "-stream", "L2", footprints.l2,
                {DefaultWidth, random});
            const Estimate l2Sectors = EstimateEnergy(rows, fits, operation + "-run32", "L2", footprints.l2,
                {DefaultWidth, random});
            std::cout << "  L2 control (" << FormatBytes(footprints.l2) << "), " << operation << ": stream "
                << Fixed(l2Stream.pJPerAccess / DefaultWidth, 1) << " pJ/B, 32 B runs "
                << Fixed(l2Sectors.pJPerAccess / DefaultWidth, 1) << " pJ/B\n";
            table["dram_locality_l2_control"][operation] = {
                {"stream_pj_per_byte", ToJson(l2Stream.pJPerAccess / DefaultWidth)},
                {"run32_pj_per_byte", ToJson(l2Sectors.pJPerAccess / DefaultWidth)}};
        }
    }
}

void PrintSubSector(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const Footprints& footprints,
    nlohmann::json& table)
{
    const std::string random = PayloadName(PayloadRandom);

    // Sub-sector accesses: one word of 32 sectors per warp instruction, the sectors of a warp adjacent (4 per L2 request) or
    // scattered (1 per request), compared with the coalesced 16-byte access of the same level (probeGlobal).
    if (!ProbeRows(rows, "st-sub-scattered", "DRAM", footprints.dram, {4, random}).empty())
    {
        std::cout << "\nSub-sector accesses (every thread walks a sector of its own, one word per access; random data):\n"
            << "  " << std::left << std::setw(4) << "op" << std::setw(6) << "level" << std::setw(11) << "sectors" << std::right
            << std::setw(4) << "W" << std::setw(9) << "pJ/B" << std::setw(15) << "GB/s at 100 %" << std::setw(18)
            << "vs 16 B coalesced" << "  source\n";
        table["sub_sector"] = nlohmann::json::array();

        for (const auto& [level, footprint] : std::vector<std::pair<std::string, uint64_t>>{{"DRAM", footprints.dram},
            {"L2", footprints.l2}})
        {
            for (const std::string operation : {"ld", "st"})
            {
                const Estimate coalesced = EstimateEnergy(rows, fits, operation, level, footprint, {DefaultWidth, random});

                for (const std::string spread : {"adjacent", "scattered"})
                {
                    for (const uint64_t width : {4, 8})
                    {
                        const std::string name = operation + "-sub-" + spread;
                        const Estimate estimate = EstimateEnergy(rows, fits, name, level, footprint, {width, random});
                        const double perByte = estimate.pJPerAccess / static_cast<double>(width);

                        if (std::isnan(perByte))
                        {
                            continue;
                        }

                        const Row* full = nullptr;

                        for (const Row* row : ProbeRows(rows, name, level, footprint, {width, random}))
                        {
                            full = full == nullptr || row->duty > full->duty ? row : full;
                        }

                        const double ratio = perByte / (coalesced.pJPerAccess / DefaultWidth);
                        std::cout << "  " << std::left << std::setw(4) << operation << std::setw(6) << level << std::setw(11)
                            << spread << std::right << std::setw(4) << width << std::setw(9) << Fixed(perByte, 1)
                            << std::setw(15) << Fixed(full != nullptr ? full->bandwidthGBs : NaN, 0) << std::setw(17)
                            << Fixed(ratio, 2) << "x  " << estimate.source << "\n";
                        table["sub_sector"].push_back({{"operation", operation}, {"level", level}, {"spread", spread},
                            {"width_bytes", width}, {"pj_per_byte", ToJson(perByte)},
                            {"bandwidth_gbs", ToJson(full != nullptr ? full->bandwidthGBs : NaN)},
                            {"vs_coalesced", ToJson(ratio)}, {"source", estimate.source}});
                    }
                }
            }
        }
    }
}

void PrintWrites(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const Footprints& footprints,
    nlohmann::json& table)
{
    const std::string random = PayloadName(PayloadRandom);
    // The (operation, level, footprint, width) of every write probe the run has.
    std::set<std::tuple<std::string, std::string, uint64_t, uint64_t>> probes;

    for (const auto& row : rows)
    {
        if (row.probe == "write" && row.payload == random)
        {
            probes.insert({row.operation, row.level, row.footprint, row.width});
        }
    }

    if (probes.empty())
    {
        return;
    }

    // Loads and stores together, against a load plus a store of the plain probes at the same level and width (probeGlobal;
    // their energies are per access of one thread, as here). Local memory has no plain counterpart.
    std::cout << "\nLoads and stores together (one access = one load and one store; random data):\n"
        << "  " << std::left << std::setw(8) << "op" << std::setw(6) << "level" << std::right << std::setw(4) << "W"
        << std::setw(11) << "pJ/access" << std::setw(8) << "pJ/B" << std::setw(15) << "GB/s at 100 %" << std::setw(12)
        << "ld + st" << std::setw(9) << "ratio" << "  source\n";
    table["writes"] = nlohmann::json::array();
    const std::vector<std::string> order = {"copy", "update", "local"};

    for (const std::string& operation : order)
    {
        for (const auto& [name, level, footprint, width] : probes)
        {
            if (name != operation)
            {
                continue;
            }

            const Estimate estimate = EstimateEnergy(rows, fits, name, level, footprint, {width, random});
            double plain = NaN;

            if (operation != "local")
            {
                const uint64_t plainFootprint = level == "DRAM" ? footprints.dram : footprints.l2;
                plain = EstimateEnergy(rows, fits, "ld", level, plainFootprint, {width, random}).pJPerAccess
                    + EstimateEnergy(rows, fits, "st", level, plainFootprint, {width, random}).pJPerAccess;
            }

            const Row* full = nullptr;

            for (const Row* row : ProbeRows(rows, name, level, footprint, {width, random}))
            {
                full = full == nullptr || row->duty > full->duty ? row : full;
            }

            // Bytes moved per access: one element loaded and one stored.
            const double perByte = estimate.pJPerAccess / static_cast<double>(2 * width);
            const double bandwidth = full != nullptr ? 2.0 * full->bandwidthGBs : NaN;
            std::cout << "  " << std::left << std::setw(8) << name << std::setw(6) << level << std::right << std::setw(4)
                << width << std::setw(11) << Fixed(estimate.pJPerAccess, 1) << std::setw(8) << Fixed(perByte, 1)
                << std::setw(15) << Fixed(bandwidth, 0) << std::setw(12) << Fixed(plain, 1) << std::setw(8)
                << Fixed(estimate.pJPerAccess / plain, 2) << "x  " << estimate.source << "\n";
            table["writes"].push_back({{"operation", name}, {"level", level}, {"footprint_bytes", footprint},
                {"width_bytes", width}, {"pj_per_access", ToJson(estimate.pJPerAccess)}, {"pj_per_byte", ToJson(perByte)},
                {"bandwidth_gbs", ToJson(bandwidth)}, {"ld_plus_st_pj", ToJson(plain)},
                {"vs_ld_plus_st", ToJson(estimate.pJPerAccess / plain)}, {"source", estimate.source}});
        }
    }

    std::cout << "  (GB/s: loaded and stored bytes together. ld + st: the plain load and store probes of the level, "
        << "the same width)\n";
}

void PrintLatency(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const Footprints& footprints,
    nlohmann::json& table)
{
    const std::string random = PayloadName(PayloadRandom);
    std::vector<const Row*> latency;

    for (const auto& row : rows)
    {
        if (row.probe == "latency" && row.payload == random && row.duty == 100 && ClockMatched(row))
        {
            latency.push_back(&row);
        }
    }

    if (latency.empty())
    {
        return;
    }

    // Dependent loads by memory-level parallelism (chains per warp x working warps per SM), against the plain 4-byte load
    // of the level at full bandwidth (probeGlobal's fit). Energies at duty 100: (power - baseline) / load rate.
    std::cout << "\nDependent loads (4 bytes per thread, random lines; duty 100, random data):\n"
        << "  " << std::left << std::setw(6) << "level" << std::right << std::setw(8) << "chains" << std::setw(10)
        << "warps/SM" << std::setw(12) << "in flight" << std::setw(10) << "GB/s" << std::setw(9) << "P-base" << std::setw(9)
        << "pJ/B" << std::setw(17) << "vs ld at full BW" << "\n"
        << "  ('vs ld': the plain 4-byte load of the level in this run; '-' when the run has none, as with --latency alone)\n";
    table["latency"] = nlohmann::json::array();
    std::sort(latency.begin(), latency.end(), [](const Row* a, const Row* b)
    {
        return a->level != b->level ? a->level > b->level : a->bandwidthGBs < b->bandwidthGBs;
    });

    for (const Row* row : latency)
    {
        uint64_t chains = 0;
        uint64_t warps = 0;

        if (std::sscanf(row->operation.c_str(), "chase-c%" SCNu64 "-w%" SCNu64, &chains, &warps) != 2)
        {
            continue;
        }

        const uint64_t footprint = row->level == "DRAM" ? footprints.dram : footprints.l2;
        const double plain = EstimateEnergy(rows, fits, "ld", row->level, footprint, {4, random}).pJPerAccess / 4.0;
        const double perByte = row->pJPerAccess / 4.0;
        std::cout << "  " << std::left << std::setw(6) << row->level << std::right << std::setw(8) << chains << std::setw(10)
            << warps << std::setw(12) << chains * warps << std::setw(10) << Fixed(row->bandwidthGBs, 0) << std::setw(9)
            << Fixed(row->powerW - row->baselineW, 1) << std::setw(9) << Fixed(perByte, 1) << std::setw(16)
            << Fixed(perByte / plain, 2) << (std::isfinite(perByte / plain) ? "x" : " ") << "\n";
        table["latency"].push_back({{"level", row->level}, {"chains", chains}, {"warps_per_sm", warps},
            {"bandwidth_gbs", ToJson(row->bandwidthGBs)}, {"dynamic_w", ToJson(row->powerW - row->baselineW)},
            {"pj_per_byte", ToJson(perByte)}, {"vs_full_bandwidth", ToJson(perByte / plain)}});
    }
}

void PrintBankConflicts(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, nlohmann::json& table)
{
    const std::string random = PayloadName(PayloadRandom);
    auto variant = [&](const uint64_t width) { return Variant{width, random}; };

    // Bank conflicts: the same shared access of 4, 8 or 16 bytes per thread, with the lanes of every wavefront on fewer
    // banks, so that the instruction takes `ways` times the wavefronts of the conflict-free access (1, 2 or 4). The energy
    // above the conflict-free access of the same width, divided by the extra wavefronts, is what a replayed wavefront
    // costs; a replay moves only the elements of the lanes that waited. For comparison, an extra full wavefront (128
    // bytes): the conflict-free 8- and 16-byte accesses above the 4-byte one (a wider load feeds more XORs, so the loops
    // differ slightly). Runs from before the probe had 8- and 16-byte accesses use the 16-byte access of probeShared.
    if (ProbeRows(rows, "ld.shared-1way", "shared", 0, variant(4)).empty()
        && ProbeRows(rows, "st.shared-1way", "shared", 0, variant(4)).empty())
    {
        return;
    }

    auto fullRateRow = [&](const std::string& operation, const uint64_t width)
    {
        const Row* full = nullptr;

        for (const Row* row : ProbeRows(rows, operation, "shared", 0, variant(width)))
        {
            full = full == nullptr || row->duty > full->duty ? row : full;
        }

        return full;
    };

    std::cout << "\nShared-memory bank conflicts (random data; ways = times the wavefronts of the conflict-free access; a warp"
        << " instruction is 32 thread accesses):\n"
        << "  " << std::left << std::setw(11) << "op" << std::right << std::setw(3) << "W" << std::setw(6) << "ways"
        << std::setw(11) << "pJ/access" << std::setw(15) << "nJ/warp instr" << std::setw(12) << "wavefronts" << std::setw(20)
        << "nJ/extra wavefront" << std::setw(15) << "GB/s at 100 %" << "  source\n";
    table["bank_conflicts"] = nlohmann::json::array();

    for (const std::string operation : {"ld.shared", "st.shared"})
    {
        const std::string wavefrontCounter = operation == "ld.shared"
            ? "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum" : "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum";

        for (const uint64_t width : BankWidths)
        {
            const Estimate conflictFree = EstimateEnergy(rows, fits, operation + "-1way", "shared", 0, variant(width));
            const double idealWavefronts = static_cast<double>(width) / 4.0;

            for (const uint64_t ways : BankWaysOfWidth(width))
            {
                const std::string name = operation + "-" + std::to_string(ways) + "way";
                const Estimate estimate = EstimateEnergy(rows, fits, name, "shared", 0, variant(width));

                if (std::isnan(estimate.pJPerAccess))
                {
                    continue;
                }

                // Wavefronts per warp instruction, with --profile only.
                const Row* full = fullRateRow(name, width);
                double wavefronts = NaN;

                if (full != nullptr && full->counters.count(wavefrontCounter) > 0 && full->accesses > 0)
                {
                    wavefronts = full->counters.at(wavefrontCounter) / (static_cast<double>(full->accesses) / 32.0);
                }

                const double perInstructionNJ = estimate.pJPerAccess * 32.0 / 1000.0;
                const double extraNJ = ways > 1 ? (estimate.pJPerAccess - conflictFree.pJPerAccess) * 32.0 / 1000.0
                    / (static_cast<double>(ways - 1) * idealWavefronts) : NaN;
                std::cout << "  " << std::left << std::setw(11) << operation << std::right << std::setw(3) << width
                    << std::setw(6) << ways << std::setw(11) << Fixed(estimate.pJPerAccess, 1) << std::setw(15)
                    << Fixed(perInstructionNJ, 2) << std::setw(12) << Fixed(wavefronts, 2) << std::setw(20)
                    << Fixed(extraNJ, 3) << std::setw(15) << Fixed(full != nullptr ? full->bandwidthGBs : NaN, 0) << "  "
                    << estimate.source << "\n";
                table["bank_conflicts"].push_back({{"operation", operation}, {"width_bytes", width}, {"ways", ways},
                    {"pj_per_access", ToJson(estimate.pJPerAccess)}, {"nj_per_warp_instruction", ToJson(perInstructionNJ)},
                    {"wavefronts_per_instruction", ToJson(wavefronts)}, {"nj_per_extra_wavefront", ToJson(extraNJ)},
                    {"bandwidth_gbs", ToJson(full != nullptr ? full->bandwidthGBs : NaN)}, {"source", estimate.source}});
            }
        }

        // An extra full wavefront: conflict-free accesses of 8 and 16 bytes (2 and 4 wavefronts) above the 4-byte one.
        const Estimate narrow = EstimateEnergy(rows, fits, operation + "-1way", "shared", 0, variant(4));
        struct Wide
        {
            std::string label;
            uint64_t width;
            Estimate estimate;
        };

        std::vector<Wide> wide;

        for (const uint64_t width : {8, 16})
        {
            const Estimate estimate = EstimateEnergy(rows, fits, operation + "-1way", "shared", 0, variant(width));

            if (!std::isnan(estimate.pJPerAccess))
            {
                wide.push_back({std::to_string(width) + "-byte conflict-free access", width, estimate});
            }
        }

        if (wide.empty())
        {
            wide.push_back({"16-byte access of probeShared", DefaultWidth, EstimateEnergy(rows, fits, operation, "shared", 0,
                variant(DefaultWidth))});
        }

        for (const auto& [label, width, estimate] : wide)
        {
            const double extraWavefronts = static_cast<double>(width) / 4.0 - 1.0;

            if (std::isnan(estimate.pJPerAccess) || std::isnan(narrow.pJPerAccess))
            {
                continue;
            }

            const double extraNJ = (estimate.pJPerAccess - narrow.pJPerAccess) * 32.0 / 1000.0 / extraWavefronts;
            std::cout << "  " << operation << ", " << label << " (" << width / 4 << " wavefronts of 128 B): "
                << Fixed(estimate.pJPerAccess * 32.0 / 1000.0, 2) << " nJ per warp instruction, " << Fixed(extraNJ, 3)
                << " nJ per extra full wavefront (" << estimate.source << ")\n";
            table["bank_conflicts"].push_back({{"operation", operation}, {"reference", label}, {"width_bytes", width},
                {"pj_per_access", ToJson(estimate.pJPerAccess)}, {"nj_per_extra_full_wavefront", ToJson(extraNJ)},
                {"source", estimate.source}});
        }
    }
}

void PrintTexture(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const Footprints& footprints,
    nlohmann::json& table)
{
    const std::string random = PayloadName(PayloadRandom);

    // Texture fetches: the footprints and stripes of the plain loads, through the texture path of L1TEX instead of the
    // load path. The SASS differs (TLD and a texel index instead of LDG and an address), so 'tex - ld' includes that.
    if (!ProbeRows(rows, "tex", "L1", footprints.l1).empty())
    {
        std::cout << "\nTexture fetches (16-byte texels, random data; compared with the plain 16-byte load of the level):\n"
            << "  " << std::left << std::setw(6) << "level" << std::right << std::setw(10) << "tex pJ/B" << std::setw(10)
            << "ld pJ/B" << std::setw(11) << "tex - ld" << std::setw(18) << "per warp instr" << "  source\n";
        table["texture"] = nlohmann::json::array();

        for (const auto& [level, footprint] : std::vector<std::pair<std::string, uint64_t>>{{"L1", footprints.l1},
            {"L2", footprints.l2}, {"DRAM", footprints.dram}})
        {
            const Estimate fetch = EstimateEnergy(rows, fits, "tex", level, footprint, {DefaultWidth, random});
            const Estimate load = EstimateEnergy(rows, fits, "ld", level, footprint, {DefaultWidth, random});
            const double extra = fetch.pJPerAccess - load.pJPerAccess;
            std::cout << "  " << std::left << std::setw(6) << level << std::right << std::setw(10)
                << Fixed(fetch.pJPerAccess / DefaultWidth, 2) << std::setw(10) << Fixed(load.pJPerAccess / DefaultWidth, 2)
                << std::setw(11) << Fixed(extra / DefaultWidth, 2) << std::setw(14) << Fixed(extra * 32.0 / 1000.0, 2) << " nJ"
                << "  " << fetch.source << "\n";
            table["texture"].push_back({{"level", level}, {"pj_per_access", ToJson(fetch.pJPerAccess)},
                {"pj_per_byte", ToJson(fetch.pJPerAccess / DefaultWidth)}, {"ld_pj_per_access", ToJson(load.pJPerAccess)},
                {"extra_pj_per_access", ToJson(extra)}, {"source", fetch.source}});
        }
    }
}

// Mixed intensity: the 16-byte loads of each level with 0 to 1024 FFMAs per load, against the sum of the two probes, the load
// of the level at the measured load rate plus the FFMA (net of its loop) at the FFMA rate, as Antepara et al. check their
// model over Mixbench's arithmetic intensities. The sum leaves out the 3 instructions of the mixed probe's loop around every
// 64 FFMAs (IADD3, ISETP, BRA). Without --compute in the same run there is no FFMA energy, and only the measurement is printed.
// Near the balance at DRAM both units run at full rate, and an RTX 3090 reaches its power limit there: a row whose clock the
// power cap lowered is shown with its clock and limiter, but it is not in the error summary (the probes ran at the baseline
// clock).
void PrintMixedIntensity(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const Footprints& footprints,
    nlohmann::json& table)
{
    const std::string random = PayloadName(PayloadRandom);
    const Variant variant{DefaultWidth, random};

    if (ProbeRows(rows, MixedName(0), "L1", footprints.l1, variant).empty())
    {
        return;
    }

    const Variant anyWidth{0, random};
    const Estimate ffma = EstimateEnergy(rows, fits, ComputeName(ComputeFfma), "compute", 0, anyWidth);
    const Estimate loop = EstimateEnergy(rows, fits, ComputeName(ComputeLoop), "compute", 0, anyWidth);
    const double ffmaNet = ffma.pJPerAccess - (std::isnan(loop.pJPerAccess) ? 0.0 : loop.pJPerAccess);
    std::cout << "\nMixed intensity (16-byte loads with FFMAs between them, duty 100, random data; 'sum' = the ld probe of the "
        << "level at the load rate + the FFMA probe net of its loop, " << Fixed(ffmaNet, 2) << " pJ, at the FFMA rate):\n"
        << "  " << std::left << std::setw(6) << "level" << std::right << std::setw(10) << "FFMA/ld" << std::setw(9) << "GB/s"
        << std::setw(10) << "GFLOP/s" << std::setw(8) << "T [C]" << std::setw(12) << "P-base [W]" << std::setw(8) << "ld [W]"
        << std::setw(10) << "FFMA [W]" << std::setw(9) << "sum [W]" << std::setw(8) << "error" << std::setw(10) << "pJ/load"
        << "\n";
    table["mixed"] = nlohmann::json::array();
    std::string summary;

    for (const auto& [level, footprint] : std::vector<std::pair<std::string, uint64_t>>{{"L1", footprints.l1},
        {"L2", footprints.l2}, {"DRAM", footprints.dram}})
    {
        const Estimate load = EstimateEnergy(rows, fits, "ld", level, footprint, variant);
        std::vector<double> errors;

        for (const uint64_t fmas : MixedFmas)
        {
            const Row* row = SelectRow(rows, MixedName(fmas), level, footprint, variant);

            if (row == nullptr || std::isnan(row->baselineW) || !(row->durationS > 0.0))
            {
                continue;
            }

            const double loadRate = static_cast<double>(row->accesses) / row->durationS;
            const double ffmaRate = static_cast<double>(fmas) * loadRate;
            const double dynamicW = row->powerW - row->baselineW;
            const double loadW = load.pJPerAccess * loadRate * 1e-12;
            const double ffmaW = fmas == 0 ? 0.0 : ffmaNet * ffmaRate * 1e-12;
            const double sumW = loadW + ffmaW;
            const double error = sumW / dynamicW - 1.0;
            const bool matched = ClockMatched(*row);
            std::string note;

            if (!matched)
            {
                AppendNote(note, "SM clock " + SignedPercent(row->clockDifference) + ", not in the summary");
            }

            if (!row->limiters.empty())
            {
                AppendNote(note, row->limiters);
            }

            if (!std::isnan(error) && matched)
            {
                errors.push_back(error);
            }

            std::cout << "  " << std::left << std::setw(6) << level << std::right << std::setw(10) << fmas << std::setw(9)
                << Fixed(row->bandwidthGBs, 0) << std::setw(10) << Fixed(2.0 * ffmaRate * 1e-9, 0) << std::setw(8)
                << Fixed(row->temperatureC, 0) << std::setw(12) << Fixed(dynamicW, 1) << std::setw(8) << Fixed(loadW, 1)
                << std::setw(10) << Fixed(ffmaW, 1) << std::setw(9) << Fixed(sumW, 1) << std::setw(8) << SignedPercent(error)
                << std::setw(10) << Fixed(row->pJPerAccess, 1) << (note.empty() ? "" : "  " + note) << "\n";
            table["mixed"].push_back({{"level", level}, {"fmas_per_load", fmas}, {"gb_per_s", ToJson(row->bandwidthGBs)},
                {"gflop_per_s", ToJson(2.0 * ffmaRate * 1e-9)}, {"dynamic_w", ToJson(dynamicW)}, {"ld_w", ToJson(loadW)},
                {"ffma_w", ToJson(ffmaW)}, {"sum_w", ToJson(sumW)}, {"pj_per_load", ToJson(row->pJPerAccess)},
                {"clock_matched", matched}, {"limiters", row->limiters}});
        }

        if (!errors.empty() && !std::isnan(ffmaNet))
        {
            double absolute = 0.0;
            double signedSum = 0.0;

            for (const double error : errors)
            {
                absolute += std::abs(error);
                signedSum += error;
            }

            const double count = static_cast<double>(errors.size());
            summary += (summary.empty() ? "" : "; ") + level + " " + Fixed(100.0 * absolute / count, 1) + " % (bias "
                + SignedPercent(signedSum / count) + ")";
        }
    }

    std::cout << "  sum vs measured, mean absolute error: " << (summary.empty() ? "no FFMA energy (run with --compute)"
        : summary) << "\n";
}

void PrintControlVsDatapath(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits,
    const Footprints& footprints, const Options& options, nlohmann::json& table)
{
    const std::string random = PayloadName(PayloadRandom);
    const std::string zero = PayloadName(PayloadZero);
    const Variant anyWidth{0, random};
    const std::vector<uint64_t> computeOps = MeasuredComputeOps(rows);

    // Control vs datapath (Antepara et al.): with zeros, only control (fetch, decode, addresses, tags, ...) switches; the
    // difference to random data is the energy of moving and computing the bits.
    const std::vector<Probe> probes = {
        {"shared ld", "ld.shared", "shared", 0, DefaultWidth},
        {"shared st", "st.shared", "shared", 0, DefaultWidth},
        {"L1 ld", "ld", "L1", footprints.l1, DefaultWidth},
        {"L1 ld.nc", "ld.nc", "L1", footprints.l1, DefaultWidth},
        {"L2 ld", "ld", "L2", footprints.l2, DefaultWidth},
        {"L2 st", "st", "L2", footprints.l2, DefaultWidth},
        {"DRAM ld", "ld", "DRAM", footprints.dram, DefaultWidth},
        {"DRAM st", "st", "DRAM", footprints.dram, DefaultWidth},
        {"no-load", "no-load", "none", 0, DefaultWidth},
        {"L2 red", "red", "L2", footprints.l2, AtomicWidth},
        {"L2 atom", "atom", "L2", footprints.l2, AtomicWidth},
        {"DRAM red", "red", "DRAM", footprints.dram, AtomicWidth},
        {"DRAM atom", "atom", "DRAM", footprints.dram, AtomicWidth},
        {"shared red", "red.shared", "shared", 0, AtomicWidth},
        {"L1 tex", "tex", "L1", footprints.l1, DefaultWidth},
        {"L2 tex", "tex", "L2", footprints.l2, DefaultWidth},
        {"DRAM tex", "tex", "DRAM", footprints.dram, DefaultWidth}};

    if (std::find(options.payloads.cbegin(), options.payloads.cend(), PayloadZero) != options.payloads.cend())
    {
        std::cout << "\nControl vs datapath (zero vs random data; pJ/B for memory, pJ/op for arithmetic):\n"
            << "  " << std::left << std::setw(10) << "probe" << std::right << std::setw(10) << "total" << std::setw(10)
            << "control" << std::setw(10) << "datapath" << std::setw(10) << "datapath%" << "\n";
        table["control_datapath"] = nlohmann::json::array();

        auto line = [&](const std::string& label, const Estimate& total, const Estimate& control, const double divisor)
        {
            if (std::isnan(total.pJPerAccess) || std::isnan(control.pJPerAccess))
            {
                return;
            }

            const double t = total.pJPerAccess / divisor;
            const double c = control.pJPerAccess / divisor;
            std::cout << "  " << std::left << std::setw(10) << label << std::right << std::setw(10) << Fixed(t, 2)
                << std::setw(10) << Fixed(c, 2) << std::setw(10) << Fixed(t - c, 2) << std::setw(10)
                << Fixed((t - c) / t * 100.0, 0) << "\n";
            table["control_datapath"].push_back({{"probe", label}, {"total", t}, {"control", c}, {"datapath", t - c},
                {"unit", divisor == 1.0 ? "pJ/op" : "pJ/B"}});
        };

        for (const auto& probe : probes)
        {
            line(probe.label, EstimateEnergy(rows, fits, probe.operation, probe.level, probe.footprint, {probe.width, random}),
                EstimateEnergy(rows, fits, probe.operation, probe.level, probe.footprint, {probe.width, zero}),
                static_cast<double>(probe.width));
        }

        for (const uint64_t op : computeOps)
        {
            line(ComputeName(op), EstimateEnergy(rows, fits, ComputeName(op), "compute", 0, anyWidth),
                EstimateEnergy(rows, fits, ComputeName(op), "compute", 0, {0, zero}), 1.0);
        }

        std::cout << "  no-load moves no data, so its datapath share should be about 0 (a check of the method).\n";
    }
}

void PrintWidthModel(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const Footprints& footprints,
    const Options& options, nlohmann::json& table)
{
    const std::string random = PayloadName(PayloadRandom);

    // Energy per request vs per byte (Wattchmen measures loads of 8 to 128 bits): the access energy net of the loop at each
    // width, fitted as per-request energy + width x per-byte energy.
    if (options.widths.size() > 1)
    {
        std::cout << "\nEnergy per request vs per byte (net of the no-load loop of the same width, random data):\n"
            << "  " << std::left << std::setw(6) << "level";

        for (const uint64_t width : options.widths)
        {
            std::cout << std::right << std::setw(11) << (std::to_string(width) + " B [pJ]");
        }

        std::cout << std::setw(13) << "pJ/request" << std::setw(9) << "pJ/B" << std::setw(7) << "R^2" << "\n";
        table["width_model"] = nlohmann::json::array();

        for (const auto& [level, footprint] : std::vector<std::pair<std::string, uint64_t>>{{"L1", footprints.l1},
            {"L2", footprints.l2}, {"DRAM", footprints.dram}})
        {
            std::vector<std::pair<double, double>> points;
            std::cout << "  " << std::left << std::setw(6) << level << std::right;

            for (const uint64_t width : options.widths)
            {
                const Estimate access = EstimateEnergy(rows, fits, "ld", level, footprint, {width, random});
                const Estimate overhead = EstimateEnergy(rows, fits, "no-load", "none", 0, {width, random});
                const double net = access.pJPerAccess - overhead.pJPerAccess;
                std::cout << std::setw(11) << Fixed(net, 1);

                if (!std::isnan(net))
                {
                    points.emplace_back(static_cast<double>(width), net);
                }
            }

            const Line fit = FitLine(points);
            std::cout << std::setw(13) << Fixed(fit.intercept, 1) << std::setw(9) << Fixed(fit.slope, 2) << std::setw(7)
                << Fixed(fit.r2, 3) << "\n";
            table["width_model"].push_back({{"level", level}, {"pj_per_request", ToJson(fit.intercept)},
                {"pj_per_byte", ToJson(fit.slope)}, {"r2", ToJson(fit.r2)}});
        }

        std::cout << "  a request is one thread's access; a warp instruction issues 32 of them.\n";
    }
}

void PrintSmSweep(const std::vector<Row>& rows, const Footprints& footprints, nlohmann::json& table)
{
    const std::string random = PayloadName(PayloadRandom);
    const std::vector<const Row*> idle = IdleRows(rows);

    // Power of SMs (AccelWattch models power-gated SMs): the sleep baseline and some probes with only N SMs working.
    std::map<uint64_t, const Row*> sweepSleeps;

    for (const auto& row : rows)
    {
        if (row.level == "baseline" && row.activeSms != 0 && row.hasPower)
        {
            sweepSleeps[row.activeSms] = &row;
        }
    }

    if (!sweepSleeps.empty())
    {
        std::cout << "\nSM sweep (duty 100, random data; probes compared with the sleep baseline on the same SMs):\n";
        std::vector<std::pair<double, double>> sleepPoints;

        for (const auto& [count, row] : sweepSleeps)
        {
            sleepPoints.emplace_back(static_cast<double>(count), row->powerW);
        }

        const Line sleepLine = FitLine(sleepPoints);
        std::cout << "  sleep baseline: " << Fixed(sleepLine.intercept, 1) << " W + " << Fixed(sleepLine.slope, 3)
            << " W per SM with sleeping warps (R^2 " << Fixed(sleepLine.r2, 3) << ")";

        if (!idle.empty())
        {
            std::cout << "; idle without a kernel: " << Fixed(idle.front()->powerW, 1) << " W";
        }

        std::cout << "\n  " << std::left << std::setw(10) << "probe" << std::right << std::setw(5) << "SMs" << std::setw(9)
            << "P [W]" << std::setw(11) << "P-sleep" << std::setw(11) << "pJ/access" << std::setw(10) << "GB/s" << "\n";
        table["sm_sweep"] = {{"sleep_intercept_w", ToJson(sleepLine.intercept)},
            {"sleep_w_per_sm", ToJson(sleepLine.slope)}, {"probes", nlohmann::json::array()}};

        const std::vector<Probe> sweepProbes = {
            {"L1 ld", "ld", "L1", footprints.l1, DefaultWidth},
            {"no-load", "no-load", "none", 0, DefaultWidth},
            {"L2 ld", "ld", "L2", footprints.l2, DefaultWidth},
            {"shared ld", "ld.shared", "shared", 0, DefaultWidth},
            {"ffma", ComputeName(ComputeFfma), "compute", 0, 0},
            // With --latency: dependent loads with 1 and 16 chains per warp (operations chase-c<chains>-w<warps per SM>).
            {"L2 c1", "chase-c1-", "L2", footprints.l2, 4},
            {"L2 c16", "chase-c16-", "L2", footprints.l2, 4},
            {"DRAM c1", "chase-c1-", "DRAM", footprints.dram, 4},
            {"DRAM c16", "chase-c16-", "DRAM", footprints.dram, 4}};

        for (const auto& probe : sweepProbes)
        {
            std::vector<std::pair<double, double>> points;

            for (const auto& row : rows)
            {
                // An operation that ends in '-' is a prefix (the latency probes' names end in their warps per SM).
                const bool operation = probe.operation.back() == '-' ? row.operation.rfind(probe.operation, 0) == 0
                    : row.operation == probe.operation;

                if (row.activeSms == 0 || !operation || row.level != probe.level
                    || std::isnan(row.baselineW) || (probe.footprint != 0 && row.footprint != probe.footprint))
                {
                    continue;
                }

                std::cout << "  " << std::left << std::setw(10) << probe.label << std::right << std::setw(5) << row.activeSms
                    << std::setw(9) << Fixed(row.powerW, 1) << std::setw(11) << Fixed(row.powerW - row.baselineW, 1)
                    << std::setw(11) << Fixed(row.pJPerAccess, 1) << std::setw(10) << Fixed(row.bandwidthGBs, 1) << "  "
                    << row.note << "\n";

                // A row that ran at another clock than the baseline (throttled below the lock) is listed, but it is not a
                // point of the line. A row at or below its baseline (a few SMs, within the noise) is one.
                if (ClockMatched(row) || std::isnan(row.clockDifference))
                {
                    points.emplace_back(static_cast<double>(row.activeSms), row.powerW - row.baselineW);
                }
            }

            if (points.empty())
            {
                continue;
            }

            const Line fit = FitLine(points);
            std::cout << "  " << std::left << std::setw(10) << probe.label << std::right << "  P-sleep = "
                << Fixed(fit.intercept, 1) << " W + " << Fixed(fit.slope, 3) << " W per active SM (R^2 " << Fixed(fit.r2, 3)
                << ")\n";
            table["sm_sweep"]["probes"].push_back({{"probe", probe.label}, {"intercept_w", ToJson(fit.intercept)},
                {"w_per_sm", ToJson(fit.slope)}, {"r2", ToJson(fit.r2)}});
        }

        std::cout << "  an intercept above 0 is power that comes with the first working SM (chip-wide parts such as L2 and the\n"
            << "  crossbar), the slope the power of each further SM.\n";
    }
}

} // namespace

nlohmann::json PrintStudies(const std::vector<Row>& rows, const std::vector<ProbeFit>& fits, const Footprints& footprints,
    const BaselineModel& baseline, const Hardware& hardware, const Options& options)
{
    nlohmann::json table;
    table["device"] = hardware.name;
    // tools/check_counters.py reads it to tell counters a GPU cannot have from missing ones.
    table["compute_capability"] = std::to_string(hardware.computeCapabilityMajor) + "."
        + std::to_string(hardware.computeCapabilityMinor);
    table["sm_clock_mhz"] = ToJson(baseline.clockMHz);
    table["fit_min_duty_pct"] = options.fitMinDuty;
    table["baseline"] = {{"runs", baseline.runs}, {"mean_w", ToJson(baseline.meanW)},
        {"slope_w_per_c", baseline.slopeWPerC}, {"mean_temperature_c", ToJson(baseline.meanTemperatureC)}};

    PrintConstantAndStaticPower(rows, baseline, table);
    AddFits(fits, table);
    PrintArithmetic(rows, fits, table);
    PrintIndexing(rows, fits, table);
    PrintAtomics(rows, fits, footprints, table);
    PrintConstantMemory(rows, fits, footprints, table);
    PrintDramLocality(rows, fits, footprints, table);
    PrintSubSector(rows, fits, footprints, table);
    PrintBankConflicts(rows, fits, table);
    PrintTexture(rows, fits, footprints, table);
    PrintMixedIntensity(rows, fits, footprints, table);
    PrintWrites(rows, fits, footprints, table);
    PrintLatency(rows, fits, footprints, table);
    PrintControlVsDatapath(rows, fits, footprints, options, table);
    PrintWidthModel(rows, fits, footprints, options, table);
    PrintSmSweep(rows, footprints, table);
    return table;
}
