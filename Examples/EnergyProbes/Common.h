// Constants, operation codes and small helpers shared by all parts of the program.
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <json.hpp>

constexpr uint64_t KiB = 1024;
constexpr uint64_t MiB = 1024 * KiB;
constexpr uint64_t GiB = 1024 * MiB;

constexpr uint64_t DefaultWidth = 16;       // bytes per access and thread: one ld.v4.u32
constexpr uint64_t SectorBytes = 32;
constexpr uint64_t LineBytes = 128;
constexpr uint64_t SmCounterEntries = 1024; // must match SM_COUNTER_ENTRIES in kernels/Common.cuh
constexpr uint64_t ConstantWords = 16384;   // 64 KiB of constant memory, must match CONSTANT_WORDS in kernels/Constant.cu
constexpr uint64_t AtomicWidth = 4;         // bytes per atomic access and thread (32-bit atomics)
constexpr uint64_t WavefrontBytes = 128;    // a shared-memory wavefront serves up to 128 bytes
constexpr uint32_t ScatterStride = 0x9e3779b1u; // odd multiplier that places the runs of the locality probe
constexpr uint64_t WriteLocalFrameBytes = 1024; // local memory per thread, must match LOCAL_FRAME_BYTES in kernels/Write.cu
constexpr uint32_t AllSms = 0xffffffffu;    // activeSms argument of the kernels: every SM works
constexpr double ClockTolerance = 0.02;     // relative SM clock difference to the baseline above which energies are withheld
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// Must match LOAD_OP in kernels/Global.cu (1, 3 and 4 are the ld.cg, ld.ca and ld.nc variants, which the run leaves out).
enum LoadOp : uint64_t
{
    LoadDefault = 0,
    LoadNone = 2,
    StoreDefault = 5,
    AtomicRed = 6, // red.global.add.u32, atomicAdd with unused result
    AtomicAdd = 7  // atom.global.add.u32, returns the old value
};

// Must match SHARED_OP in kernels/Shared.cu.
enum SharedOp : uint64_t
{
    SharedLoad = 0,
    SharedStore = 1,
    SharedRed = 2
};

// Must match COMPUTE_OP in kernels/Compute.cu.
enum ComputeOp : uint64_t
{
    ComputeLoop = 0,
    ComputeFfma = 1,          // x * x + c: one register read twice and a constant
    ComputeDfma = 2,
    ComputeImad = 3,
    ComputeHfma2 = 4,
    ComputeHmma = 5,
    ComputeMufu = 6,
    ComputeDynamicIndex = 8,  // 32 FFMA accumulators with a runtime index: compares and guarded moves
    ComputeFfma3 = 9,         // FFMA with three different register operands (acc = a * b + acc in real kernels)
    ComputeShfl = 10,         // warp shuffle
    ComputeBar = 11,          // block barrier
    ComputeFfmaLoop = 12,     // the FFMA chain in a loop that is not unrolled (branches)
    // The other tensor-core paths, one warp instruction per step as HMMA (IMMA in the compute group, the others in the
    // optional group "tensor").
    ComputeImma = 13,         // mma.sync m16n8k32 s8 (sm_80+)
    ComputeDmma = 14,         // mma.sync m8n8k4 f64 (sm_80+)
    ComputeHmmaSparse = 15,   // mma.sp m16n8k32 f16, 2:4 sparsity (sm_80+)
    ComputeQmmaFp8 = 16,      // mma.sync m16n8k32 e4m3 (native on sm_89 and sm_120 only)
    ComputeOpcode = 17        // one SASS opcode (COMPUTE_OPCODE), the optional group "opcodes"
};

// COMPUTE_OPCODE of the opcode probes (kernels/Compute.cu): 1-based names, and whether a constant-operand variant exists.
std::string OpcodeName(const uint64_t opcode);
constexpr uint64_t OpcodeCount = 19;
constexpr uint64_t OpcodeLastWithConstant = 7;
// COMPUTE_OPCODE 20-22: FFMA with the same operand values in all chains, from 0, 1 or 2 shared registers (the operand
// reuse cache), the optional group "reuse"; not part of the "opcodes" group.
constexpr uint64_t OpcodeReuseFirst = 20;
constexpr uint64_t OpcodeReuseLast = 22;

// Must match WRITE_OP in kernels/Write.cu.
enum WriteOp : uint64_t
{
    WriteCopy = 0,
    WriteUpdate = 1,
    WriteLocal = 2
};

std::string OperationName(const uint64_t loadOp);

std::string ComputeName(const uint64_t computeOp);

bool IsPowerOfTwo(const uint64_t value);

uint64_t FloorPowerOfTwo(const uint64_t value);

uint64_t CeilPowerOfTwo(const uint64_t value);

std::string FormatBytes(const uint64_t bytes);

// "-" for a missing value.
std::string Fixed(const double value, const int precision);

// Empty cell for missing values. The energy counter counts from driver load, so it needs many digits to keep mJ resolution.
std::string CsvValue(const double value, const int precision = 10);

// A text cell of a CSV file: the cells are written without quotes, so commas and line breaks become spaces.
std::string CsvText(std::string text);

void AppendNote(std::string& note, const std::string& text);

std::string SignedPercent(const double fraction);

std::string JoinList(const std::vector<uint64_t>& values, const std::string& separator);

// Whether a clock ran where it was expected (Fan, Cosenza, Juurlink, ICPP'19: some clock pairs that NVML lists as
// supported are not applied). Empty if `observedMHz` is within one clock step of `expectedMHz`, else a warning naming
// both. One step: the distance from the supported clock nearest to `expectedMHz` to its nearest supported neighbour
// (`supportedMHz`, NVML's list), or `fallbackStepMHz` without a list. Empty too if either value is unknown (NaN).
std::string ClockMismatch(const std::string& domain, const double expectedMHz, const double observedMHz,
    const std::vector<double>& supportedMHz, const double fallbackStepMHz);

nlohmann::json ToJson(const double value);

uint32_t FloatBits(const float value);

// Least-squares line y = intercept + slope * x; r2 with three points or more.
struct Line
{
    double slope = NaN;
    double intercept = NaN;
    double r2 = NaN;
};

Line FitLine(const std::vector<std::pair<double, double>>& points);
