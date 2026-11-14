// Constants, operation codes and small helpers shared by all parts of the program.
#pragma once

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>
#include <json.hpp>

constexpr uint64_t KiB = 1024;

constexpr uint64_t MiB = 1024 * KiB;

constexpr uint64_t GiB = 1024 * MiB;

constexpr uint64_t DefaultWidth = 16; // bytes per access and thread: one ld.v4.u32

constexpr uint64_t SectorBytes = 32;

constexpr uint64_t SmCounterEntries = 1024; // must match SM_COUNTER_ENTRIES in kernels/Common.cuh

constexpr uint64_t ConstantWords = 16384;   // 64 KiB of constant memory, must match CONSTANT_WORDS in kernels/Constant.cu

constexpr uint64_t AtomicWidth = 4;         // bytes per atomic access and thread (32-bit atomics)

// The bank-conflict probe (probeSharedBanks): bytes per access and thread, and conflict degrees (times the wavefronts of
// the conflict-free access). A wavefront serves up to 128 bytes, so a degree runs only with widths of ways * width <= 128:
// up to 32 wavefronts per instruction.
const std::vector<uint64_t> BankWidths = {4, 8, 16};

const std::vector<uint64_t> BankWays = {1, 2, 4, 8, 16, 32};

constexpr uint64_t WavefrontBytes = 128;

// The degrees the bank-conflict probe runs for one width.
std::vector<uint64_t> BankWaysOfWidth(const uint64_t width);

// Bytes of consecutive sectors per run of the locality probe (probeScatter); 0 = one stream through the footprint.
const std::vector<uint64_t> LocalityRuns = {0, 32, 64, 128, 512, 2048, 8192};

constexpr uint32_t ScatterStride = 0x9e3779b1u; // odd multiplier that places the runs (Fibonacci hashing)

// FFMAs per load of the mixed-intensity probe (probeMixed), from memory-bound to compute-bound at every level (on an RTX
// 3090 the two take the same time at about 14, 81 and 185 FFMAs per load at L1, L2 and DRAM). The FFMAs come in groups of
// MixedFmaGroup after every --unroll loads, so a count must be a multiple of MixedFmaGroup / --unroll (4 at the default).
const std::vector<uint64_t> MixedFmas = {0, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

constexpr uint64_t MixedFmaGroup = 64; // must match MIX_GROUP in kernels/Mixed.cu

// Operation name of the mixed-intensity probe with `fmasPerLoad` FFMAs per load ("ld-ffma16"), and the count of such a name.
std::string MixedName(const uint64_t fmasPerLoad);

std::optional<uint64_t> MixedFmasOf(const std::string& operation);

// The write probe (probeWrite, kernels/Write.cu): loads and stores together. WRITE_OP: copy (a read and a write stream),
// update in place (lines loaded, dirtied and evicted dirty) and local memory (LDL, STL, as spills). Copy and update
// run at the L2 and DRAM footprints; local memory per SM at the L1 footprint and with the whole frame of every thread.
enum WriteOp : uint64_t
{
    WriteCopy = 0,
    WriteUpdate = 1,
    WriteLocal = 2
};

const std::vector<uint64_t> WriteWidths = {4, 16};

constexpr uint64_t WriteLocalFrameBytes = 1024; // local memory per thread, must match LOCAL_FRAME_BYTES in kernels/Write.cu

constexpr uint64_t WriteGroup = 4; // loads, then stores, per group of the write probe; must match WRITE_GROUP in kernels/Write.cu

std::string WriteName(const uint64_t operation);

// The latency probe (probeLatency, kernels/Latency.cu): chains of dependent loads per warp, and working warps per block
// (x BLOCKS_PER_SM per SM), from latency-bound to the bandwidth of the plain loads. Duty 100, random data.
const std::vector<uint64_t> LatencyChains = {1, 4, 16};
const std::vector<uint64_t> LatencyWarps = {1, 2, 8};

// Operation name of a latency probe: "chase-c4-w8" for 4 chains per warp and 8 working warps per SM.
std::string LatencyName(const uint64_t chains, const uint64_t warpsPerSm);

// Whether an operation name is one of the write probe's.
bool IsWriteOperation(const std::string& operation);

constexpr uint32_t AllSms = 0xffffffffu;    // activeSms argument: every SM works

constexpr double ClockTolerance = 0.02; // relative SM clock difference to the baseline above which energies are withheld

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// Must match LOAD_OP in kernels/Global.cu.
enum LoadOp : uint64_t
{
    LoadDefault = 0,
    LoadCg = 1,
    LoadNone = 2,
    LoadCa = 3,
    LoadNc = 4,
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
    ComputeFfma = 1,
    ComputeDfma = 2,
    ComputeImad = 3,
    ComputeHfma2 = 4,
    ComputeHmma = 5,
    ComputeMufu = 6,
    ComputeStaticIndex = 7, // 32 FFMA accumulators, static indices
    ComputeDynamicIndex = 8, // the same with a runtime index: compares and guarded moves
    ComputeFfma3 = 9,        // FFMA with three different register operands (acc = a * b + acc in real kernels)
    ComputeShfl = 10,        // instruction classes without a probe before: warp shuffle,
    ComputeBar = 11,         // block barrier,
    ComputeFfmaLoop = 12,    // the FFMA chain in a loop that is not unrolled (branches),
    ComputeFmulFadd = 13,    // x * x + c as a separate FMUL and FADD,
    ComputeMufuSin = 14,     // MUFU.SIN
    // --lanes: partially active warps, LANES of 32 lanes active
    ComputeFfmaLanes = 15,       // FFMA with a guard predicate,
    ComputeFfmaLanesBranch = 16, // FFMA, the other lanes branch around the body,
    ComputeIaddLanes = 17,       // IADD with a guard predicate,
    ComputeIaddLanesBranch = 18, // IADD in a branch
    // --instr-classes: instruction classes inside one event
    ComputeLop3 = 19,     // xor chain (ptxas merges pairs into three-input LOP3s)
    ComputeShf = 20,      // rotate by a variable amount
    ComputeImnmx = 21,    // min, then xor
    ComputeFadd = 22,     // x -> c - x
    ComputeFmul = 23,     // x -> x * k, then x * (1 / k)
    ComputeMufuRcp = 24,  // the FFMA recurrence, MUFU, FADD into a sum
    ComputeMufuSqrt = 25,
    ComputeMufuEx2 = 26,
    ComputeMufuLg2 = 27,
    ComputeMufuRsq = 28,
    // --instr-classes, added 2026-11-14 (NEXT_STEPS_PLAN M4)
    // (A register move cannot be measured alone: ptxas removes moves that the PTX asks for.)
    ComputeIsetpSel = 29, // integer compare + select
    ComputeFsetpSel = 30, // float compare + select
    ComputeCvt = 31,      // integer -> float, FADD, -> integer (ptxas folds the two conversions without the FADD)
    ComputeFmnmx = 32,    // float min, then max
    ComputeIadd3 = 33,    // three-input integer add
    // --lanes, added 2026-11-14 (NEXT_STEPS_PLAN M5)
    ComputeFfmaLanesRotating = 34 // FFMA with a guard predicate, the active lanes rotating from step to step
};

// --instr-classes: the operations of the group.
inline const std::vector<uint64_t> InstructionClassOps = {ComputeLop3, ComputeShf, ComputeImnmx, ComputeFadd, ComputeFmul,
    ComputeMufuRcp, ComputeMufuSqrt, ComputeMufuEx2, ComputeMufuLg2, ComputeMufuRsq, ComputeIsetpSel,
    ComputeFsetpSel, ComputeCvt, ComputeFmnmx, ComputeIadd3};

// --misaligned: start offsets in bytes (0 = aligned; 16 = half a sector; 32, 64, 96 = 1-3 sectors off the line). Multiples
// of 16 only: the store probe seeds its values with a 16-byte read at the start of its data (a 4-byte offset stopped the
// run with CUDA_ERROR_MISALIGNED_ADDRESS).
inline const std::vector<uint64_t> MisalignedOffsets = {0, 16, 32, 64, 96};

// --lanes: the active lanes per warp of the partially active probes.
// 0 since 2026-11-14 (M5): the step from no active lane to one is the whole fixed part of a warp instruction.
inline const std::vector<uint64_t> LaneCounts = {0, 1, 2, 4, 8, 16, 24, 32};

// PAYLOAD parameter: the data the probes read, store or compute on.
enum Payload : uint64_t
{
    PayloadRandom = 0,
    PayloadZero = 1,
    PayloadPatternFirst = 2 // the patterns of --payload-bits, in their order (Payloads.h)
};

std::string PayloadName(const uint64_t payload);

std::string ComputeName(const uint64_t operation);

// OPERAND_OP parameter of probeOperands (--payload-bits, kernels/Operands.cu).
enum OperandOperation : uint64_t
{
    OperandFfma = 0,
    OperandDfma = 1,
    OperandHmma = 2 // sm_80 or newer
};

std::string OperandName(const uint64_t operation);

// FLOPs (integer operations for IMAD) of one operation of one thread; HMMA's depend on the shape of the GPU (compute capability major 7: 2048 FLOPs
// per warp instruction, 8 and newer: 4096).
double FlopsPerOperation(const uint64_t operation, const uint32_t computeCapabilityMajor = 8);

std::string OperationName(const uint64_t operation);

bool IsPowerOfTwo(const uint64_t value);

uint64_t FloorPowerOfTwo(const uint64_t value);

uint64_t CeilPowerOfTwo(const uint64_t value);

std::string FormatBytes(const uint64_t bytes);

std::string Fixed(const double value, const int precision);

// Empty cell for missing values. The energy counter counts from driver load, so it needs many digits to keep mJ resolution.
std::string CsvValue(const double value, const int precision = 10);

void AppendNote(std::string& note, const std::string& text);

// A text cell of a CSV file: the cells are written without quotes, so a comma or a line break inside one would shift every
// column after it (a placement note with a comma once moved all CUPTI counters of its row by one column). Both become
// spaces.
std::string CsvText(std::string text);

std::string SignedPercent(const double fraction);

std::string JoinList(const std::vector<uint64_t>& values, const std::string& separator);

nlohmann::json ToJson(const double value);

uint32_t FloatBits(const float value);
