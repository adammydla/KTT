// Common part of the energy probes, put in front of every probe file by the host (see KernelSource in ProbeKernels.cpp),
// which compiles every probe file as a program of its own.
//
// All probes are time-bounded: every thread loops until `durationNs` has passed (measured with %globaltimer) and counts
// how many accesses (or operations) it issued. Block b adds its count to counts[b] and its active time to
// counts[gridDim.x + b], and block 0 counts the launch in counts[3 * gridDim.x]. The host clears the buffer before the
// launches of a configuration, so after KTT's repeated launches it holds sums over all of them. The probes that split
// their footprint by SM (probeGlobal, probeTexture, probeConstant) also write their SM id and slot into
// counts[2 * gridDim.x + b] (overwritten on every launch).
//
// Tuning parameters shared by several probes (injected by KTT as #defines; each probe file lists its own):
//   BLOCK_SIZE      threads per block (power of two)
//   BLOCKS_PER_SM   resident blocks per SM; the probes with SM slots split their footprint into stripes so that the blocks
//                   (warps) of one SM share no lines
//   UNROLL          accesses between two timer checks
//   BURST_ACCESSES  accesses per burst (multiple of UNROLL, default 64). After every burst a thread sleeps for idleRatio
//                   times the burst's duration. Long enough that the sleep after the fastest probe's burst is several
//                   microseconds, since nanosleep cannot honour sleeps much shorter than that.
//   LOAD_OP, WIDTH, FOOTPRINT, GRID_STRIPES   the global-memory probes (see Global.cu)
//
// Runtime arguments, set by the host for every configuration (arguments rather than parameters, because ptxas folds
// compile-time values and the SASS would differ between configurations that should only differ in data):
//   idleRatio   (100 - duty) / duty in 16.16 fixed point, 0 at 100 %. The duty is the percentage of time spent issuing
//               accesses. Varying it varies the access rate; fitting power against rate separates the cost per access from
//               the power that does not depend on the rate. Accesses are counted, so the achieved duty need not be exact.
//   activeSms   blocks on SMs with %smid >= activeSms return at once, so only that many SMs work while the others idle
//               (all SMs for ~0u).
//   payloadMask probeShared and probeCompute: ~0u for random data, 0 for zeros. Computing on zeros exercises only the
//               control path (fetch, decode, addresses, tags); the difference to random data is datapath energy. The
//               global-memory probes get a buffer of zeros from the host instead.
//
// The access loops are spelled out in every probe, also where they repeat (the stored-word recurrence of the store
// probes): factoring them into helper structs changed the register allocation and instruction order of measured
// variants, and every probe's SASS must stay as measured (tools/check_sass.py --against).
//
// Footprint masks go through Opaque() so that the compiler cannot exploit them (e.g. notice that a small footprint repeats
// addresses and drop address arithmetic). Where ptxas folded a compile-time footprint anyway, the probe takes it as an
// argument. Either way the SASS is identical for all footprints except at most one immediate, which is what makes
// differences between footprints attributable to the memory hierarchy alone.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCKS_PER_SM
#define BLOCKS_PER_SM 4
#endif
#ifndef BURST_ACCESSES
#define BURST_ACCESSES 64
#endif
#ifndef LOAD_OP
#define LOAD_OP 0
#endif
#ifndef WIDTH
#define WIDTH 16
#endif
#ifndef FOOTPRINT
#define FOOTPRINT 32768
#endif
#ifndef GRID_STRIPES
#define GRID_STRIPES 1024
#endif

static_assert(BURST_ACCESSES % UNROLL == 0 && BURST_ACCESSES > 0, "BURST_ACCESSES must be a positive multiple of UNROLL");

// Entries of the per-SM counter buffer of the probes with SM slots. Must match SmCounterEntries in Common.h.
#define SM_COUNTER_ENTRIES 1024

typedef unsigned int u32;
typedef unsigned long long u64;

// One access of a global-memory probe: WIDTH bytes per thread.
#if WIDTH == 16
typedef uint4 Element;
#elif WIDTH == 8
typedef uint2 Element;
#elif WIDTH == 4
typedef u32 Element;
#else
#error "Unsupported WIDTH"
#endif

__device__ __forceinline__ u64 globalTimer()
{
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Number of the SM the calling thread runs on.
__device__ __forceinline__ u32 SmId()
{
    u32 id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
    return id;
}

__host__ __device__ constexpr u32 CeilPowerOfTwo(const u32 value)
{
    u32 result = 1;

    while (result < value)
    {
        result *= 2;
    }

    return result;
}

__host__ __device__ constexpr u32 Log2(const u32 powerOfTwo)
{
    u32 result = 0;

    while ((1u << result) < powerOfTwo)
    {
        ++result;
    }

    return result;
}

// Hides a compile-time value from the optimizer without emitting an instruction.
__device__ __forceinline__ u32 Opaque(u32 value)
{
    asm volatile("" : "+r"(value));
    return value;
}

// Hides a pointer from the optimizer, so that offsets added to it are not folded into every address computed from it.
template <typename T>
__device__ __forceinline__ T* OpaquePointer(T* pointer)
{
    asm volatile("" : "+l"(pointer));
    return pointer;
}

// Integer hash used to fill shared memory and the compute operands with non-trivial data (bit toggling affects power).
__device__ __forceinline__ u32 mix(u32 x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Called after every UNROLL accesses. After BURST_ACCESSES accesses, sleeps for idleRatio (16.16 fixed point) times the
// duration of the burst and starts the next burst.
//
// The sleep code must cost the same instructions per access at every level and every duty, or differences between rows
// contain instruction energy:
//   - A burst is a fixed number of accesses. With a minimum burst time, slow probes ran the sleep code after fewer
//     accesses (at duty 50, 9.4 warp instructions per access for DRAM vs 7.3 for L1).
//   - One nanosleep per burst and no loop that waits for the wake-up time.
//   - The duty is a runtime value and the code also runs at 100 % (nanosleep 0). As a compile-time constant, ptxas
//     dropped the code at 100 % (6.50 instead of 7.08 instructions per access, which put the 100 % row off the line of
//     the other duties) and compiled a different division for every duty.
// nanosleep sleeps between 0 and twice the requested time (at most 1 ms), so the achieved duty is approximate. Before
// Volta there is no nanosleep, so the thread polls the timer instead, which is not level-independent.
__device__ __forceinline__ void IdleForDuty(const u32 idleRatio, u32& burstIterations, u64& burstStart, u64& now)
{
    if (--burstIterations != 0)
    {
        return;
    }

    burstIterations = BURST_ACCESSES / UNROLL;
    const u64 sleepNs = ((now - burstStart) * idleRatio) >> 16;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("nanosleep.u32 %0;" : : "r"(static_cast<u32>(sleepNs < 1000000ull ? sleepNs : 1000000ull)));
    now = globalTimer();
#else
    const u64 wakeUp = now + sleepNs;

    while (now < wakeUp)
    {
        now = globalTimer();
    }
#endif

    burstStart = now;
}

// Duty of the arithmetic probes. A few warps saturate an arithmetic pipeline, so when every warp sleeps at its own time
// (IdleForDuty), the others fill the pipeline and the rate hardly drops (FFMA ran at 85 % of its full rate at duty 50,
// DFMA and HMMA at 99 %). Here all warps of the GPU work in the same windows of the global timer: in every period of
// DUTY_PERIOD_NS they work for activeNs, then sleep until the next period. Checked after every loop iteration (64
// operations; the slowest, DFMA, takes about 30 us for them). At duty 100, activeNs is the whole period.
#define DUTY_PERIOD_NS (1u << 19) // 524 us

__device__ __forceinline__ void IdleForDutyWindow(const u32 activeNs, u64& now)
{
    u32 phase = static_cast<u32>(now) & (DUTY_PERIOD_NS - 1u);

    while (phase >= activeNs)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        asm volatile("nanosleep.u32 %0;" : : "r"(DUTY_PERIOD_NS - phase));
#endif
        now = globalTimer();
        phase = static_cast<u32>(now) & (DUTY_PERIOD_NS - 1u);
    }
}

// Block 0 counts the launch, before any block can leave because its SM is not active.
__device__ __forceinline__ void countLaunch(u64* counts)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        atomicAdd(&counts[3 * gridDim.x], 1ull);
    }
}

// Whether the calling block works. Uniform per block: all its threads run on the same SM.
__device__ __forceinline__ bool smActive(const u32 activeSms)
{
    return SmId() < activeSms;
}

// Adds the block's numbers to the sums over all launches of the configuration.
__device__ __forceinline__ void writeBlockStats(u64* blockCount, u64* counts, const u64 iterations, const u64 activeNs)
{
    atomicAdd(blockCount, iterations);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicAdd(&counts[blockIdx.x], *blockCount);
        atomicAdd(&counts[gridDim.x + blockIdx.x], activeNs);
    }
}

// Called by thread 0: claims the block's slot on its SM (0 .. BLOCKS_PER_SM-1) from the SM's counter, and records the SM
// id and slot in counts[2 * gridDim.x + blockIdx.x] for the host's placement check (see probeGlobal).
__device__ __forceinline__ u32 claimSlot(u32* smCounters, u64* counts)
{
    const u32 sm = SmId();
    const u32 slot = atomicAdd(&smCounters[sm % SM_COUNTER_ENTRIES], 1u) % BLOCKS_PER_SM;
    counts[2 * gridDim.x + blockIdx.x] = (static_cast<u64>(sm) << 32) | slot;
    return slot;
}

// One global access of WIDTH bytes per thread at `address`. LOAD(op) loads with the PTX instruction `op` (for example
// "ld.global.cg") into the words x..w; NO_LOAD fills the same words from the address; STORE writes the words s0..s3.
#if WIDTH == 16
#define WORD_VARIABLES u32 x, y, z, w
#define LOAD(op) asm volatile(op ".v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(address))
#define NO_LOAD() asm volatile("mov.b64 {%0,%1}, %4;\n\tmov.b64 {%2,%3}, %4;" \
    : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(address))
#define STORE() asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" : : "l"(address), "r"(s0), "r"(s1), "r"(s2), "r"(s3))
#define WORDS (x ^ y ^ z ^ w)
#elif WIDTH == 8
#define WORD_VARIABLES u32 x, y
#define LOAD(op) asm volatile(op ".v2.u32 {%0,%1}, [%2];" : "=r"(x), "=r"(y) : "l"(address))
#define NO_LOAD() asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(x), "=r"(y) : "l"(address))
#define STORE() asm volatile("st.global.v2.u32 [%0], {%1,%2};" : : "l"(address), "r"(s0), "r"(s1))
#define WORDS (x ^ y)
#else
#define WORD_VARIABLES u32 x
#define LOAD(op) asm volatile(op ".u32 %0, [%1];" : "=r"(x) : "l"(address))
#define NO_LOAD() asm volatile("{\n\t.reg .b32 high;\n\tmov.b64 {%0,high}, %1;\n\t}" : "=r"(x) : "l"(address))
#define STORE() asm volatile("st.global.u32 [%0], %1;" : : "l"(address), "r"(s0))
#define RED() asm volatile("red.global.add.u32 [%0], %1;" : : "l"(address), "r"(s0))
#define ATOM() asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(x) : "l"(address), "r"(s0))
#define WORDS (x)
#endif
