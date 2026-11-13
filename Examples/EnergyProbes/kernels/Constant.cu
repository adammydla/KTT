// Constant-memory loads (probeConstant).
//
// Parameter CONST_BYTES (bytes of constant memory read, power of two, at most 64 KiB) only names the configuration: the
// kernel gets the footprint as the argument constantBytes.

// Words of constant memory: 64 KiB, the most a module can have. Must match ConstantWords in Common.h.
#define CONSTANT_WORDS 16384

// Constant memory, filled with the random payload by the host (a symbol argument).
__constant__ u32 constantData[CONSTANT_WORDS];

// Constant-memory loads (LDC). All threads of a warp read the same word, as when they read the same coefficient, so one
// warp instruction is one request to the constant cache. The footprint constantBytes (the CONST_BYTES parameter, passed as
// an argument: as a compile-time value, ptxas folded it into the masks and shifts of every access) selects the level, the
// constant L1 of an SM or the L1.5 behind it, and the SASS is the same for every footprint. A thread's access is counted
// like the other probes', so the energy of one warp instruction is 32 accesses.
//
// As in probeGlobal, the warps of one SM must not read the lines that another warp just brought in. Every block claims its
// slot on its SM, and warp w of the block in slot s walks stripe s * BLOCK_SIZE / 32 + w of the footprint. A step of 17
// words (68 bytes) moves to another 64-byte line every time and visits every word of the stripe.
extern "C" __global__ void probeConstant(u32* __restrict__ sink, u64* __restrict__ counts, u32* __restrict__ smCounters,
    const u32 constantBytes, const u32 activeSms, const u32 idleRatio, const u64 durationNs)
{
    __shared__ u64 blockCount;
    __shared__ u32 slot;
    countLaunch(counts);

    if (threadIdx.x == 0)
    {
        blockCount = 0;
        slot = claimSlot(smCounters, counts);
    }

    __syncthreads();

    if (!smActive(activeSms))
    {
        return;
    }

    // Byte offsets: the step and the stripes keep them multiples of 4.
    constexpr u32 warpsPerBlock = BLOCK_SIZE / 32;
    const u32 stripeBytes = constantBytes >> Log2(CeilPowerOfTwo(BLOCKS_PER_SM * warpsPerBlock));
    const u32 mask = stripeBytes - 1u;
    const char* const stripe = reinterpret_cast<const char*>(constantData) + (slot * warpsPerBlock + threadIdx.x / 32)
        * stripeBytes;
    u32 offset = 0;
    u32 accumulator = 0;
    u64 iterations = 0;
    const u64 start = globalTimer();
    u64 now = start;
    u64 burstStart = start;
    u32 burstIterations = BURST_ACCESSES / UNROLL;

    do
    {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
        {
            accumulator ^= *reinterpret_cast<const u32*>(stripe + offset);
            offset = (offset + 68u) & mask;
        }

        iterations += UNROLL;
        now = globalTimer();
        IdleForDuty(idleRatio, burstIterations, burstStart, now);
    }
    while (now - start < durationNs);

    writeBlockStats(&blockCount, counts, iterations, now - start);
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = accumulator;
}
