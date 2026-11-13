// Mixed intensity (probeMixed): the plain loads of probeGlobal with FFMAs between them, from memory-bound to compute-bound,
// as Mixbench sweeps arithmetic intensity in Antepara et al. (SC '25). The question is whether the energy of such a kernel
// is the sum of the energies of its loads and its FFMAs, also where both units are busy at once.
//
// Parameters: WIDTH, FOOTPRINT and GRID_STRIPES as in probeGlobal (Global.cu). MIX_FMAS (FFMAs per load) only names the
// configuration: the host passes the count as the argument fmaGroups, so that the SASS is the same for every intensity.
//
// Every UNROLL loads (the loop of probeGlobal with LOAD_OP 0: same stripes, slots, timer checks and bursts) are followed by
// fmaGroups groups of MIX_GROUP FFMAs in MIX_CHAINS independent chains (x -> x * x + c with c = -1.9, as in probeCompute),
// so a load is followed by fmaGroups * MIX_GROUP / UNROLL FFMAs on average. The FFMAs do not depend on the loads: with 32
// warps per SM, the warps in their load phase and the warps in their FFMA phase keep both units busy, as in a kernel that
// loads a tile and computes on it. Counted per load (one access); random data only.

// Independent FFMA chains per thread, and FFMAs per group (16 per chain, one pass of the unrolled loop). Must match
// MixedFmaGroup in Common.h.
#define MIX_CHAINS 4
#define MIX_GROUP 64

extern "C" __global__ void probeMixed(Element* __restrict__ data, u32* __restrict__ sink, u64* __restrict__ counts,
    u32* __restrict__ smCounters, const u32 gridSplit, const u32 activeSms, const u32 idleRatio, const u32 operandC,
    const u32 fmaGroups, const u64 durationNs)
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

    // The stripes of probeGlobal.
    const u32 stripeBits = gridSplit != 0u ? Log2(GRID_STRIPES) : Log2(CeilPowerOfTwo(BLOCKS_PER_SM));
    const u32 stripeElements = Opaque(static_cast<u32>(static_cast<u64>(FOOTPRINT) / WIDTH)) >> stripeBits;
    const u32 stripe = gridSplit != 0u ? blockIdx.x : slot;
    Element* const stripeData = OpaquePointer(data + static_cast<u64>(stripe) * stripeElements);
    const u32 mask = stripeElements - 1u;
    const u32 rows = (mask + 1u) / BLOCK_SIZE;
    const u32 rowBits = 31u - static_cast<u32>(__clz(rows));
    const u32 startRow = rowBits == 0 ? 0u : (blockIdx.x * 2654435769u) >> (32u - rowBits);
    u32 index = (startRow * BLOCK_SIZE + threadIdx.x) & mask;

    const u32 seed = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 16u;
    const float c = __uint_as_float(operandC);
    float value[MIX_CHAINS];

    for (int i = 0; i < MIX_CHAINS; ++i)
    {
        value[i] = __uint_as_float(0x3f800000u | (mix(seed + i) >> 10));
    }

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
            Element* const address = stripeData + index;
            WORD_VARIABLES;
            LOAD("ld.global");
            accumulator ^= WORDS;
            index = (index + BLOCK_SIZE) & mask;
        }

        #pragma unroll 1
        for (u32 group = 0; group < fmaGroups; ++group)
        {
            #pragma unroll
            for (int step = 0; step < MIX_GROUP / MIX_CHAINS; ++step)
            {
                #pragma unroll
                for (int i = 0; i < MIX_CHAINS; ++i)
                {
                    value[i] = fmaf(value[i], value[i], c);
                }
            }
        }

        iterations += UNROLL;
        now = globalTimer();
        IdleForDuty(idleRatio, burstIterations, burstStart, now);
    }
    while (now - start < durationNs);

    writeBlockStats(&blockCount, counts, iterations, now - start);
    u32 result = accumulator;

    for (int i = 0; i < MIX_CHAINS; ++i)
    {
        result ^= __float_as_uint(value[i]);
    }

    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = result;
}
