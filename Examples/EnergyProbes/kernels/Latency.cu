// Dependent loads at low memory-level parallelism (probeLatency).
//
// Parameters: FOOTPRINT (as for probeGlobal; the line mask is an argument, so every footprint runs the same SASS), CHAINS
// (independent chains per warp). Runtime arguments: activeWarps (the warps of a block that work; the others only count),
// lineMask (lines of the footprint - 1), zero (always 0).
//
// The other probes keep many loads in flight, so they say what a load costs when bandwidth limits. In real kernels it is
// often latency: in the Reduction example, the configuration with one block per SM runs 4x longer than the others on the
// same bytes and costs 18 % more than the model predicts. Here every working warp follows CHAINS chains of pseudo-random
// lines of the footprint. A step loads one 4-byte word per lane from the chain's current line (one coalesced 128-byte
// request) and moves to the next line by an LCG whose state also adds the loaded word AND zero: the next address depends
// on the data of the load, so a chain has exactly one load in flight, and a warp CHAINS. With few working warps per SM and
// few chains the probe is latency-bound, and with many it approaches the bandwidth of the plain loads. One access is one
// load of one thread.

#ifndef CHAINS
#define CHAINS 1
#endif

// ROWS 1 (the rows group, 2026-11-17): instead of chains, every thread walks a row of its own, one 4-byte word per step,
// as a kernel with one thread per matrix row does (PolyBench gesummv, atax): a warp instruction touches 32 lines, one
// word each, and a thread's next 7 words hit the sector in L1. The rows are footprint / threads apart (rounded down to a
// power of two), so with the DRAM footprint every row is in a DRAM page of its own. The loads of a thread are independent
// (UNROLL of them in flight), so the rate grows with the working warps. ROWS 0 compiles to the measured chains.
#ifndef ROWS
#define ROWS 0
#endif

// ptxas for sm_75 gives the 16-chain variant 72 registers per thread, so 3 of 4 blocks fit per SM (the RTX 2080 Ti,
// 2026-11-16). There __launch_bounds__ holds it to 64 without spilling. The other variants and architectures need 32-64
// registers and keep their SASS (bounds would change it, on sm_86 too).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750 && CHAINS > 4
#define LATENCY_BOUNDS __launch_bounds__(BLOCK_SIZE, BLOCKS_PER_SM)
#else
#define LATENCY_BOUNDS
#endif

extern "C" __global__ void LATENCY_BOUNDS probeLatency(const u32* __restrict__ data, u32* __restrict__ sink,
    u64* __restrict__ counts, const u32 activeSms, const u32 activeWarps, const u32 lineMask, const u32 zero,
    const u32 idleRatio, const u64 durationNs)
{
    __shared__ u64 blockCount;
    countLaunch(counts);

    if (!smActive(activeSms))
    {
        return;
    }

    if (threadIdx.x == 0)
    {
        blockCount = 0;
    }

    __syncthreads();

    const u32 lane = threadIdx.x % 32;
    const u32 warp = blockIdx.x * (BLOCK_SIZE / 32) + threadIdx.x / 32;
    // The chains' states, the same in every lane of a warp (the zero term keeps them so).
    u32 state[CHAINS];

    for (int c = 0; c < CHAINS; ++c)
    {
        state[c] = mix(warp * CHAINS + c + 1);
    }

    u32 accumulator = 0;
#if ROWS
    const u32 threads = gridDim.x * BLOCK_SIZE;
    const u32 words = (lineMask + 1u) * 32u;
    const u32 rowWords = 1u << (31 - __clz(max(words / threads, 1u)));
    const u32* const row = data + static_cast<u64>(blockIdx.x * BLOCK_SIZE + threadIdx.x) * rowWords;
    const u32 rowMask = Opaque(rowWords - 1u);
    u32 column = 0;
#endif
    u64 iterations = 0;
    const u64 start = globalTimer();
    u64 now = start;
    u64 burstStart = start;
    u32 burstIterations = BURST_ACCESSES / UNROLL;

    if (threadIdx.x / 32 < activeWarps)
    {
        do
        {
#if ROWS
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u)
            {
                u32 word;
                asm volatile("ld.global.u32 %0, [%1];" : "=r"(word) : "l"(row + ((column + u) & rowMask)));
                accumulator ^= word;
            }

            column += UNROLL;
            iterations += UNROLL;
#else
            #pragma unroll
            for (int u = 0; u < UNROLL; ++u)
            {
                #pragma unroll
                for (int c = 0; c < CHAINS; ++c)
                {
                    // Bits 6 and up of the LCG state pick the line (its low bits have short periods).
                    const u32* const address = data + static_cast<u64>((state[c] >> 6) & lineMask) * 32 + lane;
                    u32 word;
                    asm volatile("ld.global.u32 %0, [%1];" : "=r"(word) : "l"(address));
                    accumulator ^= word;
                    state[c] = state[c] * 1664525u + 1013904223u + (word & zero);
                }
            }

            iterations += UNROLL * CHAINS;
#endif
            now = globalTimer();
            IdleForDuty(idleRatio, burstIterations, burstStart, now);
        }
        while (now - start < durationNs);
    }

    writeBlockStats(&blockCount, counts, iterations, now - start);
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = accumulator;
}
