// L1 sweep (probeL1Sweep, --l1-sweep, NEXT_STEPS_PLAN M2, 2026-11-14): loads and stores at the L1 footprint whose warp
// instructions touch SWEEP_SECTORS sectors (1 = all lanes read the same sector; 32 = every lane a sector of its own) in
// SWEEP_LINES 128-byte lines (1, 4 or 32). In the default probes "inst lsu global", "L1 sector", the L2 request and
// "inst alu" occur in fixed proportions, so the fit can trade their prices; here the sectors and lines per instruction
// move against the instructions. Parameters LOAD_OP (0 = ld, 5 = st), WIDTH (4 or 16) as in Global.cu.
//
// Lane l reads sector s = l / (32 / SWEEP_SECTORS) of its warp's group, in line s % SWEEP_LINES at sector (s / SWEEP_LINES)
// % 4, and in it the word (l % (32 / SWEEP_SECTORS)) % (32 / WIDTH) (lanes beyond a sector's words share words). The
// warps of the GPU are numbered by a counter that advances by the warps of the grid each step; warp group c starts at line
// (c * SWEEP_LINES) & lineMask. lineMask (lines of the footprint minus 1) is the argument sectorMask of the other probes.

#ifndef SWEEP_SECTORS
#define SWEEP_SECTORS 32
#endif
#ifndef SWEEP_LINES
#define SWEEP_LINES 4
#endif

extern "C" __global__ void probeL1Sweep(Element* __restrict__ data, u32* __restrict__ sink, u64* __restrict__ counts,
    const u32 lineMask, const u32 activeSms, const u32 idleRatio, const u64 durationNs)
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

    constexpr u32 wordsPerSector = 32 / WIDTH;
    constexpr u32 wordsPerLine = 128 / WIDTH;
    constexpr u32 lanesPerSector = 32 / SWEEP_SECTORS;
    const u32 lane = threadIdx.x & 31u;
    const u32 sector = lane / lanesPerSector;
    const u32 laneOffset = (sector % SWEEP_LINES) * wordsPerLine + ((sector / SWEEP_LINES) % 4u) * wordsPerSector
        + (lane % lanesPerSector) % wordsPerSector;
    const u32 warps = gridDim.x * (BLOCK_SIZE / 32);
    u32 counter = blockIdx.x * (BLOCK_SIZE / 32) + threadIdx.x / 32u;

#if LOAD_OP == 5
    const uint4 seed = reinterpret_cast<const uint4*>(data)[blockIdx.x * BLOCK_SIZE + threadIdx.x];
    u32 s0 = seed.x;
    u32 s1 = seed.y;
    u32 s2 = seed.z;
    u32 s3 = seed.w;
#else
    u32 accumulator = 0;
#endif
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
            Element* const address = data + static_cast<u64>((counter * SWEEP_LINES) & lineMask) * wordsPerLine + laneOffset;
            counter += warps;
#if LOAD_OP == 5
            STORE();
            const u32 next = s0 ^ s1;
            s0 = s1;
            s1 = s2;
            s2 = s3;
            s3 = next;
#else
            WORD_VARIABLES;
            LOAD("ld.global");
            accumulator ^= WORDS;
#endif
        }

        iterations += UNROLL;
        now = globalTimer();
        IdleForDuty(idleRatio, burstIterations, burstStart, now);
    }
    while (now - start < durationNs);

    writeBlockStats(&blockCount, counts, iterations, now - start);
#if LOAD_OP == 5
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = s0 ^ s1 ^ s2 ^ s3;
#else
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = accumulator;
#endif
}
