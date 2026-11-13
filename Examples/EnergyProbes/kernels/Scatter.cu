// Access granularity and DRAM row locality (probeScatter). Parameters LOAD_OP (0 or 5) and WIDTH as in Global.cu; the
// footprint, run length and placement are runtime arguments.

// Global accesses with a chosen granularity and DRAM row locality (LOAD_OP 5 = st, any other value ld; the host uses 0 and
// 5. No #error for other values: KTT compiles the whole file for every kernel, also for probeGlobal's other LOAD_OPs).
// Every 2 threads access one 32-byte sector. The sectors of the whole GPU are numbered by one counter c, which every step
// advances by the sectors of the grid, so no two threads access the same sector in a step. Sector c of the footprint is
//     sector(c) = (((c >> runBits) * stride) << runBits | (c & (2^runBits - 1))) & sectorMask:
// runs of 2^runBits consecutive sectors, and the runs placed by an odd multiplier (a permutation of the footprint's
// sectors). stride = 1 is a stream through the footprint, coalesced as in probeGlobal (a warp instruction touches 4 lines
// of 4 sectors). A large stride puts every run in another line and DRAM row: runs of 32 bytes make every sector an L2
// request of its own (16 per warp instruction) and every DRAM access a row of its own; runs of 128 bytes and more are whole
// lines again, and only the row locality is lost. runBits, stride and sectorMask (sectors of the FOOTPRINT parameter minus
// 1: as a compile-time value, ptxas folded it into every access) are arguments, so the SASS is the same for every
// granularity, locality and footprint.
extern "C" __global__ void probeScatter(Element* __restrict__ data, u32* __restrict__ sink, u64* __restrict__ counts,
    const u32 sectorMask, const u32 runBits, const u32 stride, const u32 activeSms, const u32 idleRatio,
    const u64 durationNs)
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

    constexpr u32 threadsPerSector = 32 / WIDTH;
    constexpr u32 sectorsPerBlock = BLOCK_SIZE / threadsPerSector;
    const u32 runMask = (1u << runBits) - 1u;
    const u32 sectorsPerStep = gridDim.x * sectorsPerBlock;
    Element* const sectorData = data + threadIdx.x % threadsPerSector;
    u32 counter = blockIdx.x * sectorsPerBlock + threadIdx.x / threadsPerSector;

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
            const u32 sector = ((((counter >> runBits) * stride) << runBits) | (counter & runMask)) & sectorMask;
            Element* const address = sectorData + static_cast<u64>(sector) * threadsPerSector;
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
            counter += sectorsPerStep;
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
