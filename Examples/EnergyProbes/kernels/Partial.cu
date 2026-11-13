// Sub-sector accesses (probePartial). Parameters LOAD_OP (0 or 5) and WIDTH (4 or 8) as in Global.cu; the footprint and
// the placement of the sectors are runtime arguments.

// Sub-sector accesses (LOAD_OP 5 = st, any other value ld; WIDTH 4 or 8 bytes): every thread walks a 32-byte sector of
// its own, one WIDTH-byte word per step, and moves to its next sector after the last word. A warp instruction thus
// touches 32 sectors, each with one word, as when a transpose writes (or reads) a column: L2 receives a partial write of
// every sector and merges the 32 / WIDTH writes of a sector before it writes the sector to DRAM; a load brings the
// sector into L1 with its first word, and the thread's next words hit there. The sectors of the whole GPU are numbered
// by thread (counter c, advanced by the threads of the grid after every sector) and placed at (c * stride) & sectorMask:
// stride = 1 puts the sectors of a warp next to each other (8 lines, 4 sectors per L2 request), a large odd stride
// scatters them (32 lines, one sector per request). stride and sectorMask are arguments, so the SASS is the same for both
// and for every footprint.
extern "C" __global__ void probePartial(Element* __restrict__ data, u32* __restrict__ sink, u64* __restrict__ counts,
    const u32 sectorMask, const u32 stride, const u32 activeSms, const u32 idleRatio, const u64 durationNs)
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

    // Whole sectors per loop iteration (no static_assert: KTT compiles this file for every kernel and UNROLL).
    constexpr u32 wordsPerSector = 32 / WIDTH;
    constexpr u32 sectorsPerIteration = UNROLL >= wordsPerSector ? UNROLL / wordsPerSector : 1;
    const u32 threads = gridDim.x * BLOCK_SIZE;
    u32 counter = blockIdx.x * BLOCK_SIZE + threadIdx.x;

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
        for (int v = 0; v < static_cast<int>(sectorsPerIteration); ++v)
        {
            Element* const sectorData = data + static_cast<u64>((counter * stride) & sectorMask) * wordsPerSector;
            counter += threads;

            #pragma unroll
            for (int u = 0; u < static_cast<int>(wordsPerSector); ++u)
            {
                Element* const address = sectorData + u;
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
        }

        iterations += sectorsPerIteration * wordsPerSector;
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
