// Shared-memory loads, stores and atomics (probeShared), and bank conflicts (probeSharedBanks).
//
// Parameters:
//   SMEM_BYTES  bytes of shared memory per block (power of two)
//   SHARED_OP   0 = ld.shared, 1 = st.shared, 2 = red.shared.add.u32 (32-bit atomic add; probeShared only)
//   SHARED_WIDTH  probeSharedBanks: bytes per access and thread, 4, 8 or 16
//   SHARED_GROUP  probeSharedBanks: lanes that read the same address (1 = every lane its own; 2, 16, 32 = broadcast)
//
// probeSharedBanks takes the conflict degree as the argument `ways` (the host's parameter BANK_WAYS), so that all degrees
// run the same SASS: as a compile-time constant, ptxas drops the multiplication for one way.

#ifndef SMEM_BYTES
#define SMEM_BYTES 8192
#endif
#ifndef SHARED_OP
#define SHARED_OP 0
#endif
#ifndef SHARED_WIDTH
#define SHARED_WIDTH 4
#endif
#ifndef SHARED_GROUP
#define SHARED_GROUP 1
#endif

// Shared memory accesses, conflict-free: a warp reads or writes 512 consecutive bytes per instruction.
extern "C" __global__ void probeShared(u32* __restrict__ sink, u64* __restrict__ counts, const u32 activeSms,
    const u32 idleRatio, const u32 payloadMask, const u64 durationNs)
{
    __shared__ uint4 buffer[SMEM_BYTES / 16];
    __shared__ u64 blockCount;
    countLaunch(counts);

    if (!smActive(activeSms))
    {
        return;
    }

    const u32 mask = Opaque(SMEM_BYTES / 16 - 1);

    for (u32 i = threadIdx.x; i <= mask; i += BLOCK_SIZE)
    {
        const u32 seed = (blockIdx.x * (mask + 1u) + i) * 4u;
        uint4 value;
        value.x = mix(seed) & payloadMask;
        value.y = mix(seed + 1u) & payloadMask;
        value.z = mix(seed + 2u) & payloadMask;
        value.w = mix(seed + 3u) & payloadMask;
        buffer[i] = value;
    }

    if (threadIdx.x == 0)
    {
        blockCount = 0;
    }

    __syncthreads();

    u64 sharedBase;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(sharedBase) : "l"(buffer));

#if SHARED_OP == 2
    // 32-bit atomics: one word per thread, so a warp covers 128 consecutive bytes, one word in each bank.
    const u32 wordMask = Opaque(SMEM_BYTES / 4 - 1);
    u32 index = threadIdx.x & wordMask;
#else
    u32 index = threadIdx.x & mask;
#endif
#if SHARED_OP == 1 || SHARED_OP == 2
    // Stored words (atomic operands): the same recurrence as the global stores, starting from the (random or zero) shared
    // data.
    const uint4 seed = buffer[threadIdx.x & mask];
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
#if SHARED_OP == 2
            const u64 address = sharedBase + static_cast<u64>(index) * 4ull;
            asm volatile("red.shared.add.u32 [%0], %1;" : : "l"(address), "r"(s0));
            const u32 next = s0 ^ s1;
            s0 = s1;
            s1 = s2;
            s2 = s3;
            s3 = next;
            index = (index + BLOCK_SIZE) & wordMask;
#else
            const u64 address = sharedBase + static_cast<u64>(index) * 16ull;
#if SHARED_OP == 1
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" : : "l"(address), "r"(s0), "r"(s1), "r"(s2), "r"(s3));
            const u32 next = s0 ^ s1;
            s0 = s1;
            s1 = s2;
            s2 = s3;
            s3 = next;
#else
            u32 x, y, z, w;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(address));
            accumulator ^= x ^ y ^ z ^ w;
#endif
            index = (index + BLOCK_SIZE) & mask;
#endif
        }

        iterations += UNROLL;
        now = globalTimer();
        IdleForDuty(idleRatio, burstIterations, burstStart, now);
    }
    while (now - start < durationNs);

    writeBlockStats(&blockCount, counts, iterations, now - start);
#if SHARED_OP == 1 || SHARED_OP == 2
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = s0 ^ s1 ^ s2 ^ s3;
#else
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = accumulator;
#endif
}

// Bank conflicts: loads or stores (SHARED_OP 0 or 1) of SHARED_WIDTH bytes per thread that take `ways` times the
// wavefronts of the conflict-free access. Shared memory has 32 banks of 4 bytes and a wavefront serves up to 128 bytes, so
// a conflict-free access of 4, 8 or 16 bytes takes 1, 2 or 4 wavefronts (a whole, half or quarter warp each). Lane l
// accesses element l * ways (elements of SHARED_WIDTH bytes): the lanes of a wavefront fall on 1 / ways of its banks, and
// every wavefront is replayed `ways` times, up to 32 wavefronts per instruction (ways <= 128 / SHARED_WIDTH). The
// instructions are the same for every degree. Every iteration moves all threads by BLOCK_SIZE * ways + 1 elements: the
// data changes and the banks rotate, the conflict degree stays. The host keeps SMEM_BYTES at least 32 * 32 words, so that
// the elements of a warp instruction are distinct for every degree.
// Broadcast (SHARED_GROUP > 1, the optional group "broadcast"): SHARED_GROUP consecutive lanes read the same element, so a
// warp instruction reads 32 / SHARED_GROUP distinct elements, as kernels that take one operand of a product from a row of
// shared memory (lud_internal: 2 or 16 distinct words per warp load). The wavefronts stay those of the conflict-free
// access (or fewer, where the distinct bytes fit fewer); Nsight Compute's sm__sass_data_bytes_mem_shared_op_ld counts the
// distinct bytes only. Only the start index changes; for SHARED_GROUP 1 the SASS is that of the conflict-free probe.
extern "C" __global__ void probeSharedBanks(u32* __restrict__ sink, u64* __restrict__ counts, const u32 ways,
    const u32 activeSms, const u32 idleRatio, const u32 payloadMask, const u64 durationNs)
{
    __shared__ u32 words[SMEM_BYTES / 4];
    __shared__ u64 blockCount;
    countLaunch(counts);

    if (!smActive(activeSms))
    {
        return;
    }

    const u32 mask = Opaque(SMEM_BYTES / 4 - 1);

    for (u32 i = threadIdx.x; i <= mask; i += BLOCK_SIZE)
    {
        words[i] = mix(blockIdx.x * (mask + 1u) + i) & payloadMask;
    }

    if (threadIdx.x == 0)
    {
        blockCount = 0;
    }

    __syncthreads();

    u64 sharedBase;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(sharedBase) : "l"(words));

    const u32 elementMask = mask / (SHARED_WIDTH / 4);
    u32 index = ((threadIdx.x / SHARED_GROUP) * ways) & elementMask;
    const u32 step = BLOCK_SIZE * ways + 1u;
#if SHARED_OP == 1
    // Stored words: the same recurrence as the other stores, starting from the (random or zero) shared data.
    u32 s0 = words[(4u * threadIdx.x) & mask];
    u32 s1 = words[(4u * threadIdx.x + 1u) & mask];
    u32 s2 = words[(4u * threadIdx.x + 2u) & mask];
    u32 s3 = words[(4u * threadIdx.x + 3u) & mask];
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
            const u64 address = sharedBase + static_cast<u64>(index) * static_cast<u64>(SHARED_WIDTH);
#if SHARED_OP == 1
#if SHARED_WIDTH == 16
            asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" : : "l"(address), "r"(s0), "r"(s1), "r"(s2), "r"(s3));
#elif SHARED_WIDTH == 8
            asm volatile("st.shared.v2.u32 [%0], {%1,%2};" : : "l"(address), "r"(s0), "r"(s1));
#else
            asm volatile("st.shared.u32 [%0], %1;" : : "l"(address), "r"(s0));
#endif
            const u32 next = s0 ^ s1;
            s0 = s1;
            s1 = s2;
            s2 = s3;
            s3 = next;
#else
#if SHARED_WIDTH == 16
            u32 x, y, z, w;
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(address));
            accumulator ^= x ^ y ^ z ^ w;
#elif SHARED_WIDTH == 8
            u32 x, y;
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(x), "=r"(y) : "l"(address));
            accumulator ^= x ^ y;
#else
            u32 x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "l"(address));
            accumulator ^= x;
#endif
#endif
            index = (index + step) & elementMask;
        }

        iterations += UNROLL;
        now = globalTimer();
        IdleForDuty(idleRatio, burstIterations, burstStart, now);
    }
    while (now - start < durationNs);

    writeBlockStats(&blockCount, counts, iterations, now - start);
#if SHARED_OP == 1
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = s0 ^ s1 ^ s2 ^ s3;
#else
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = accumulator;
#endif
}
