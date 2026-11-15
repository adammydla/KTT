// Loads and stores together (probeWrite): what the other probes do not do, since every one of them either reads or
// writes. In the example kernels, the configurations whose energy is mostly L2 and DRAM traffic, all of which write, came
// out about 11 % below their measurement, against 2 % for the others.
//
// Parameters:
//   WRITE_OP    0 = copy: load an element from the first half of the stripe and store one to the second half (a read
//                   stream and a write stream at once, as in a copy or a transpose),
//               1 = update: load an element and store it back to the same place (the loads bring the lines in, the stores
//                   dirty them, and L2 evicts them dirty later, as for x[i] = f(x[i])),
//               2 = local: the same in the thread's local memory (LDL, STL), as register spills do. On an RTX 3090
//                   the loads hit L1 at the L1 footprint, while the stores go on to L2 (4 L2 write sectors per warp
//                   access of 4 bytes): L1 is write-through for local memory too.
//   WIDTH       bytes per element and thread, 4, 8 or 16
//   FOOTPRINT   WRITE_OP 0 and 1: bytes of the global working set, as for probeGlobal (GRID_STRIPES, the stripes and the
//               slots as there). WRITE_OP 2: bytes of local memory walked per SM (BLOCKS_PER_SM x BLOCK_SIZE threads, each
//               walking FOOTPRINT / (WIDTH x BLOCK_SIZE x BLOCKS_PER_SM) elements of its frame), at most
//               LOCAL_FRAME_BYTES per thread.
//
// One access is one load and one store of WIDTH bytes. The accesses of a loop iteration go in groups of WRITE_GROUP: the
// group's loads first, then its stores (at the addresses the loads computed), so that a store does not wait for its own
// load. 4 elements in flight per warp are plenty for the bandwidth; 16 took over 64 registers per thread for 16-byte
// elements (and __launch_bounds__ then spilled them). Every stored element is the loaded one XOR the stored-word
// recurrence of the store probes, so that the stored bits toggle as in real data.

#ifndef WRITE_OP
#define WRITE_OP 0
#endif

#define LOCAL_FRAME_BYTES 1024
#define WRITE_GROUP 4

#if WRITE_OP == 2 && FOOTPRINT / (WIDTH * BLOCK_SIZE * BLOCKS_PER_SM) * WIDTH > LOCAL_FRAME_BYTES
#error "WRITE_OP 2: FOOTPRINT / (BLOCK_SIZE x BLOCKS_PER_SM) must be at most LOCAL_FRAME_BYTES"
#endif

// One element of WIDTH bytes into or from the words v[0 .. WORDS_PER_ELEMENT-1], with the PTX instruction `op`.
#if WIDTH == 16
#define LOAD_ELEMENT(op, address, v) asm volatile(op ".v4.u32 {%0,%1,%2,%3}, [%4];" \
    : "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]) : "l"(address))
#define STORE_ELEMENT(op, address, v) asm volatile(op ".v4.u32 [%0], {%1,%2,%3,%4};" \
    : : "l"(address), "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]))
#elif WIDTH == 8
#define LOAD_ELEMENT(op, address, v) asm volatile(op ".v2.u32 {%0,%1}, [%2];" : "=r"(v[0]), "=r"(v[1]) : "l"(address))
#define STORE_ELEMENT(op, address, v) asm volatile(op ".v2.u32 [%0], {%1,%2};" : : "l"(address), "r"(v[0]), "r"(v[1]))
#else
#define LOAD_ELEMENT(op, address, v) asm volatile(op ".u32 %0, [%1];" : "=r"(v[0]) : "l"(address))
#define STORE_ELEMENT(op, address, v) asm volatile(op ".u32 [%0], %1;" : : "l"(address), "r"(v[0]))
#endif

#if WRITE_OP == 2
#define LOAD_OP_NAME "ld.local"
#define STORE_OP_NAME "st.local"
#else
#define LOAD_OP_NAME "ld.global"
#define STORE_OP_NAME "st.global"
#endif

// Stores `value` to `slot` and loads it back, both volatile (as in Compute.cu): a register whose value ptxas cannot know.
__device__ __forceinline__ u32 ThroughMemory(u32* slot, const u32 value)
{
    u32 result;
    asm volatile("st.volatile.global.u32 [%1], %2;\n\tld.volatile.global.u32 %0, [%1];" : "=r"(result) : "l"(slot), "r"(value)
        : "memory");
    return result;
}

// The arguments are those of probeGlobal; WRITE_OP 2 does not use data (except for the seed) and gridSplit.
extern "C" __global__ void probeWrite(Element* __restrict__ data, u32* __restrict__ sink, u64* __restrict__ counts,
    u32* __restrict__ smCounters, const u32 gridSplit, const u32 activeSms, const u32 idleRatio, const u64 durationNs)
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

    // The stored-word recurrence of the store probes, s[n+4] = s[n] ^ s[n+1], from the payload.
    const uint4 seed = reinterpret_cast<const uint4*>(data)[blockIdx.x * BLOCK_SIZE + threadIdx.x];
    u32 s0 = seed.x;
    u32 s1 = seed.y;
    u32 s2 = seed.z;
    u32 s3 = seed.w;

#if WRITE_OP == 2
    // The thread's frame in local memory, walked element by element. Its size is fixed, so that the SASS is the same for
    // every footprint. How much of it is walked comes from FOOTPRINT through memory (ThroughMemory), because ptxas saw
    // through Opaque and put the mask into every address as an immediate. The frame starts with recurrence words.
    Element frame[LOCAL_FRAME_BYTES / WIDTH];
    const u64 frameBase = __cvta_generic_to_local(frame);
    const u32 mask = ThroughMemory(&sink[blockIdx.x * BLOCK_SIZE + threadIdx.x],
        static_cast<u32>(FOOTPRINT / (WIDTH * BLOCK_SIZE * BLOCKS_PER_SM)) - 1u);
    u32 index = 0;

    #pragma unroll 1
    for (u32 e = 0; e <= mask; ++e)
    {
        const u32 init[4] = {s0 ^ e, s1, s2, s3};
        const u64 address = frameBase + static_cast<u64>(e) * WIDTH;
        STORE_ELEMENT(STORE_OP_NAME, address, init);
    }
#else
    const u32 stripeBits = gridSplit != 0u ? Log2(GRID_STRIPES) : Log2(CeilPowerOfTwo(BLOCKS_PER_SM));
    const u32 stripeElements = Opaque(static_cast<u32>(static_cast<u64>(FOOTPRINT) / WIDTH)) >> stripeBits;
    const u32 stripe = gridSplit != 0u ? blockIdx.x : slot;
    Element* const stripeData = OpaquePointer(data + static_cast<u64>(stripe) * stripeElements);
#if WRITE_OP == 0
    // Loads walk the first half of the stripe, stores the second.
    const u32 mask = stripeElements / 2u - 1u;
    Element* const storeData = OpaquePointer(stripeData + stripeElements / 2u);
#else
    const u32 mask = stripeElements - 1u;
    Element* const storeData = stripeData;
#endif
    // Start rows as in probeGlobal (Fibonacci hashing of the block index).
    const u32 rows = (mask + 1u) / BLOCK_SIZE;
    const u32 rowBits = 31u - static_cast<u32>(__clz(rows));
    const u32 startRow = rowBits == 0 ? 0u : (blockIdx.x * 2654435769u) >> (32u - rowBits);
    u32 index = (startRow * BLOCK_SIZE + threadIdx.x) & mask;
#endif

    u32 accumulator = 0;
    u64 iterations = 0;
    const u64 start = globalTimer();
    u64 now = start;
    u64 burstStart = start;
    u32 burstIterations = BURST_ACCESSES / UNROLL;

    do
    {
        #pragma unroll
        for (int g = 0; g < UNROLL / WRITE_GROUP; ++g)
        {
            u32 words[WRITE_GROUP][4];
            u32 offsets[WRITE_GROUP];

            #pragma unroll
            for (int u = 0; u < WRITE_GROUP; ++u)
            {
                offsets[u] = index;
#if WRITE_OP == 2
                const u64 address = frameBase + static_cast<u64>(index) * WIDTH;
                index = (index + 1u) & mask;
#else
                const Element* const address = stripeData + index;
                index = (index + BLOCK_SIZE) & mask;
#endif
                LOAD_ELEMENT(LOAD_OP_NAME, address, words[u]);
            }

            #pragma unroll
            for (int u = 0; u < WRITE_GROUP; ++u)
            {
#if WRITE_OP == 2
                const u64 address = frameBase + static_cast<u64>(offsets[u]) * WIDTH;
#else
                Element* const address = storeData + offsets[u];
#endif
                words[u][0] ^= s0;
#if WIDTH >= 8
                words[u][1] ^= s1;
#endif
#if WIDTH == 16
                words[u][2] ^= s2;
                words[u][3] ^= s3;
#endif
                STORE_ELEMENT(STORE_OP_NAME, address, words[u]);
                accumulator ^= words[u][0];
                const u32 next = s0 ^ s1;
                s0 = s1;
                s1 = s2;
                s2 = s3;
                s3 = next;
            }
        }

        // Whole groups only: with an UNROLL that is not a multiple of WRITE_GROUP the loop above issues fewer accesses than
        // UNROLL (the host refuses such an --unroll).
        iterations += (UNROLL / WRITE_GROUP) * WRITE_GROUP;
        now = globalTimer();
        IdleForDuty(idleRatio, burstIterations, burstStart, now);
    }
    while (now - start < durationNs);

    writeBlockStats(&blockCount, counts, iterations, now - start);
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = accumulator ^ s0 ^ s1 ^ s2 ^ s3;
}
