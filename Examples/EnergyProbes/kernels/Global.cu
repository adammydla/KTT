// Global loads, stores and atomics (probeGlobal).
//
// Parameters:
//   LOAD_OP     0 = ld.global (default caching, as in normal code), 1 = ld.global.cg (L1 bypassed),
//               2 = no memory access (same loop and address arithmetic; instruction-overhead reference),
//               3 = ld.global.ca (explicit L1 caching, as used by MT4G), 4 = ld.global.nc (read-only data path),
//               5 = st.global (stores; L1 is write-through for global stores, so L2 or DRAM serves them),
//               6 = red.global.add.u32 (atomic add without result, as atomicAdd whose result is unused; WIDTH 4),
//               7 = atom.global.add.u32 (atomic add that returns the old value; WIDTH 4). Global atomics are performed in
//               L2, so like stores they start at the L2 footprint.
//   WIDTH       bytes per access and thread, 4, 8 or 16
//   FOOTPRINT   bytes of the global working set (power of two)
//   GRID_STRIPES   power of two >= number of blocks; stripes of a footprint split over the whole grid
//   MIX_FFMA, MIX_L1, MIX_SHARED   (the optional groups "mix" and "saturate", 2026-11-18; LOAD_OP 0 only, default 0 = none,
//               and then the code is the plain probe's) work added to every access: MIX_FFMA FFMAs with three registers
//               (the recurrence of the compute probe's COMPUTE_OP 9, four copies, so a multiple of 4), MIX_L1 more loads of
//               WIDTH bytes from two rows of the block's own stripe (L1 hits after the first pass), MIX_SHARED conflict-free
//               16-byte shared loads from two rows of the block's shared array. They test whether the energies of mixed
//               work add up as the model adds them: FFMA next to loads, and memory-instruction work while DRAM or L2 is
//               the bottleneck (Bicg on the random sweep looked as if that work were free).

#ifndef MIX_FFMA
#define MIX_FFMA 0
#endif
#ifndef MIX_L1
#define MIX_L1 0
#endif
#ifndef MIX_SHARED
#define MIX_SHARED 0
#endif
#define MIX_ANY (MIX_FFMA > 0 || MIX_L1 > 0 || MIX_SHARED > 0)
#if MIX_ANY && LOAD_OP != 0
#error "MIX_FFMA, MIX_L1 and MIX_SHARED go with LOAD_OP 0"
#endif
#if MIX_FFMA % 4 != 0
#error "MIX_FFMA must be a multiple of 4"
#endif

#if (LOAD_OP == 6 || LOAD_OP == 7) && WIDTH != 4
#error "The atomics (LOAD_OP 6 and 7) are 32-bit: WIDTH must be 4"
#endif

// Operations that write words (stores, atomics) and operations that return them (loads, atom).
#define WRITES_WORDS (LOAD_OP == 5 || LOAD_OP == 6 || LOAD_OP == 7)
#define RETURNS_WORDS (LOAD_OP != 5 && LOAD_OP != 6)

// Global memory accesses. The cache level is selected by FOOTPRINT (and LOAD_OP): the SASS is the same for every footprint,
// only immediates change.
//
// No two blocks on one SM access the same lines. When they did (every block walked the whole footprint), a block trailing
// another one hit L1 on the lines the leader had brought in, sped up and stayed behind it: such convoys turned about 3 of
// 4 accesses of the L2 probe into L1 hits. Thread 0 claims the block's slot on its SM (0 .. BLOCKS_PER_SM-1) from a
// counter per SM, and the footprint is split into stripes:
//   - gridSplit = 0: the block in slot s of every SM walks stripe s. The blocks of one SM together cover the whole
//     footprint (per-SM working set = FOOTPRINT), and all SMs share it in L2.
//   - gridSplit = 1 (DRAM footprints): block b walks stripe b of GRID_STRIPES, so no two blocks on the GPU share data and
//     blocks on different SMs cannot convoy in L2 either. Only gridDim.x of the stripes are accessed.
// gridSplit is an argument rather than a parameter because ptxas folds compile-time choices, which changed the SASS of the
// split footprints. The counter is not reset between launches. Consecutive claims are distinct modulo BLOCKS_PER_SM, so the
// slots of one launch are distinct as long as no SM gets more than BLOCKS_PER_SM blocks; the host checks this from the SM
// ids and slots written to counts[2 * gridDim.x + blockIdx.x].
#if MIX_ANY
// As in Compute.cu: stores `value` to `slot` and loads it back (volatile), so ptxas cannot know the value: the FFMA copies
// stay separate, and the extra loads' addresses (two rows, read over and over) cannot be proven equal and merged.
__device__ __forceinline__ u32 MixThroughMemory(u32* slot, const u32 value)
{
    u32 result;
    asm volatile("st.volatile.global.u32 [%1], %2;\n\tld.volatile.global.u32 %0, [%1];" : "=r"(result) : "l"(slot), "r"(value)
        : "memory");
    return result;
}
#endif

extern "C" __global__ void probeGlobal(Element* __restrict__ data, u32* __restrict__ sink, u64* __restrict__ counts,
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

    const u32 stripeBits = gridSplit != 0u ? Log2(GRID_STRIPES) : Log2(CeilPowerOfTwo(BLOCKS_PER_SM));
    const u32 stripeElements = Opaque(static_cast<u32>(static_cast<u64>(FOOTPRINT) / WIDTH)) >> stripeBits;
    const u32 stripe = gridSplit != 0u ? blockIdx.x : slot;
    // Without OpaquePointer the compiler adds the stripe offset in every access (two extra instructions per load, which
    // raised the cost of the no-load reference by 40 % compared to walking the whole footprint).
    Element* const stripeData = OpaquePointer(data + static_cast<u64>(stripe) * stripeElements);
    const u32 mask = stripeElements - 1u;

    // Each block walks its stripe, one row of BLOCK_SIZE elements per step. Start rows follow the golden-ratio sequence
    // (Fibonacci hashing of the block index), so that blocks of different SMs on the same stripe start far apart.
    const u32 rows = (mask + 1u) / BLOCK_SIZE;
    const u32 rowBits = 31u - static_cast<u32>(__clz(rows));
    const u32 startRow = rowBits == 0 ? 0u : (blockIdx.x * 2654435769u) >> (32u - rowBits);
    u32 index = (startRow * BLOCK_SIZE + threadIdx.x) & mask;

#if WRITES_WORDS
    // Stored words (and the atomics' operands): the XOR recurrence s[n+4] = s[n] ^ s[n+1] (primitive over GF(2), period 15
    // per bit), one instruction per store like the XOR of the loads. It starts from the payload.
    const uint4 seed = reinterpret_cast<const uint4*>(data)[blockIdx.x * BLOCK_SIZE + threadIdx.x];
    u32 s0 = seed.x;
    u32 s1 = seed.y;
    u32 s2 = seed.z;
    u32 s3 = seed.w;
#endif
#if RETURNS_WORDS
    u32 accumulator = 0;
#endif
#if MIX_ANY
    u32* const mixSlot = &sink[blockIdx.x * BLOCK_SIZE + threadIdx.x];
#endif
#if MIX_L1 > 0 || MIX_SHARED > 0
    const u32 mixStep = MixThroughMemory(mixSlot, BLOCK_SIZE);
    const u32 mixMask = MixThroughMemory(mixSlot, 2u * BLOCK_SIZE - 1u);
#endif
#if MIX_FFMA > 0
    // Four copies of one chain x -> x * x + c, updated as a = a*b + c0, b = c*d + c1, c = a*c + c2, d = b*d + c3 (three
    // different registers per FFMA), c = -1.9 from a start in [1, 2): the compute probe's COMPUTE_OP 9.
    const u32 mixInitial = 0x3f800000u | (mix((blockIdx.x * BLOCK_SIZE + threadIdx.x) * 16u) >> 10);
    float copies[4];
    float constants[4];

    for (int i = 0; i < 4; ++i)
    {
        copies[i] = __uint_as_float(MixThroughMemory(mixSlot, mixInitial));
        constants[i] = __uint_as_float(MixThroughMemory(mixSlot, 0xbff33333u));
    }
#endif
#if MIX_L1 > 0
    // Two rows of the block's stripe, read over and over: L1 hits once loaded.
    u32 l1Index = threadIdx.x;
#endif
#if MIX_SHARED > 0
    __shared__ uint4 mixShared[2 * BLOCK_SIZE];

    for (u32 i = threadIdx.x; i < 2 * BLOCK_SIZE; i += BLOCK_SIZE)
    {
        mixShared[i] = make_uint4(mix(4 * i), mix(4 * i + 1), mix(4 * i + 2), mix(4 * i + 3));
    }

    __syncthreads();
    const u32 sharedBase = static_cast<u32>(__cvta_generic_to_shared(mixShared));
    u32 sharedIndex = threadIdx.x;
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
            Element* const address = stripeData + index;

#if LOAD_OP == 5
            STORE();
#elif LOAD_OP == 6
            RED();
#elif LOAD_OP == 7
            WORD_VARIABLES;
            ATOM();
            accumulator ^= WORDS;
#else
            WORD_VARIABLES;
#if LOAD_OP == 0
            LOAD("ld.global");
#elif LOAD_OP == 1
            LOAD("ld.global.cg");
#elif LOAD_OP == 2
            NO_LOAD(); // same registers produced from the address, but no memory access
#elif LOAD_OP == 3
            LOAD("ld.global.ca");
#elif LOAD_OP == 4
            LOAD("ld.global.nc");
#else
#error "Unsupported LOAD_OP"
#endif
            accumulator ^= WORDS;
#endif
#if MIX_L1 > 0
            #pragma unroll
            for (int e = 0; e < MIX_L1; ++e)
            {
                Element* const address = stripeData + l1Index;
                WORD_VARIABLES;
                LOAD("ld.global");
                accumulator ^= WORDS;
                l1Index = (l1Index + mixStep) & mixMask;
            }
#endif
#if MIX_SHARED > 0
            #pragma unroll
            for (int e = 0; e < MIX_SHARED; ++e)
            {
                u32 sx, sy, sz, sw;
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(sx), "=r"(sy), "=r"(sz), "=r"(sw)
                    : "r"(sharedBase + 16u * sharedIndex));
                accumulator ^= sx ^ sy ^ sz ^ sw;
                sharedIndex = (sharedIndex + mixStep) & mixMask;
            }
#endif
#if MIX_FFMA > 0
            #pragma unroll
            for (int f = 0; f < MIX_FFMA / 4; ++f)
            {
                const float a = copies[0];
                const float b = copies[1];
                const float c = copies[2];
                const float d = copies[3];
                copies[0] = fmaf(a, b, constants[0]);
                copies[1] = fmaf(c, d, constants[1]);
                copies[2] = fmaf(a, c, constants[2]);
                copies[3] = fmaf(b, d, constants[3]);
            }
#endif
#if WRITES_WORDS
            const u32 next = s0 ^ s1;
            s0 = s1;
            s1 = s2;
            s2 = s3;
            s3 = next;
#endif
            index = (index + BLOCK_SIZE) & mask;
        }

        iterations += UNROLL;
        now = globalTimer();
        IdleForDuty(idleRatio, burstIterations, burstStart, now);
    }
    while (now - start < durationNs);

    writeBlockStats(&blockCount, counts, iterations, now - start);
#if WRITES_WORDS && RETURNS_WORDS
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = accumulator ^ s0 ^ s1 ^ s2 ^ s3;
#elif WRITES_WORDS
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = s0 ^ s1 ^ s2 ^ s3;
#elif MIX_FFMA > 0
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = accumulator ^ __float_as_uint(copies[0]) ^ __float_as_uint(copies[1])
        ^ __float_as_uint(copies[2]) ^ __float_as_uint(copies[3]);
#else
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = accumulator;
#endif
}
