// Texture fetches (probeTexture): the footprints, stripes and slots of probeGlobal, through the texture path. Parameters
// FOOTPRINT and GRID_STRIPES as in Global.cu; the texture object over the payload is a runtime argument.

// Texture fetches: tex.1d from a texture object over linear memory, the texture path of L1TEX (which addresses, filters and
// formats every fetch) instead of the load path. The host creates the object with the driver API over a copy of the
// payload (or zeros): 4 x 32-bit integer texels, 16 bytes like the default load, no filtering, read as integers. FOOTPRINT,
// the stripes and the slots are those of probeGlobal (see there), so the footprints select the same levels; texel
// indices replace the addresses.
extern "C" __global__ void probeTexture(const u64 texture, u32* __restrict__ sink, u64* __restrict__ counts,
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
    const u32 stripeTexels = Opaque(static_cast<u32>(static_cast<u64>(FOOTPRINT) / 16)) >> stripeBits;
    const u32 stripe = gridSplit != 0u ? blockIdx.x : slot;
    const u32 firstTexel = stripe * stripeTexels;
    const u32 mask = stripeTexels - 1u;

    const u32 rows = (mask + 1u) / BLOCK_SIZE;
    const u32 rowBits = 31u - static_cast<u32>(__clz(rows));
    const u32 startRow = rowBits == 0 ? 0u : (blockIdx.x * 2654435769u) >> (32u - rowBits);
    u32 index = (startRow * BLOCK_SIZE + threadIdx.x) & mask;

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
            u32 x, y, z, w;
            asm volatile("tex.1d.v4.u32.s32 {%0,%1,%2,%3}, [%4, {%5}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "l"(texture), "r"(firstTexel + index));
            accumulator ^= x ^ y ^ z ^ w;
            index = (index + BLOCK_SIZE) & mask;
        }

        iterations += UNROLL;
        now = globalTimer();
        IdleForDuty(idleRatio, burstIterations, burstStart, now);
    }
    while (now - start < durationNs);

    writeBlockStats(&blockCount, counts, iterations, now - start);
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = accumulator;
}
