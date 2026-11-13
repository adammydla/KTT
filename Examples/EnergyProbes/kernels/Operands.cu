// Arithmetic on operands with controlled bits (probeOperands, --payload-bits): how the energy of an FFMA, a DFMA and an
// HMMA depends on the data, over the fraction of one bits and the fraction of bits that toggle between consecutive
// operands. The arithmetic probes of Compute.cu follow chaotic recurrences, whose bits are random (the upper end), or take
// a zero payload (the lower end); here the operands come from the payloads of --payload-bits.
//
// Parameter OPERAND_OP: 0 = FFMA, 1 = DFMA, 2 = HMMA m16n8k16 (f16 inputs, f32 accumulators; sm_80 or newer).
//
// Every thread reads two sets (s = 0, 1) of operands per chain from the payload and computes d = a_s * b_s + c_s with the
// two sets in turn, so that consecutive instructions of a chain differ in the bits in which the two sets differ. The two
// sets sit at even and odd words of the payload: a payload that alternates two words (flip:T) makes them differ in a
// fraction T of their bits, one that repeats a word (same:Q) makes them equal. The exponents are fixed so that every
// operand is finite and normal: floats in +-[1, 2) (sign and mantissa from the payload), doubles likewise (sign and the top
// 20 mantissa bits from one word, the low 32 bits from another), halves in +-[1, 2). The host computes the statistics of
// these operand words (OperandStatistics in ProbeKernels.cpp, <prefix>_payloads.json).
//
// With fixed operands, every second result of a chain is the same, and ptxas merged them (8 of 64 FFMAs were left when the
// results were XORed into an accumulator). So the first operand of every operation is a_s XOR (previous result AND zero),
// with `zero` a kernel argument that is 0 at run time: ptxas must compute every operation in turn, the operand still has
// the payload's bits, and the cost is one LOP3 per operation for every payload. The duty is the global-timer window of
// Compute.cu (IdleForDutyWindow).

#ifndef OPERAND_OP
#define OPERAND_OP 0
#endif

#if OPERAND_OP == 2 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "OPERAND_OP 2 (HMMA m16n8k16) needs sm_80 or newer"
#endif

// Independent chains per thread: four for FFMA; two for DFMA (its operands take two registers each, and it runs at 1/64
// of the FFMA rate on GA102) and HMMA (10 registers per chain and set), so that 4 blocks of 256 threads fit per SM.
#if OPERAND_OP != 0
#define OPERAND_CHAINS 2
#else
#define OPERAND_CHAINS 4
#endif
#define OPERAND_WORDS 128 // payload words per thread

__device__ __forceinline__ float OperandFloat(const u32 word)
{
    return __uint_as_float((word & 0x807fffffu) | 0x3f800000u);
}

__device__ __forceinline__ double OperandDouble(const u32 high, const u32 low)
{
    return __hiloint2double(static_cast<int>((high & 0x800fffffu) | 0x3ff00000u), static_cast<int>(low));
}

__device__ __forceinline__ u32 OperandHalves(const u32 word)
{
    return (word & 0x83ff83ffu) | 0x3c003c00u;
}

extern "C" __global__ void probeOperands(const u32* __restrict__ data, u32* __restrict__ sink, u64* __restrict__ counts,
    const u32 activeSms, const u32 idleRatio, const u32 zero, const u64 durationNs)
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

    const u32* const words = data + static_cast<u64>(blockIdx.x * BLOCK_SIZE + threadIdx.x) * OPERAND_WORDS;
    u32 previous[OPERAND_CHAINS]; // bits of the last result of each chain

    // Operand j of chain i in set s: word 2 * (j + 3 * i) + s (doubles: low word 4 * (j + 3 * i) + s, high word + 2).
#if OPERAND_OP == 0
    float a[2][OPERAND_CHAINS], b[2][OPERAND_CHAINS], c[2][OPERAND_CHAINS];

    for (int s = 0; s < 2; ++s)
    {
        for (int i = 0; i < OPERAND_CHAINS; ++i)
        {
            a[s][i] = OperandFloat(words[2 * (0 + 3 * i) + s]);
            b[s][i] = OperandFloat(words[2 * (1 + 3 * i) + s]);
            c[s][i] = OperandFloat(words[2 * (2 + 3 * i) + s]);
        }
    }
#elif OPERAND_OP == 1
    double a[2][OPERAND_CHAINS], b[2][OPERAND_CHAINS], c[2][OPERAND_CHAINS];

    for (int s = 0; s < 2; ++s)
    {
        for (int i = 0; i < OPERAND_CHAINS; ++i)
        {
            a[s][i] = OperandDouble(words[4 * (0 + 3 * i) + 2 + s], words[4 * (0 + 3 * i) + s]);
            b[s][i] = OperandDouble(words[4 * (1 + 3 * i) + 2 + s], words[4 * (1 + 3 * i) + s]);
            c[s][i] = OperandDouble(words[4 * (2 + 3 * i) + 2 + s], words[4 * (2 + 3 * i) + s]);
        }
    }
#elif OPERAND_OP == 2
    // Per chain and set: A in 4 registers of two halves, B in 2, C in 4 floats (word 2 * (r + 10 * i) + s, r = 0..9).
    u32 a[2][OPERAND_CHAINS][4], b[2][OPERAND_CHAINS][2];
    float c[2][OPERAND_CHAINS][4];

    for (int s = 0; s < 2; ++s)
    {
        for (int i = 0; i < OPERAND_CHAINS; ++i)
        {
            for (int r = 0; r < 4; ++r)
            {
                a[s][i][r] = OperandHalves(words[2 * (r + 10 * i) + s]);
                c[s][i][r] = OperandFloat(words[2 * (6 + r + 10 * i) + s]);
            }

            for (int r = 0; r < 2; ++r)
            {
                b[s][i][r] = OperandHalves(words[2 * (4 + r + 10 * i) + s]);
            }
        }
    }
#else
#error "Unsupported OPERAND_OP"
#endif

    for (int i = 0; i < OPERAND_CHAINS; ++i)
    {
        previous[i] = 0;
    }

    const u32 activeNs = static_cast<u32>((static_cast<u64>(DUTY_PERIOD_NS) << 16) / (65536u + idleRatio));
    u64 iterations = 0;
    const u64 start = globalTimer();
    u64 now = start;

    do
    {
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
        {
            const int s = u & 1;

            #pragma unroll
            for (int i = 0; i < OPERAND_CHAINS; ++i)
            {
#if OPERAND_OP == 0
                const float first = __uint_as_float(__float_as_uint(a[s][i]) ^ (previous[i] & zero));
                float d;
                asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(first), "f"(b[s][i]), "f"(c[s][i]));
                previous[i] = __float_as_uint(d);
#elif OPERAND_OP == 1
                const long long bits = __double_as_longlong(a[s][i]) ^ static_cast<long long>(previous[i] & zero);
                double d;
                asm volatile("fma.rn.f64 %0, %1, %2, %3;" : "=d"(d) : "d"(__longlong_as_double(bits)), "d"(b[s][i]),
                    "d"(c[s][i]));
                previous[i] = static_cast<u32>(__double_as_longlong(d));
#elif OPERAND_OP == 2
                const u32 first = a[s][i][0] ^ (previous[i] & zero);
                float d0, d1, d2, d3;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                    "{%10,%11,%12,%13};"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    : "r"(first), "r"(a[s][i][1]), "r"(a[s][i][2]), "r"(a[s][i][3]), "r"(b[s][i][0]), "r"(b[s][i][1]),
                      "f"(c[s][i][0]), "f"(c[s][i][1]), "f"(c[s][i][2]), "f"(c[s][i][3]));
                previous[i] = __float_as_uint(d0) ^ __float_as_uint(d1) ^ __float_as_uint(d2) ^ __float_as_uint(d3);
#endif
            }
        }

        iterations += UNROLL * OPERAND_CHAINS;
        now = globalTimer();
        IdleForDutyWindow(activeNs, now);
    }
    while (now - start < durationNs);

    u32 result = 0;

    for (int i = 0; i < OPERAND_CHAINS; ++i)
    {
        result ^= previous[i];
    }

    writeBlockStats(&blockCount, counts, iterations, now - start);
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = result;
}
