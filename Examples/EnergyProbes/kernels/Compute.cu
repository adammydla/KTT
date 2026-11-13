// Arithmetic and register-array indexing (probeCompute).
//
// Parameter COMPUTE_OP: 0 = loop without operations (overhead reference), 1 = FFMA, 2 = DFMA, 3 = IMAD, 4 = HFMA2,
// 5 = HMMA (mma.sync with f16 inputs and f32 accumulators, in the GPU's shape: m16n8k16 from sm_80, m16n8k8 on sm_75,
// m8n8k4 on sm_70), 6 = MUFU (rsqrt.approx, followed by the
// FFMA that feeds the chain), 7 = 32 FFMA accumulators with static indices, 8 = the same with a runtime index (compares
// and guarded moves, as in the Coulomb example), 9 = FFMA with three different register operands (see below);
// instruction classes that the operations above do not cover: 10 = SHFL (shfl.sync.bfly), 11 = BAR (bar.sync),
// 12 = the FFMA chain in a loop that is not unrolled (a branch and its compare per 4 FFMAs), 13 = x * x + c as a separate
// FMUL and FADD, 14 = MUFU.SIN (x -> sin(a * x + c)).

#ifndef COMPUTE_OP
#define COMPUTE_OP 1
#endif

// The registers per thread of the mma.sync operands (A and B hold two f16 each, D four or eight f32): m16n8k16 from sm_80,
// m16n8k8 on sm_75, m8n8k4 on sm_70 (where each quarter of the warp computes an 8x8x4 product). Every shape is one warp
// instruction of 2048 (sm_7x) or 4096 (sm_80 and newer) FLOPs; on sm_70 ptxas splits m8n8k4 into several HMMA.884
// instructions, which the tensor-pipe counter counts.
#if COMPUTE_OP == 5
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define HMMA_A 4
#define HMMA_B 2
#define HMMA_D 4
#elif __CUDA_ARCH__ >= 750
#define HMMA_A 2
#define HMMA_B 1
#define HMMA_D 4
#elif __CUDA_ARCH__ >= 700
#define HMMA_A 2
#define HMMA_B 2
#define HMMA_D 8
#else
#error "COMPUTE_OP 5 (HMMA) needs sm_70 or newer"
#endif
#endif

// Independent dependency chains per thread, so that the pipelines are not latency-bound.
#define COMPUTE_CHAINS 4

// Accumulators of the indexing probes (COMPUTE_OP 7 and 8), as Z_ITERATIONS in the Coulomb example, and the unroll factor
// of their loop in COMPUTE_OP 8 (its INNER_UNROLL_FACTOR).
#define INDEXED_VALUES 32
#define INDEXED_UNROLL 4

// Arithmetic: COMPUTE_CHAINS independent chains per thread, one operation per chain and step, counted per thread (one warp
// instruction is 32 operations). The operands stay bounded but keep changing, so that the datapath toggles as with real
// data (as in Antepara et al., SC '25): floating point follows x -> x * x + c with c = -1.9 from a start in [1, 1.5)
// (chaotic and inside [-1.9, 1.71]; c = -2 would collapse onto the fixed point 2 in finite precision), integers follow the
// LCG x -> x * a + c. HMMA accumulates D += A * B with two alternating sets of A and B. MUFU follows x -> a * rsqrt(|x|) + c
// with a = -1, c = 1.5 (bounded except for rare spikes, which the next step maps back to about 1.5; the absolute value is an
// operand modifier, so one step is one MUFU.RSQ and one FFMA). The host passes c (operandC) and a
// (operandA); a zero payload makes all operands and constants zero. COMPUTE_OP 0 is the same loop without operations; its
// cost per counted operation is the loop overhead to subtract.
//
// Register-array indexing, counted per accumulator update of one thread: INDEXED_VALUES accumulators follow the FFMA
// recurrence. COMPUTE_OP 7 updates them in a fully unrolled loop (static indices: FFMAs only); COMPUTE_OP 8 unrolls the
// same loop only INDEXED_UNROLL times, as the Coulomb example's inner loop over Z_ITERATIONS, so the index is a runtime value
// and ptxas compiles every access into compares and guarded moves that select the register (most of them with the guard
// false in every lane). The difference is what dynamic indexing of a register array costs.
//
// FFMA with three register operands (COMPUTE_OP 9): COMPUTE_OP 1 computes x * x + c, which reads one register twice and
// takes c from the constant bank, while the FMAs of real kernels (acc = a * b + acc) read three different registers. Here
// one chain is held in four registers a, b, c, d, updated from the old values as a = a*b + c0, b = c*d + c1, c = a*c + c2,
// d = b*d + c3 with c0 = ... = c3 = c in registers: four independent FFMAs per step, as the COMPUTE_CHAINS chains of the
// other operations. The copies stay bit-identical, so the values follow the recurrence of COMPUTE_OP 1, but every FFMA
// reads three different registers, and no two FFMAs multiply the same pair (with two copies x = x*y, y = y*x, ptxas put
// each pair back to back and read both factors of the second from the operand reuse cache). It still reuses some operands
// (39 of 192 per loop iteration on sm_86). The copies and the constants pass through memory (ThroughMemory, before the
// loop), so that ptxas cannot see that they are equal and neither merges FFMAs nor folds c into the instruction (Opaque()
// emits no PTX and did not stop it).
// Stores `value` to `slot` and loads it back, both volatile: the result is a register whose value ptxas cannot know.
__device__ __forceinline__ u32 ThroughMemory(u32* slot, const u32 value)
{
    u32 result;
    asm volatile("st.volatile.global.u32 [%1], %2;\n\tld.volatile.global.u32 %0, [%1];" : "=r"(result) : "l"(slot), "r"(value)
        : "memory");
    return result;
}

extern "C" __global__ void probeCompute(u32* __restrict__ sink, u64* __restrict__ counts, const u32 activeSms,
    const u32 idleRatio, const u32 payloadMask, const u32 operandA, const u32 operandC, const u64 durationNs)
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

    const u32 seed = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 16u;

#if COMPUTE_OP == 1
    float value[COMPUTE_CHAINS];
    const float c = __uint_as_float(operandC);

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        value[i] = __uint_as_float((0x3f800000u | (mix(seed + i) >> 10)) & payloadMask);
    }
#elif COMPUTE_OP == 2
    double value[COMPUTE_CHAINS];
    const double c = static_cast<double>(__uint_as_float(operandC));

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        value[i] = static_cast<double>(__uint_as_float((0x3f800000u | (mix(seed + i) >> 10)) & payloadMask));
    }
#elif COMPUTE_OP == 3
    u32 value[COMPUTE_CHAINS];
    const u32 a = operandA;
    const u32 c = operandC;

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        value[i] = mix(seed + i) & payloadMask;
    }
#elif COMPUTE_OP == 4
    u32 value[COMPUTE_CHAINS]; // two halves per register
    const u32 c = operandC;

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        value[i] = (0x3c003c00u | (mix(seed + i) & 0x01ff01ffu)) & payloadMask;
    }
#elif COMPUTE_OP == 6
    float value[COMPUTE_CHAINS];
    const float a = __uint_as_float(operandA);
    const float c = __uint_as_float(operandC);

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        value[i] = __uint_as_float((0x3f800000u | (mix(seed + i) >> 10)) & payloadMask);
    }
#elif COMPUTE_OP == 5
    u32 a[2][HMMA_A];
    u32 b[2][HMMA_B];
    float d[COMPUTE_CHAINS][HMMA_D];

    for (int s = 0; s < 2; ++s)
    {
        for (int j = 0; j < HMMA_A; ++j)
        {
            a[s][j] = (0x3c003c00u | (mix(seed + 8 * s + j) & 0x01ff01ffu)) & payloadMask;
        }

        for (int j = 0; j < HMMA_B; ++j)
        {
            b[s][j] = (0x3c003c00u | (mix(seed + 8 * s + 4 + j) & 0x01ff01ffu)) & payloadMask;
        }
    }

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        for (int j = 0; j < HMMA_D; ++j)
        {
            d[i][j] = 0.0f;
        }
    }
#elif COMPUTE_OP == 9
    static_assert(COMPUTE_CHAINS == 4, "COMPUTE_OP 9 updates four copies of one chain");
    u32* const slot = &sink[blockIdx.x * BLOCK_SIZE + threadIdx.x];
    const u32 initial = (0x3f800000u | (mix(seed) >> 10)) & payloadMask;
    float copies[COMPUTE_CHAINS];
    float constants[COMPUTE_CHAINS];

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        copies[i] = __uint_as_float(ThroughMemory(slot, initial));
        constants[i] = __uint_as_float(ThroughMemory(slot, operandC));
    }
#elif COMPUTE_OP == 10
    // Shuffles of random words between the lanes of a warp (each chain with its own lane mask): no arithmetic, only SHFL.
    u32 value[COMPUTE_CHAINS];

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        value[i] = mix(seed + i) & payloadMask;
    }
#elif COMPUTE_OP == 11
    // Block barriers only. A warp that left the loop while the others wait at a barrier would hang them, so the block
    // decides together when to stop (COMPUTE_MORE).
#elif COMPUTE_OP == 12 || COMPUTE_OP == 13
    // The FFMA recurrence of COMPUTE_OP 1: 12 in a loop of UNROLL iterations that is not unrolled (each iteration the four
    // chains' FFMAs, an IADD, a compare and a branch), 13 with the multiply and the add as separate FMUL and FADD.
    float value[COMPUTE_CHAINS];
    const float c = __uint_as_float(operandC);

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        value[i] = __uint_as_float((0x3f800000u | (mix(seed + i) >> 10)) & payloadMask);
    }
#elif COMPUTE_OP == 14
    // x -> sin(a * x + c) with a = 3, c = 0.5 (chaotic inside [-1, 1]): an FFMA and __sinf's FMUL and MUFU.SIN per step.
    float value[COMPUTE_CHAINS];
    const float a = __uint_as_float(operandA);
    const float c = __uint_as_float(operandC);

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        value[i] = __uint_as_float((0x3f800000u | (mix(seed + i) >> 10)) & payloadMask);
    }
#elif COMPUTE_OP == 7 || COMPUTE_OP == 8
    float value[INDEXED_VALUES];
    const float c = __uint_as_float(operandC);

    for (int i = 0; i < INDEXED_VALUES; ++i)
    {
        value[i] = __uint_as_float((0x3f800000u | (mix(seed + i) >> 10)) & payloadMask);
    }
#elif COMPUTE_OP != 0
#error "Unsupported COMPUTE_OP"
#endif

    // Whether to run another loop iteration: for the barrier probe, decided by the whole block.
#if COMPUTE_OP == 11
#define COMPUTE_MORE(now) (__syncthreads_or((now) - start < durationNs) != 0)
#else
#define COMPUTE_MORE(now) ((now) - start < durationNs)
#endif

    // Working time per period for the duty 1 / (1 + idleRatio).
    const u32 activeNs = static_cast<u32>((static_cast<u64>(DUTY_PERIOD_NS) << 16) / (65536u + idleRatio));
    u64 iterations = 0;
    const u64 start = globalTimer();
    u64 now = start;

    do
    {
#if COMPUTE_OP == 7
        // One pass over the accumulators per timer check, as one pass of the Coulomb example's inner loop.
        #pragma unroll
        for (int j = 0; j < INDEXED_VALUES; ++j)
        {
            value[j] = fmaf(value[j], value[j], c);
        }
#elif COMPUTE_OP == 8
        #pragma unroll INDEXED_UNROLL
        for (int j = 0; j < INDEXED_VALUES; ++j)
        {
            value[j] = fmaf(value[j], value[j], c);
        }
#elif COMPUTE_OP == 12
        #pragma unroll 1
        for (int u = 0; u < UNROLL; ++u)
        {
            #pragma unroll
            for (int i = 0; i < COMPUTE_CHAINS; ++i)
            {
                value[i] = fmaf(value[i], value[i], c);
            }
        }
#else
        #pragma unroll
        for (int u = 0; u < UNROLL; ++u)
        {
            #pragma unroll
            for (int i = 0; i < COMPUTE_CHAINS; ++i)
            {
#if COMPUTE_OP == 1
                value[i] = fmaf(value[i], value[i], c);
#elif COMPUTE_OP == 2
                value[i] = fma(value[i], value[i], c);
#elif COMPUTE_OP == 3
                value[i] = value[i] * a + c;
#elif COMPUTE_OP == 4
                asm("fma.rn.f16x2 %0, %0, %0, %1;" : "+r"(value[i]) : "r"(c));
#elif COMPUTE_OP == 5
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                    "{%0,%1,%2,%3};"
                    : "+f"(d[i][0]), "+f"(d[i][1]), "+f"(d[i][2]), "+f"(d[i][3])
                    : "r"(a[i % 2][0]), "r"(a[i % 2][1]), "r"(a[i % 2][2]), "r"(a[i % 2][3]), "r"(b[i % 2][0]),
                      "r"(b[i % 2][1]));
#elif __CUDA_ARCH__ >= 750
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                    : "+f"(d[i][0]), "+f"(d[i][1]), "+f"(d[i][2]), "+f"(d[i][3])
                    : "r"(a[i % 2][0]), "r"(a[i % 2][1]), "r"(b[i % 2][0]));
#else
                asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, "
                    "{%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
                    : "+f"(d[i][0]), "+f"(d[i][1]), "+f"(d[i][2]), "+f"(d[i][3]), "+f"(d[i][4]), "+f"(d[i][5]),
                      "+f"(d[i][6]), "+f"(d[i][7])
                    : "r"(a[i % 2][0]), "r"(a[i % 2][1]), "r"(b[i % 2][0]), "r"(b[i % 2][1]));
#endif
#elif COMPUTE_OP == 6
                float root;
                asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(root) : "f"(fabsf(value[i])));
                value[i] = fmaf(root, a, c);
#elif COMPUTE_OP == 10
                asm volatile("shfl.sync.bfly.b32 %0, %0, %1, 0x1f, 0xffffffff;" : "+r"(value[i]) : "r"(1u << i));
#elif COMPUTE_OP == 11
                asm volatile("bar.sync 0;" ::: "memory");
#elif COMPUTE_OP == 13
                value[i] = __fadd_rn(__fmul_rn(value[i], value[i]), c);
#elif COMPUTE_OP == 14
                value[i] = __sinf(fmaf(value[i], a, c));
#elif COMPUTE_OP == 9
                // One step updates all four copies (from their old values) at i == 0: four FFMAs, as four chains.
                if (i == 0)
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
            }
        }
#endif

#if COMPUTE_OP == 7 || COMPUTE_OP == 8
        iterations += INDEXED_VALUES;
#else
        iterations += UNROLL * COMPUTE_CHAINS;
#endif
        now = globalTimer();
        IdleForDutyWindow(activeNs, now);
    }
    while (COMPUTE_MORE(now));

    u32 result = 0;

#if COMPUTE_OP == 1 || COMPUTE_OP == 6 || COMPUTE_OP == 12 || COMPUTE_OP == 13 || COMPUTE_OP == 14
    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        result ^= __float_as_uint(value[i]);
    }
#elif COMPUTE_OP == 2
    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        result ^= static_cast<u32>(__double_as_longlong(value[i]));
    }
#elif COMPUTE_OP == 3 || COMPUTE_OP == 4 || COMPUTE_OP == 10
    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        result ^= value[i];
    }
#elif COMPUTE_OP == 5
    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        for (int j = 0; j < HMMA_D; ++j)
        {
            result ^= __float_as_uint(d[i][j]);
        }
    }
#elif COMPUTE_OP == 7 || COMPUTE_OP == 8
    for (int i = 0; i < INDEXED_VALUES; ++i)
    {
        result ^= __float_as_uint(value[i]);
    }
#elif COMPUTE_OP == 9
    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        result ^= __float_as_uint(copies[i]);
    }
#endif

    writeBlockStats(&blockCount, counts, iterations, now - start);
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = result;
}
