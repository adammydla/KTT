// Arithmetic and other instruction classes (probeCompute).
//
// Parameter COMPUTE_OP: 0 = loop without operations (overhead reference), 1 = FFMA (x * x + c), 2 = DFMA, 3 = IMAD,
// 4 = HFMA2, 5 = HMMA (mma.sync with f16 inputs and f32 accumulators, in the GPU's shape: m16n8k16 from sm_80, m16n8k8
// on sm_75, m8n8k4 on sm_70), 6 = MUFU (rsqrt.approx, followed by the FFMA that feeds the chain), 8 = 32 FFMA
// accumulators with a runtime index (compares and guarded moves, as in the Coulomb example), 9 = FFMA with three
// different register operands (see below), 10 = SHFL (shfl.sync.bfly), 11 = BAR (bar.sync), 12 = the FFMA chain in a
// loop that is not unrolled (a branch and its compare per 4 FFMAs).
//
// The other tensor-core paths (NEWER_ARCHITECTURES.md items 7 and 13; one warp instruction per step and chain, as HMMA;
// IMMA in the compute group since 2026-11-16, the others in the optional group "tensor"): 13 = IMMA (mma.sync m16n8k32 s8 x s8 + s32, sm_80 and newer; cuBLAS emulates FP64 GEMM
// with it on Blackwell), 14 = DMMA (mma.sync m8n8k4 f64, sm_80 and newer; on GA102/AD102 counted on the fp64 pipe, on
// GH100/GB100 on the tensor pipe), 15 = sparse HMMA (mma.sp m16n8k32 f16 with 2:4 metadata, sm_80 and newer),
// 16 = FP8 QMMA (mma.sync m16n8k32 e4m3 x e4m3 + f32: native only on sm_89 and sm_120; on sm_90 and sm_100 ptxas
// emulates it with conversions and HMMA, so the planner runs it only on 8.9 and 12.x).
// 17 = one SASS opcode (COMPUTE_OPCODE, the optional group "opcodes", 2026-11-17): see "Opcode probes" below.

#ifndef COMPUTE_OP
#define COMPUTE_OP 1
#endif

// Partly idle warps (the "lanes" group, 2026-11-17): only the first COMPUTE_LANES lanes of every warp run the operations,
// the others branch around them (divergence, as in a reduction tree or AccelWattch's ACT_CORE benchmarks), and only the
// working lanes count their operations. With 32 (the default) the code is the full-warp probe's.
#ifndef COMPUTE_LANES
#define COMPUTE_LANES 32
#endif
#if COMPUTE_LANES < 1 || COMPUTE_LANES > 32
#error "COMPUTE_LANES must be 1-32"
#endif

// Opcode probes (COMPUTE_OP 17, the optional group "opcodes"): the energy of single SASS opcodes within a pipe, which the
// model prices per pipe (Wattchmen prices every opcode). COMPUTE_OPCODE picks the opcode; each chain and step is one
// instruction of it, plus helpers where the opcode alone cannot keep its data changing (tools/opcode_energy.py counts every
// opcode of the loop from the SASS and subtracts the helpers). Operands: 16 registers per thread, a different one in each
// step of a chain, so the operand bits change from instruction to instruction as with real data; they come in cancelling
// pairs (o and -o for additions, m and 1/m for multiplications), so the chains stay bounded. COMPUTE_CONSTANT 1 makes the
// operands almost constant (the same value, only its sign or its inverse alternates), as the constant operands of
// AccelWattch's benchmarks and of many real kernels (scale factors, coefficients).
//   1 FADD  x += o              2 FMUL  x *= m              3 FFMA  x = x * m + o (three registers)
//   4 DADD  x += o              5 DMUL  x *= m              6 DFMA  x = x * m + o (three registers)
//   7 IADD3 x += o              8 LOP3  x ^= o              9 SHF   x = funnel shift of (x, o)
//   10-15 MUFU EX2, LG2, SIN, RCP, SQRT, TANH of the FFMA chain x * x + c (helpers: the FFMA, and LOP3 that folds the
//         result into an accumulator)
//   16 I2F of the IMAD chain (LCG) (helpers: IMAD, LOP3)   17 F2I of the FFMA chain (helpers: FFMA, FMUL, LOP3)
//   18 POPC of the IMAD chain (helpers: IMAD, IADD3)
//   19 F2F float to double and back of the FFMA chain (helpers: FFMA, DMUL by the constant 0.1, LOP3), as kernels
//      that mix float data with double literals (hotspot's 2.0 * t)
//   20-22 FFMA x = x * m + o with the operand reuse cache (the optional group "reuse", 2026-11-18): all four chains
//         take the same m and o in a step (m0, m1, 1/m0, 1/m1 and o0, o1, -o0, -o1, as each chain of opcode 3), so the
//         operand values are the same in all three; only the registers they come from differ. 20: every chain reads its
//         own copy of m and o (no two consecutive FFMAs read the same register, so nothing is reused); 21: all chains
//         read one register of m (ptxas marks m .reuse in three of the four FFMAs of a step); 22: one register of m and
//         one of o (two operands reused). The copies pass through memory (ThroughMemory), so that ptxas cannot merge
//         them.
#ifndef COMPUTE_OPCODE
#define COMPUTE_OPCODE 1
#endif
#ifndef COMPUTE_CONSTANT
#define COMPUTE_CONSTANT 0
#endif
#define OPCODE_OPERANDS 16
#define OPCODE_FLOAT (COMPUTE_OPCODE <= 3)
#define OPCODE_DOUBLE (COMPUTE_OPCODE >= 4 && COMPUTE_OPCODE <= 6)
#define OPCODE_INTEGER (COMPUTE_OPCODE >= 7 && COMPUTE_OPCODE <= 9)
#define OPCODE_MUFU (COMPUTE_OPCODE >= 10 && COMPUTE_OPCODE <= 15)
#define OPCODE_REUSE (COMPUTE_OPCODE >= 20 && COMPUTE_OPCODE <= 22)

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

#if COMPUTE_OP >= 13 && COMPUTE_OP <= 15 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#error "COMPUTE_OP 13-15 (IMMA, DMMA, sparse HMMA) need sm_80 or newer"
#endif
#if COMPUTE_OP == 16 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
#error "COMPUTE_OP 16 (FP8 mma.sync) needs sm_89 or newer"
#endif

// The 2:4 metadata of the sparse HMMA: every group of four A elements keeps elements 0 and 1 (indices 0b00, 0b01).
#define SPARSE_METADATA 0x44444444u

// Independent dependency chains per thread, so that the pipelines are not latency-bound.
#define COMPUTE_CHAINS 4

// Accumulators of the indexing probe (COMPUTE_OP 8), as Z_ITERATIONS in the Coulomb example, and the unroll factor of its
// loop (the example's INNER_UNROLL_FACTOR).
#define INDEXED_VALUES 32
#define INDEXED_UNROLL 4

// Arithmetic: COMPUTE_CHAINS independent chains per thread, one operation per chain and step, counted per thread (one warp
// instruction is 32 operations). The operands stay bounded but keep changing, so that the datapath toggles as with real
// data (as in Antepara et al., SC '25): floating point follows x -> x * x + c with c = -1.9 from a start in [1, 1.5)
// (chaotic and inside [-1.9, 1.71]; c = -2 would collapse onto the fixed point 2 in finite precision), integers follow the
// LCG x -> x * a + c. HMMA accumulates D += A * B with two alternating sets of A and B. MUFU follows x -> a * rsqrt(|x|) + c
// with a = -1, c = 1.5 (bounded except for rare spikes, which the next step maps back to about 1.5; the absolute value is an
// operand modifier, so one step is one MUFU.RSQ and one FFMA). The host passes c (operandC) and a (operandA); payloadMask 0
// would make all operands zero. COMPUTE_OP 0 is the same loop without operations; its cost per counted operation is the
// loop overhead.
//
// Register-array indexing (COMPUTE_OP 8), counted per accumulator update of one thread: INDEXED_VALUES accumulators follow
// the FFMA recurrence in a loop unrolled only INDEXED_UNROLL times, as the Coulomb example's inner loop over
// Z_ITERATIONS, so the index is a runtime value and ptxas compiles every access into compares and guarded moves that select
// the register (most of them with the guard false in every lane).
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
#elif COMPUTE_OP == 13 || COMPUTE_OP == 15 || COMPUTE_OP == 16
    // A in 4 registers per thread for all three (IMMA and FP8: 16 x 32 bytes; sparse: the kept half of 16 x 32 f16), B in
    // 2 (IMMA, FP8: 32 x 8 bytes) or 4 registers (sparse: 32 x 8 f16). s8 operands take any bits; f16 operands lie in
    // [1, 2) as in HMMA; e4m3 bytes keep the exponent near 0 (no NaN encodings, values about 0.5 to 2).
#if COMPUTE_OP == 15
#define TENSOR_B 4
    const u32 base = 0x3c003c00u;
    const u32 random = 0x01ff01ffu;
#elif COMPUTE_OP == 16
#define TENSOR_B 2
    const u32 base = 0x30303030u;
    const u32 random = 0x87878787u;
#else
#define TENSOR_B 2
    const u32 base = 0u;
    const u32 random = 0xffffffffu;
#endif
    u32 a[2][4];
    u32 b[2][TENSOR_B];
#if COMPUTE_OP == 13
    int d[COMPUTE_CHAINS][4];
#else
    float d[COMPUTE_CHAINS][4];
#endif

    for (int s = 0; s < 2; ++s)
    {
        for (int j = 0; j < 4; ++j)
        {
            a[s][j] = (base | (mix(seed + 8 * s + j) & random)) & payloadMask;
        }

        for (int j = 0; j < TENSOR_B; ++j)
        {
            b[s][j] = (base | (mix(seed + 8 * s + 4 + j) & random)) & payloadMask;
        }
    }

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            d[i][j] = 0;
        }
    }
#elif COMPUTE_OP == 14
    // m8n8k4 f64: A and B one double per thread, D two.
    double a[2];
    double b[2];
    double d[COMPUTE_CHAINS][2];

    for (int s = 0; s < 2; ++s)
    {
        a[s] = static_cast<double>(__uint_as_float((0x3f800000u | (mix(seed + 2 * s) >> 10)) & payloadMask));
        b[s] = static_cast<double>(__uint_as_float((0x3f800000u | (mix(seed + 2 * s + 1) >> 10)) & payloadMask));
    }

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        d[i][0] = 0.0;
        d[i][1] = 0.0;
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
#elif COMPUTE_OP == 12
    // The FFMA recurrence of COMPUTE_OP 1 in a loop of UNROLL iterations that is not unrolled (each iteration the four
    // chains' FFMAs, an IADD, a compare and a branch).
    float value[COMPUTE_CHAINS];
    const float c = __uint_as_float(operandC);

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        value[i] = __uint_as_float((0x3f800000u | (mix(seed + i) >> 10)) & payloadMask);
    }
#elif COMPUTE_OP == 8
    float value[INDEXED_VALUES];
    const float c = __uint_as_float(operandC);

    for (int i = 0; i < INDEXED_VALUES; ++i)
    {
        value[i] = __uint_as_float((0x3f800000u | (mix(seed + i) >> 10)) & payloadMask);
    }
#elif COMPUTE_OP == 17
#if COMPUTE_OPCODE < 1 || COMPUTE_OPCODE > 22
#error "COMPUTE_OPCODE must be 1-22"
#endif
    // Cancelling operand pairs: operand k + 8 undoes operand k (k < 8). Chain i uses operands i, i + 4, i + 8, i + 12 in
    // turn, so every four of its steps cancel.
#if OPCODE_FLOAT || OPCODE_MUFU || COMPUTE_OPCODE == 17 || COMPUTE_OPCODE == 19 || OPCODE_REUSE
    float value[COMPUTE_CHAINS];
    const float c = __uint_as_float(operandC);
#elif OPCODE_DOUBLE
    double value[COMPUTE_CHAINS];
#else
    u32 value[COMPUTE_CHAINS];
    const u32 a = operandA;
#endif
#if OPCODE_FLOAT || OPCODE_DOUBLE
#if OPCODE_FLOAT
    typedef float Real;
#else
    typedef double Real;
#endif
    Real addend[OPCODE_OPERANDS];
    Real factor[OPCODE_OPERANDS];

    for (int k = 0; k < OPCODE_OPERANDS / 2; ++k)
    {
        // Addends in [-0.5, 0.5), factors in [1, 2); constant: 0.3 and 1.25.
        const u32 bits = mix(seed + 64 + k) & payloadMask;
#if COMPUTE_CONSTANT
        addend[k] = static_cast<Real>(0.3f);
        factor[k] = static_cast<Real>(1.25f);
#else
        addend[k] = static_cast<Real>(__uint_as_float(0x3f800000u | (bits >> 9)) - 1.5f);
        factor[k] = static_cast<Real>(__uint_as_float(0x3f800000u | (mix(bits) >> 9)));
#endif
        addend[k + OPCODE_OPERANDS / 2] = -addend[k];
        factor[k + OPCODE_OPERANDS / 2] = static_cast<Real>(1) / factor[k];
    }

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        value[i] = static_cast<Real>(__uint_as_float((0x3f800000u | (mix(seed + i) >> 10)) & payloadMask));
    }
#elif OPCODE_REUSE
    // The operands of step u: base index u % 4 for every chain (opcode 3's chain i takes operands i, i + 4, i + 8, i + 12).
    u32* const slot = &sink[blockIdx.x * BLOCK_SIZE + threadIdx.x];
    float baseFactor[4];
    float baseAddend[4];

    for (int k = 0; k < 2; ++k)
    {
        const u32 bits = mix(seed + 64 + k) & payloadMask;
        baseAddend[k] = __uint_as_float(0x3f800000u | (bits >> 9)) - 1.5f;
        baseFactor[k] = __uint_as_float(0x3f800000u | (mix(bits) >> 9));
        baseAddend[k + 2] = -baseAddend[k];
        baseFactor[k + 2] = 1.0f / baseFactor[k];
    }

    // One copy of each operand per chain; opcodes 21 and 22 read only chain 0's copies of the shared operands.
    float factor[COMPUTE_CHAINS][4];
    float addend[COMPUTE_CHAINS][4];

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            factor[i][j] = __uint_as_float(ThroughMemory(slot, __float_as_uint(baseFactor[j])));
            addend[i][j] = __uint_as_float(ThroughMemory(slot, __float_as_uint(baseAddend[j])));
        }

        value[i] = __uint_as_float((0x3f800000u | (mix(seed + i) >> 10)) & payloadMask);
    }
#elif OPCODE_INTEGER
    u32 operand[OPCODE_OPERANDS];

    for (int k = 0; k < OPCODE_OPERANDS; ++k)
    {
#if COMPUTE_CONSTANT
        operand[k] = 0x9e3779b9u & payloadMask;
#else
        operand[k] = mix(seed + 64 + k) & payloadMask;
#endif
    }

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        value[i] = mix(seed + i) & payloadMask;
    }
#else
    // MUFU and conversions: a chain that keeps changing (FFMA x * x + c, or the LCG), the opcode applied to it, and its
    // results folded into an accumulator so that ptxas keeps them.
    u32 folded[COMPUTE_CHAINS];

    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        folded[i] = 0;
#if COMPUTE_OPCODE == 16 || COMPUTE_OPCODE == 18
        value[i] = mix(seed + i) & payloadMask;
#else
        value[i] = __uint_as_float((0x3f800000u | (mix(seed + i) >> 10)) & payloadMask);
#endif
    }
#endif
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
#if COMPUTE_OP == 8
        // One pass over the accumulators per timer check, as one pass of the Coulomb example's inner loop.
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
#if COMPUTE_LANES < 32
        if ((threadIdx.x & 31u) < COMPUTE_LANES)
        {
#endif
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
#elif COMPUTE_OP == 13
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                    "{%0,%1,%2,%3};"
                    : "+r"(d[i][0]), "+r"(d[i][1]), "+r"(d[i][2]), "+r"(d[i][3])
                    : "r"(a[i % 2][0]), "r"(a[i % 2][1]), "r"(a[i % 2][2]), "r"(a[i % 2][3]), "r"(b[i % 2][0]),
                      "r"(b[i % 2][1]));
#elif COMPUTE_OP == 14
                asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0,%1}, {%2}, {%3}, {%0,%1};"
                    : "+d"(d[i][0]), "+d"(d[i][1])
                    : "d"(a[i % 2]), "d"(b[i % 2]));
#elif COMPUTE_OP == 15
                asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
                    "{%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
                    : "+f"(d[i][0]), "+f"(d[i][1]), "+f"(d[i][2]), "+f"(d[i][3])
                    : "r"(a[i % 2][0]), "r"(a[i % 2][1]), "r"(a[i % 2][2]), "r"(a[i % 2][3]), "r"(b[i % 2][0]),
                      "r"(b[i % 2][1]), "r"(b[i % 2][2]), "r"(b[i % 2][3]), "r"(SPARSE_METADATA));
#elif COMPUTE_OP == 16
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                    "{%0,%1,%2,%3};"
                    : "+f"(d[i][0]), "+f"(d[i][1]), "+f"(d[i][2]), "+f"(d[i][3])
                    : "r"(a[i % 2][0]), "r"(a[i % 2][1]), "r"(a[i % 2][2]), "r"(a[i % 2][3]), "r"(b[i % 2][0]),
                      "r"(b[i % 2][1]));
#elif COMPUTE_OP == 6
                float root;
                asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(root) : "f"(fabsf(value[i])));
                value[i] = fmaf(root, a, c);
#elif COMPUTE_OP == 10
                asm volatile("shfl.sync.bfly.b32 %0, %0, %1, 0x1f, 0xffffffff;" : "+r"(value[i]) : "r"(1u << i));
#elif COMPUTE_OP == 11
                asm volatile("bar.sync 0;" ::: "memory");
#elif COMPUTE_OP == 17
                {
                    const int k = (u * COMPUTE_CHAINS + i) % OPCODE_OPERANDS;
#if COMPUTE_OPCODE == 1 || COMPUTE_OPCODE == 4
                    value[i] = value[i] + addend[k];
#elif COMPUTE_OPCODE == 2 || COMPUTE_OPCODE == 5
                    value[i] = value[i] * factor[k];
#elif COMPUTE_OPCODE == 3 || COMPUTE_OPCODE == 6
                    value[i] = fma(value[i], factor[k], addend[k]);
#elif COMPUTE_OPCODE == 7
                    value[i] += operand[k];
#elif COMPUTE_OPCODE == 8
                    value[i] ^= operand[k];
#elif COMPUTE_OPCODE == 9
                    value[i] = __funnelshift_l(operand[k], value[i], 13);
#elif COMPUTE_OPCODE == 16
                    value[i] = value[i] * a + 1013904223u;
                    folded[i] ^= __float_as_uint(__int2float_rn(static_cast<int>(value[i])));
#elif COMPUTE_OPCODE == 17
                    value[i] = fmaf(value[i], value[i], c);
                    folded[i] ^= static_cast<u32>(__float2int_rn(value[i] * 1048576.0f));
#elif COMPUTE_OPCODE == 18
                    value[i] = value[i] * a + 1013904223u;
                    folded[i] += __popc(value[i]);
#elif COMPUTE_OPCODE == 20
                    value[i] = fmaf(value[i], factor[i][k / 4], addend[i][k / 4]);
#elif COMPUTE_OPCODE == 21
                    value[i] = fmaf(value[i], factor[0][k / 4], addend[i][k / 4]);
#elif COMPUTE_OPCODE == 22
                    value[i] = fmaf(value[i], factor[0][k / 4], addend[0][k / 4]);
#elif COMPUTE_OPCODE == 19
                    value[i] = fmaf(value[i], value[i], c);
                    {
                        // Times a double constant: with wide * wide ptxas sees that one FMUL gives the same float.
                        const double wide = static_cast<double>(value[i]);
                        folded[i] ^= __float_as_uint(static_cast<float>(wide * 0.1));
                    }
#else
                    value[i] = fmaf(value[i], value[i], c);
                    float result;
#if COMPUTE_OPCODE == 10
                    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value[i]));
#elif COMPUTE_OPCODE == 11
                    asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(fabsf(value[i])));
#elif COMPUTE_OPCODE == 12
                    asm("sin.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value[i]));
#elif COMPUTE_OPCODE == 13
                    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value[i]));
#elif COMPUTE_OPCODE == 14
                    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(fabsf(value[i])));
#else
                    asm("tanh.approx.f32 %0, %1;" : "=f"(result) : "f"(value[i]));
#endif
                    folded[i] ^= __float_as_uint(result);
#endif
                }
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
#if COMPUTE_LANES < 32
        }
#endif
#endif

#if COMPUTE_OP == 8
        iterations += INDEXED_VALUES;
#elif COMPUTE_LANES < 32
        iterations += (threadIdx.x & 31u) < COMPUTE_LANES ? UNROLL * COMPUTE_CHAINS : 0;
#else
        iterations += UNROLL * COMPUTE_CHAINS;
#endif
        now = globalTimer();
        IdleForDutyWindow(activeNs, now);
    }
    while (COMPUTE_MORE(now));

    u32 result = 0;

#if COMPUTE_OP == 1 || COMPUTE_OP == 6 || COMPUTE_OP == 12
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
#elif COMPUTE_OP == 8
    for (int i = 0; i < INDEXED_VALUES; ++i)
    {
        result ^= __float_as_uint(value[i]);
    }
#elif COMPUTE_OP == 13
    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            result ^= static_cast<u32>(d[i][j]);
        }
    }
#elif COMPUTE_OP == 15 || COMPUTE_OP == 16
    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            result ^= __float_as_uint(d[i][j]);
        }
    }
#elif COMPUTE_OP == 14
    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        result ^= static_cast<u32>(__double_as_longlong(d[i][0]) ^ __double_as_longlong(d[i][1]));
    }
#elif COMPUTE_OP == 9
    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
        result ^= __float_as_uint(copies[i]);
    }
#elif COMPUTE_OP == 17
    for (int i = 0; i < COMPUTE_CHAINS; ++i)
    {
#if OPCODE_FLOAT || OPCODE_REUSE
        result ^= __float_as_uint(value[i]);
#elif OPCODE_DOUBLE
        result ^= static_cast<u32>(__double_as_longlong(value[i]));
#elif OPCODE_INTEGER
        result ^= value[i];
#else
        result ^= folded[i];
#endif
    }
#endif

    writeBlockStats(&blockCount, counts, iterations, now - start);
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = result;
}
