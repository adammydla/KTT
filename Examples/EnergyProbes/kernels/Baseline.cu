// Baseline and reference probes: probeSleep (the sleep baseline every energy is measured against) and probeSpin (heavy
// instruction issue without memory traffic, a reference only). No parameters of their own.

// Baseline: every SM holds resident warps that sleep most of the time, so the GPU is awake at its current clock but almost
// nothing is issued. Power above this level is attributed to the probes. With activeSms below the SM count, only that many
// SMs hold sleeping warps (the power of an awake but idle SM). nanosleep needs Volta (sm_70) or newer; older GPUs fall
// back to polling the timer, which is the same as probeSpin.
extern "C" __global__ void probeSleep(u32* __restrict__ sink, u64* __restrict__ counts, const u32 activeSms,
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

    u64 iterations = 0;
    const u64 start = globalTimer();
    u64 now = start;

    do
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        asm volatile("nanosleep.u32 %0;" : : "r"(100000u)); // up to ~100 us per call
#endif
        ++iterations;
        now = globalTimer();
    }
    while (now - start < durationNs);

    writeBlockStats(&blockCount, counts, iterations, now - start);
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = static_cast<u32>(iterations);
}

// Reference with heavy instruction issue: every thread polls the timer in a tight loop, no memory traffic. Not a baseline:
// in the first test runs it drew more power than the memory probes, because those wait for data and issue less.
extern "C" __global__ void probeSpin(u32* __restrict__ sink, u64* __restrict__ counts, const u64 durationNs)
{
    __shared__ u64 blockCount;
    countLaunch(counts);

    if (threadIdx.x == 0)
    {
        blockCount = 0;
    }

    __syncthreads();

    u64 iterations = 0;
    const u64 start = globalTimer();
    u64 now = start;

    do
    {
        ++iterations;
        now = globalTimer();
    }
    while (now - start < durationNs);

    writeBlockStats(&blockCount, counts, iterations, now - start);
    sink[blockIdx.x * BLOCK_SIZE + threadIdx.x] = static_cast<u32>(iterations);
}
