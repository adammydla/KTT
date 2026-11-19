// The fourth held-out set (--validate heldout4, 2026-11-17): more kernels of PolyBench/GPU 1.0 (Grauer-Gray, Killian,
// Searles, Cavazos; https://github.com/sgrauerg/polybenchGpu, CUDA/), written before the model change of 2026-11-17 and
// measured before it, so that the new model is scored on kernels that did not guide it (heldout3 did: its gesummv and
// conv3d biases asked for the new probes). The computations are PolyBench's; the tuning parameters are:
//   BLOCK_X, BLOCK_Y  the thread block
//   UNROLL            unroll factor of the inner loop (1 = none)
//   ACC_REGISTER      1: accumulate in a register and write once; 0: accumulate in global memory, as PolyBench does
//   ITEMS_Y           output rows per thread
// Do not change this file: the frozen record (docs/PROBES.md) names its sha256.

#ifndef UNROLL
#define UNROLL 1
#endif

#define PRAGMA4(x) _Pragma(#x)
#define UNROLL_LOOP4_N(n) PRAGMA4(unroll n)
#define UNROLL_LOOP4 UNROLL_LOOP4_N(UNROLL)

// atax, first kernel: tmp = A x, one thread per row (PolyBench: NX = NY = 4096; here 8192).
extern "C" __global__ void atax1(const int nx, const int ny, const float* A, const float* x, float* tmp)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nx)
    {
#if ACC_REGISTER
        float sum = 0.0f;
        UNROLL_LOOP4
        for (int j = 0; j < ny; j++)
        {
            sum += A[i * ny + j] * x[j];
        }
        tmp[i] = sum;
#else
        tmp[i] = 0.0f;
        UNROLL_LOOP4
        for (int j = 0; j < ny; j++)
        {
            tmp[i] += A[i * ny + j] * x[j];
        }
#endif
    }
}

// mvt, second kernel: x2 += A^T y2, one thread per column (PolyBench: N = 4096; here 8192).
extern "C" __global__ void mvt2(const int n, const float* a, float* x2, const float* y2)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
#if ACC_REGISTER
        float sum = x2[i];
        UNROLL_LOOP4
        for (int j = 0; j < n; j++)
        {
            sum += a[j * n + i] * y2[j];
        }
        x2[i] = sum;
#else
        UNROLL_LOOP4
        for (int j = 0; j < n; j++)
        {
            x2[i] += a[j * n + i] * y2[j];
        }
#endif
    }
}

// jacobi-2d, one sweep: B = 5-point average of A; ITEMS_Y rows per thread (PolyBench: N = 1000; here 8192).
extern "C" __global__ void jacobi2d(const int n, const float* A, float* B)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int first = (blockIdx.y * blockDim.y + threadIdx.y) * ITEMS_Y;

    UNROLL_LOOP4
    for (int item = 0; item < ITEMS_Y; item++)
    {
        const int i = first + item;

        if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1)
        {
            B[i * n + j] = 0.2f * (A[i * n + j] + A[i * n + (j - 1)] + A[i * n + (1 + j)] + A[(1 + i) * n + j]
                + A[(i - 1) * n + j]);
        }
    }
}

// fdtd-2d, third kernel: hz -= 0.7 (curl of ex, ey), read-modify-write of hz (PolyBench: NX = NY = 2048; here 8192).
extern "C" __global__ void fdtd2d(const int nx, const int ny, const float* ex, const float* ey, float* hz)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny)
    {
        hz[i * ny + j] = hz[i * ny + j] - 0.7f * (ex[i * (ny + 1) + (j + 1)] - ex[i * (ny + 1) + j] + ey[(i + 1) * ny + j]
            - ey[i * ny + j]);
    }
}

// correlation, standard deviation kernel: std of every column, one thread per column, with PolyBench's sqrt and
// small-value guard (PolyBench: M = N = 2048; here 8192 columns of 2048 rows).
extern "C" __global__ void corrstd(const int m, const int n, const float floatN, const float eps, const float* mean,
    float* stddev, const float* data)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < m)
    {
#if ACC_REGISTER
        const float mu = mean[j];
        float sum = 0.0f;
        UNROLL_LOOP4
        for (int i = 0; i < n; i++)
        {
            const float d = data[i * m + j] - mu;
            sum += d * d;
        }
        sum = sqrtf(sum / floatN);
        stddev[j] = sum <= eps ? 1.0f : sum;
#else
        stddev[j] = 0.0f;
        UNROLL_LOOP4
        for (int i = 0; i < n; i++)
        {
            stddev[j] += (data[i * m + j] - mean[j]) * (data[i * m + j] - mean[j]);
        }
        stddev[j] /= floatN;
        stddev[j] = sqrtf(stddev[j]);
        stddev[j] = stddev[j] <= eps ? 1.0f : stddev[j];
#endif
    }
}
