// The third held-out set (--validate heldout3, NEXT_STEPS_PLAN M6, 2026-11-14): kernels of PolyBench/GPU 1.0
// (Grauer-Gray, Killian, Searles, Cavazos; https://github.com/sgrauerg/polybenchGpu, CUDA/), each with a small tuning
// space added here. The computations are PolyBench's; the tuning parameters are:
//   BLOCK_X, BLOCK_Y  the thread block (PolyBench: 256 x 1 or 32 x 8)
//   UNROLL            unroll factor of the inner loop (1 = none)
//   ACC_REGISTER      1: accumulate in a register and write once; 0: accumulate in global memory, as PolyBench does
//   ITEMS_Y           output rows per thread (2-D convolution)
// One kernel per computation; none of them was looked at while the energy model was designed.

#ifndef UNROLL
#define UNROLL 1
#endif

// One more macro level, so that UNROLL is expanded before it is turned into the pragma's text.
#define PRAGMA(x) _Pragma(#x)
#define UNROLL_LOOP_N(n) PRAGMA(unroll n)
#define UNROLL_LOOP UNROLL_LOOP_N(UNROLL)

// gesummv: y = alpha * A x + beta * B x, one thread per row (PolyBench: N = 4096).
extern "C" __global__ void gesummv(const int n, const float alpha, const float beta, const float* A, const float* B,
    float* tmp, const float* x, float* y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
#if ACC_REGISTER
        float t = 0.0f;
        float s = 0.0f;
        UNROLL_LOOP
        for (int j = 0; j < n; j++)
        {
            t += A[i * n + j] * x[j];
            s += B[i * n + j] * x[j];
        }
        tmp[i] = t;
        y[i] = alpha * t + beta * s;
#else
        tmp[i] = 0.0f;
        y[i] = 0.0f;
        UNROLL_LOOP
        for (int j = 0; j < n; j++)
        {
            tmp[i] += A[i * n + j] * x[j];
            y[i] += B[i * n + j] * x[j];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
#endif
    }
}

// syrk: C = alpha * A A^T + beta * C (PolyBench: NI = NJ = 1024).
extern "C" __global__ void syrk(const int ni, const int nj, const float alpha, const float beta, const float* a, float* c)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ni && j < ni)
    {
#if ACC_REGISTER
        float sum = c[i * ni + j] * beta;
        UNROLL_LOOP
        for (int k = 0; k < nj; k++)
        {
            sum += alpha * a[i * nj + k] * a[j * nj + k];
        }
        c[i * ni + j] = sum;
#else
        c[i * ni + j] *= beta;
        UNROLL_LOOP
        for (int k = 0; k < nj; k++)
        {
            c[i * ni + j] += alpha * a[i * nj + k] * a[j * nj + k];
        }
#endif
    }
}

// syr2k: C = alpha * A B^T + alpha * B A^T + beta * C (PolyBench: NI = NJ = 1024).
extern "C" __global__ void syr2k(const int ni, const int nj, const float alpha, const float beta, const float* a,
    const float* b, float* c)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < ni && j < ni)
    {
#if ACC_REGISTER
        float sum = c[i * ni + j] * beta;
        UNROLL_LOOP
        for (int k = 0; k < nj; k++)
        {
            sum += alpha * a[i * nj + k] * b[j * nj + k] + alpha * b[i * nj + k] * a[j * nj + k];
        }
        c[i * ni + j] = sum;
#else
        c[i * ni + j] *= beta;
        UNROLL_LOOP
        for (int k = 0; k < nj; k++)
        {
            c[i * ni + j] += alpha * a[i * nj + k] * b[j * nj + k] + alpha * b[i * nj + k] * a[j * nj + k];
        }
#endif
    }
}

// 2-D convolution with PolyBench's 3 x 3 coefficients; ITEMS_Y rows per thread (PolyBench: NI = NJ = 4096, one row).
extern "C" __global__ void conv2d(const int ni, const int nj, const float* A, float* B)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int first = (blockIdx.y * blockDim.y + threadIdx.y) * ITEMS_Y;
    const float c11 = +0.2f, c21 = +0.5f, c31 = -0.8f;
    const float c12 = -0.3f, c22 = +0.6f, c32 = -0.9f;
    const float c13 = +0.4f, c23 = +0.7f, c33 = +0.10f;

    UNROLL_LOOP
    for (int item = 0; item < ITEMS_Y; item++)
    {
        const int i = first + item;

        if (i < ni - 1 && j < nj - 1 && i > 0 && j > 0)
        {
            B[i * nj + j] = c11 * A[(i - 1) * nj + (j - 1)] + c21 * A[(i - 1) * nj + j] + c31 * A[(i - 1) * nj + (j + 1)]
                + c12 * A[i * nj + (j - 1)] + c22 * A[i * nj + j] + c32 * A[i * nj + (j + 1)]
                + c13 * A[(i + 1) * nj + (j - 1)] + c23 * A[(i + 1) * nj + j] + c33 * A[(i + 1) * nj + (j + 1)];
        }
    }
}

// 3-D convolution with PolyBench's coefficients and access pattern. PolyBench launches one kernel per plane i; here
// every thread walks the planes itself, so that one launch is the whole computation (NI = NJ = NK = 256).
extern "C" __global__ void conv3d(const int ni, const int nj, const int nk, const float* A, float* B)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const float c11 = +2, c21 = +5, c31 = -8;
    const float c12 = -3, c22 = +6, c32 = -9;
    const float c13 = +4, c23 = +7, c33 = +10;

    if (j < nj - 1 && k < nk - 1 && j > 0 && k > 0)
    {
        UNROLL_LOOP
        for (int i = 1; i < ni - 1; i++)
        {
            B[i * (nk * nj) + j * nk + k] = c11 * A[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)]
                + c13 * A[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)] + c21 * A[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)]
                + c23 * A[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)] + c31 * A[(i - 1) * (nk * nj) + (j - 1) * nk + (k - 1)]
                + c33 * A[(i + 1) * (nk * nj) + (j - 1) * nk + (k - 1)] + c12 * A[i * (nk * nj) + (j - 1) * nk + k]
                + c22 * A[i * (nk * nj) + j * nk + k] + c32 * A[i * (nk * nj) + (j + 1) * nk + k]
                + c11 * A[(i - 1) * (nk * nj) + (j - 1) * nk + (k + 1)] + c13 * A[(i + 1) * (nk * nj) + (j - 1) * nk + (k + 1)]
                + c21 * A[(i - 1) * (nk * nj) + j * nk + (k + 1)] + c23 * A[(i + 1) * (nk * nj) + j * nk + (k + 1)]
                + c31 * A[(i - 1) * (nk * nj) + (j + 1) * nk + (k + 1)] + c33 * A[(i + 1) * (nk * nj) + (j + 1) * nk + (k + 1)];
        }
    }
}
