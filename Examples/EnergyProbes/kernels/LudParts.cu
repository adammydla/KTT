// lud_internal of Rodinia 3.1 (lud/cuda/lud_kernel.cu, https://github.com/yuhc/gpu-rodinia; University of Virginia,
// see its LICENSE) and parts of it, for finding which part the model misprices (--validate ludinternal, 2026-11-18).
// lud is the largest miss of the whole-application validation (docs/APPLICATIONS.md); its data values and its broadcast
// shared loads were ruled out. This kernel is a diagnosis, not a held-out score: nothing is fitted on it.
//
// Parameters:
//   LUD_OFFSET  the step's offset (lud runs offset = 0, 16, ..., matrix_dim - 32): the grid has
//               (matrix_dim - LUD_OFFSET) / 16 - 1 blocks per dimension
//   LUD_PART    0 = lud_internal as Rodinia has it
//               1 = without its read-modify-write of m (the sum is stored only if it is NaN, which it never is)
//               2 = without its inner loop (the sum of the first product only; the read-modify-write stays)
//
// The matrix is random in [0, 1) and lud_internal only updates rows and columns from offset + 16 on, from the rows and
// columns before them, so repeated launches do not change what it reads (the updated values fall by about 4 per launch).

#define LUD_BLOCK 16

extern "C" __global__ void ludInternal(float* m, int matrix_dim)
{
    __shared__ float peri_row[LUD_BLOCK][LUD_BLOCK];
    __shared__ float peri_col[LUD_BLOCK][LUD_BLOCK];

    const int offset = LUD_OFFSET;
    float sum;

    int global_row_id = offset + (blockIdx.y + 1) * LUD_BLOCK;
    int global_col_id = offset + (blockIdx.x + 1) * LUD_BLOCK;

    peri_row[threadIdx.y][threadIdx.x] = m[(offset + threadIdx.y) * matrix_dim + global_col_id + threadIdx.x];
    peri_col[threadIdx.y][threadIdx.x] = m[(global_row_id + threadIdx.y) * matrix_dim + offset + threadIdx.x];

    __syncthreads();

    sum = 0;
#if LUD_PART == 2
    sum = peri_col[threadIdx.y][0] * peri_row[0][threadIdx.x];
#else
    for (int i = 0; i < LUD_BLOCK; i++)
        sum += peri_col[threadIdx.y][i] * peri_row[i][threadIdx.x];
#endif
#if LUD_PART == 1
    if (sum != sum)
        m[(global_row_id + threadIdx.y) * matrix_dim + global_col_id + threadIdx.x] = sum;
#else
    m[(global_row_id + threadIdx.y) * matrix_dim + global_col_id + threadIdx.x] -= sum;
#endif
}
