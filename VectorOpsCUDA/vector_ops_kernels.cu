#include "vector_ops_kernels.cuh"

// vector add: c = a + b
__global__ // tells complier this function is a kernel, run on GPU
void kernel_vec_add(const float* a, const float* b, float* c, int n) { // pointers to device memory for const inputs a and b, pointer to device memory for output c, size n dim of vectors
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                   // calculate global thread index (using block index and thread index within block)

    if (idx < n) {                                                     // only run thread if index is in bounds of total vector dim, avoid wasting threads
        c[idx] = a[idx] + b[idx];                                      // adding vectors by element
    }
}

// vector sub: c = a - b
__global__
void kernel_vec_sub(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] - b[idx];
    }
}

// vector mul: c = a * b (elementwise)
__global__
void kernel_vec_mul(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

// scalar mul: c = alpha * a
__global__
void kernel_scalar_mul(const float* a, float alpha, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = alpha * a[idx];
    }
}

// saxpy: y = alpha * x + y (y updated in place)
__global__
void kernel_saxpy(const float* x, float* y, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

// dot product partial: per block partial sums written to 'partial'
__global__
void kernel_dot_partial(const float* a, const float* b, float* partial, int n) {
    extern __shared__ float sdata[];                 // shared data for tree reduction sum

    int tid = threadIdx.x;                           // thread id in block
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // global element index

    float val = 0.0f;                                // value for each multiplication result
    if (idx < n) {
        val = a[idx] * b[idx];
    }

    sdata[tid] = val;
    __syncthreads();                                 // make sure all values are written

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {   // tree reduction within block, result ends up in sdata[0]
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {                                  // write total block sum into one element of 'partial'
        partial[blockIdx.x] = sdata[0];
    }
}