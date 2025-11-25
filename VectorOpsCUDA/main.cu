#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#include <vector>
#include <cmath>

// CUDA error check macro, backslashes extend macro past one line, prints error message, exits program with failure code
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err__ = (call);                                 \
                                                                    \
        if (err__ != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// ** GPU kernels
__global__ // tells complier this function is a kernel, run on GPU
void kernel_vec_add(const float* a, const float* b, float* c, int n) { // kernel for vec add, pointers to device memory for const inputs a and b, pointer device memory for output c, size n dim of vectors
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                   // calculate global thread index (using block index and thread index within block)

    if (idx < n) {                                                     // only run thread if index is in bounds of total vector dim, avoid wasting threads
        c[idx] = a[idx] + b[idx];                                      // adding vectors element-wise
    }
}

__global__
void kernel_vec_sub(const float* a, const float* b, float* c, int n) { // kernel for vec sub
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                   // calculate global thread index

    if (idx < n) {                                                     // only run if thread is within bounds
        c[idx] = a[idx] - b[idx];                                      // subtracting vectors element-wise
    }
}

__global__
void kernel_vec_mul(const float* a, const float* b, float* c, int n) { // kernel for vec mul
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                   // calculate global thread index

    if (idx < n) {                                                     // only run if thread is within bounds
        c[idx] = a[idx] * b[idx];                                      // multiplying vectors element-wise
    }
}

__global__
void kernel_scalar_mul(const float* a, const float alpha , float* c, int n) { // kernel for scalar mul
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          // calculate global thread index

    if (idx < n) {                                                            // only run if thread is within bounds
        c[idx] = alpha * a[idx];                                              // scalar mul each element
    }
}

__global__
void kernel_saxpy(const float* x, float* y, const float alpha, int n) { // kernel for vec mul
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                    // calculate global thread index

    if (idx < n) {                                                      // only run if thread is within bounds
        y[idx] = alpha * x[idx] + y[idx];                               // updates y as output after doing SAXPY
    }
}

__global__
void kernel_dot_partial(const float* a, const float* b, float* partial, int n) { // kernel for dot prod (does partial product per block)
    extern __shared__ float sdata[]; // shared data for tree reduction sum
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f; // val for mul results to sum over
    if (idx < n) {
        val = a[idx] * b[idx];
    }

    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) { // write total block sum into one part in partial sum
        partial[blockIdx.x] = sdata[tid];
    }
}

// ** CPU kernels (golden model, reference model)
void cpu_vec_add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) { // CPU kernel for vec add (no size n input b/c & is reference to full object
    std::size_t n = a.size();

    for (size_t i = 0; i < n; ++i) { // not parallel, uses for loop to do operations on vector entries
        c[i] = a[i] + b[i];
    }
}

void cpu_vec_sub(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) { // CPU kernel for vec sub
    std::size_t n = a.size();

    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

void cpu_vec_mul(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) { // CPU kernel for vec mul
    std::size_t n = a.size();

    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

void cpu_scalar_mul(const std::vector<float>& a, const float alpha, std::vector<float>& c) { // CPU kernel for scalar mul
    std::size_t n = a.size();

    for (size_t i = 0; i < n; ++i) {
        c[i] = alpha * a[i];
    }
}

void cpu_saxpy(const std::vector<float>& x, std::vector<float>& y, const float alpha) { // CPU kernel for saxpy
    std::size_t n = x.size();

    for (size_t i = 0; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

float cpu_dot(const std::vector<float>& a, const std::vector<float>& b) { // CPU kernel for dot prod
    double acc = 0.0; // precise accumulated value (double for precise)

    std::size_t n = a.size();

    for (size_t i = 0; i < n; ++i) {
        acc += (double)a[i] * (double)b[i];
    }

    return (float)acc;
}

float max_abs_error(const std::vector<float>& x, const std::vector<float>& y) { // max error between GPU and CPU output vectors
    float max_err = 0.0f;
    std::size_t n = x.size();

    for (size_t i = 0; i < n; ++i) {
        float diff = std::fabs(x[i] - y[i]);
        if (diff > max_err) max_err = diff;
    }

    return max_err;
}

int main() {
	const int N = 1 << 16;                                // vector size 2^16
    const int blockSize = 256;                            // number of threads per block
	const int gridSize = (N + blockSize - 1) / blockSize; // calculate number of blocks needed for dim N vectors

	printf("Vector ADD: N = %d\n", N);
    
    std::vector<float> h_a(N); // host vectors (CPU memory, input vectors a and b live on CPU memory, output vector c, CPU retrieves results from GPU)
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);

    std::vector<float> h_ref(N); // reference result vector

    for (int i = 0; i < N; ++i) { // initialize input vectors for testing
		h_a[i] = 1.0f * i;        // convert vector indices to floats
		h_b[i] = 2.0f * i;        // convert vector indices to floats * 2 (diff values for testing)
    }

	float* d_a = nullptr; // device vectors (GPU memory)
    float* d_b = nullptr;
    float* d_c = nullptr;

	// allocate device (GPU) memory for input and output vectors
	CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float))); // address of GPU pointer to set, size in bytes, CUDA_CHECK wrap error checks
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // copy input data from host (CPU) to device (GPU)
	CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));  // destination GPU pointer, source CPU pointer, size in bytes, direction of copy, CUDA_CHECK wrap error checks
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(blockSize); // define block dimension
    dim3 grid(gridSize);   // define grid dimension

    // ====== vec_add testing ======
    cpu_vec_add(h_a, h_b, h_ref); // run reference on CPU

    kernel_vec_add<<<grid, block>>>(d_a, d_b, d_c, N); // launch kernel on GPU
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost)); // copy result vector from device (GPU) to host (CPU))
    CUDA_CHECK(cudaGetLastError());      // check for any errors launching kernel
    CUDA_CHECK(cudaDeviceSynchronize()); // synchronize device (force CPU to wait for kernel on GPU to finish)

    printf("vec_add max error: %e\n", max_abs_error(h_c, h_ref));

    // print first 10 results
    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("i=%d, a=%f, b=%f, c_gpu=%f\n", i, h_a[i], h_b[i], h_c[i]);
        printf("i=%d, a=%f, b=%f, c_cpu=%f\n", i, h_a[i], h_b[i], h_ref[i]);
    }

    // ====== vec_sub testing ======
    cpu_vec_sub(h_a, h_b, h_ref); // run reference on CPU

    kernel_vec_sub<<<grid, block>>>(d_a, d_b, d_c, N); // launch kernel on GPU
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost)); // copy result vector from device (GPU) to host (CPU))
    CUDA_CHECK(cudaGetLastError());      // check for any errors launching kernel
    CUDA_CHECK(cudaDeviceSynchronize()); // synchronize device (force CPU to wait for kernel on GPU to finish)

    printf("vec_sub max error: %e\n", max_abs_error(h_c, h_ref));

    // print first 10 results
    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("i=%d, a=%f, b=%f, c_gpu=%f\n", i, h_a[i], h_b[i], h_c[i]);
        printf("i=%d, a=%f, b=%f, c_cpu=%f\n", i, h_a[i], h_b[i], h_ref[i]);
    }

    // ====== vec_mul testing ======
    cpu_vec_mul(h_a, h_b, h_ref); // run reference on CPU

    kernel_vec_mul<<<grid, block>>>(d_a, d_b, d_c, N); // launch kernel on GPU
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost)); // copy result vector from device (GPU) to host (CPU))
    CUDA_CHECK(cudaGetLastError());      // check for any errors launching kernel
    CUDA_CHECK(cudaDeviceSynchronize()); // synchronize device (force CPU to wait for kernel on GPU to finish)

    printf("vec_mul max error: %e\n", max_abs_error(h_c, h_ref));

	// print first 10 results
    printf("First 10 results:\n");
    for (int i = 0; i < 10; ++i) {
        printf("i=%d, a=%f, b=%f, c_gpu=%f\n", i, h_a[i], h_b[i], h_c[i]);
        printf("i=%d, a=%f, b=%f, c_cpu=%f\n", i, h_a[i], h_b[i], h_ref[i]);
    }

	// cleanup device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
	CUDA_CHECK(cudaFree(d_c));
	CUDA_CHECK(cudaDeviceReset()); // reset device

	return 0;
}