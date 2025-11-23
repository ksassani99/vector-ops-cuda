#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>

#include <vector>

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

// tells complier this function is a kernel, run on GPU
__global__ 
void kernel_vec_add(const float* a, const float* b, float* c, int n) { // pointers to device memory for const inputs a and b, pointer device memory for output c, size n dim of vectors
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                   // calculate global thread index (using block indexz and thread index within block)

    if (idx < n) {                                                     // only run thread if index is in bounds of total vector dim, avoid wasting threads
        c[idx] = a[idx] + b[idx];                                      // adding vectors element-wise
    }
}

int main() {
	const int N = 1 << 16;                                // vector size 2^16
    const int blockSize = 256;                            // number of threads per block
	const int gridSize = (N + blockSize - 1) / blockSize; // calculate number of blocks needed for dim N vectors

	printf("Vector ADD: N = %d\n", N);

    // ** host vectors (CPU memory, input vectors a and b live on CPU memory, output vector c CPU retrieves results from GPU)
    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);

    for (int i = 0; i < N; ++i) { // initialize input vectors for testing
		h_a[i] = 1.0f * i;        // convert vector indices to floats
		h_b[i] = 2.0f * i;        // convert vector indices to floats * 2 (diff values for testing)
    }

	// ** device vectors (GPU memory)
	float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

	// allocate device (GPU) memory for input and output vectors
	CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float))); // address of GPU pointer to set, size in bytes, CUDA_CHECK wrap checks for errors and exits if any
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));

    // copy input data from host (CPU) to device (GPU)
	CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));  // destination GPU pointer, source CPU pointer, size in bytes, direction of copy, CUDA_CHECK wrap checks for errors and exits if any
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));

	// ** launch kernel with calculated grid and block dimensions
    dim3 block(blockSize); // define block dimension
    dim3 grid(gridSize);   // define grid dimension

	// launch kernel on GPU
    kernel_vec_add<<<grid, block>>>(d_a, d_b, d_c, N); // launch vec add kernel passing device pointers and size N
	
	CUDA_CHECK(cudaGetLastError());      // check for any errors launching kernel
	CUDA_CHECK(cudaDeviceSynchronize()); // synchronize device (force CPU to wait for kernel on GPU to finish)
    
	CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost)); // copy result vector from device (GPU) to host (CPU))

	// print first 20 results
    printf("First 20 results:\n");
    for (int i = 0; i < 20; ++i) {
        printf("i=%d, a=%f, b=%f, c=%f\n", i, h_a[i], h_b[i], h_c[i]);
    }

	// cleanup device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
	CUDA_CHECK(cudaFree(d_c));
	CUDA_CHECK(cudaDeviceReset()); // reset device

	return 0;
}