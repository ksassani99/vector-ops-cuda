#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <vector>
#include <cmath>

#include "vector_ops_kernels.cuh"
#include "vector_ops_cpu.hpp"

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

int main() {
	const int N = 1 << 16;                                // vector size 2^16
    const int blockSize = 256;                            // number of threads per block
	const int gridSize = (N + blockSize - 1) / blockSize; // calculate number of blocks needed for dim N vectors

	printf("tests started: N = %d\n", N);
    
    std::vector<float> h_a(N); // host vectors (CPU memory, input vectors a and b live on CPU memory, output vector c, CPU retrieves results from GPU)
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);

    std::vector<float> h_ref(N);            // reference result vector
    float alpha = 3.5f;                     // test constant

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
    // run vec_add cpu reference
    cpu_vec_add(h_a, h_b, h_ref);
    
    // run vec_add gpu kernel
    kernel_vec_add<<<grid, block>>>(d_a, d_b, d_c, N);                                  // launch kernel on GPU
    CUDA_CHECK(cudaGetLastError());                                                     // check for any errors launching kernel
    CUDA_CHECK(cudaDeviceSynchronize());                                                // synchronize device (force CPU to wait for kernel on GPU to finish)

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost)); // copy result vector from device (GPU) to host (CPU))

    // print max error and results
    printf("\nvec_add max error: %e\n", max_abs_error(h_c, h_ref));

    printf("First 5 results:\n"); // print first 5 results
    for (int i = 0; i < 5; ++i) {
        printf("i=%d, a=%f, b=%f, c_gpu=%f\n", i, h_a[i], h_b[i], h_c[i]);   // gpu result
        printf("i=%d, a=%f, b=%f, c_cpu=%f\n", i, h_a[i], h_b[i], h_ref[i]); // cpu result
    }

    // ====== vec_sub testing ======
    // run vec_sub cpu reference
    cpu_vec_sub(h_a, h_b, h_ref);

    // run vec_sub gpu kernel
    kernel_vec_sub<<<grid, block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // print max error and results
    printf("\nvec_sub max error: %e\n", max_abs_error(h_c, h_ref));

    printf("First 5 results:\n");
    for (int i = 0; i < 5; ++i) {
        printf("i=%d, a=%f, b=%f, c_gpu=%f\n", i, h_a[i], h_b[i], h_c[i]);
        printf("i=%d, a=%f, b=%f, c_cpu=%f\n", i, h_a[i], h_b[i], h_ref[i]);
    }

    // ====== vec_mul testing ======
    // run vec_mul cpu reference
    cpu_vec_mul(h_a, h_b, h_ref);

    // run vec_mul gpu kernel
    kernel_vec_mul<<<grid, block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // print max error and results
    printf("\nvec_mul max error: %e\n", max_abs_error(h_c, h_ref));

    printf("First 5 results:\n");
    for (int i = 0; i < 5; ++i) {
        printf("i=%d, a=%f, b=%f, c_gpu=%f\n", i, h_a[i], h_b[i], h_c[i]);
        printf("i=%d, a=%f, b=%f, c_cpu=%f\n", i, h_a[i], h_b[i], h_ref[i]);
    }

    // ====== scalar_mul testing ======
    // run scalar_mul cpu reference
    cpu_scalar_mul(h_a, alpha, h_ref);

    // run scalar_mul gpu kernel
    kernel_scalar_mul<<<grid, block>>>(d_a, alpha, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // print max error and results
    printf("\nscalar_mul max error: %e\n", max_abs_error(h_c, h_ref));

    printf("First 5 results:\n");
    for (int i = 0; i < 5; ++i) {
        printf("i=%d, a=%f, b=%f, alpha=%f, c_gpu=%f\n", i, h_a[i], h_b[i], alpha, h_c[i]);
        printf("i=%d, a=%f, b=%f, alpha=%f, c_cpu=%f\n", i, h_a[i], h_b[i], alpha, h_ref[i]);
    }

    // ====== saxpy testing ======
    std::vector<float> h_y_cpu = h_b; // create additional h_y vectors to prevent overriding h_b
    std::vector<float> h_y_gpu = h_b;

    // run saxpy cpu reference
    cpu_saxpy(h_a, h_y_cpu, alpha);

    CUDA_CHECK(cudaMemcpy(d_c, h_y_gpu.data(), N * sizeof(float), cudaMemcpyHostToDevice)); // copy h_y to d_b

    // run saxpy gpu kernel
    kernel_saxpy<<<grid, block>>>(d_a, d_c, alpha, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_y_gpu.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost)); // copy d_b updated result back to h_y

    // print max error and results
    printf("\nsaxpy max error: %e\n", max_abs_error(h_y_gpu, h_y_cpu)); // do max error and results of h_y values

    printf("First 5 results:\n");
    for (int i = 0; i < 5; ++i) {
        printf("i=%d, a=%f, b=%f, alpha=%f, c_gpu=%f\n", i, h_a[i], h_b[i], alpha, h_y_gpu[i]);
        printf("i=%d, a=%f, b=%f, alpha=%f, c_cpu=%f\n", i, h_a[i], h_b[i], alpha, h_y_cpu[i]);
    }

    // ====== dot testing ======
    std::vector<float> h_partial(gridSize); // cpu partial for dot results, one entry for each block dot result
    float* d_partial = nullptr;             // gpu partial val for dot results

    // run dot cpu reference, store value
    float dot_res_cpu = cpu_dot(h_a, h_b);

    CUDA_CHECK(cudaMalloc(&d_partial, gridSize * sizeof(float))); // allocate memory to new d_partial val for dot block results

    size_t sharedMemSize = blockSize * sizeof(float); // calculate size of shared memory per block for vec dot results in block

    // run scalar_mul gpu kernel
    kernel_dot_partial<<<grid, block, sharedMemSize>>>(d_a, d_b, d_partial, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    float acc = 0.0f;                    // accumulated sum value
    for (int i = 0; i < gridSize; ++i) { // sum over h_partial to return total dot product value
        acc += h_partial[i];
    }

    float dot_res_gpu = acc;

    // print max error and results
    printf("\ndot abs error: %e\n", std::fabs(dot_res_gpu - dot_res_cpu));
    printf("dot rel error: %e\n", std::fabs(dot_res_gpu - dot_res_cpu) / std::fabs(dot_res_cpu));

    printf("Single value results:\n");
    printf("vec a, vec b, dot_gpu=%f\n", dot_res_gpu);
    printf("vec a, vec b, dot_cpu=%f\n", dot_res_cpu);

	// cleanup device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
	CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaFree(d_partial));
	CUDA_CHECK(cudaDeviceReset()); // reset device

	return 0;
}