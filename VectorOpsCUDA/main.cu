#include <stdio.h>

#include <vector>
#include <cuda_runtime.h>

// CUDA error check macro, backslashes extend macro past one line, prints error message, exits program with failure code
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err__ = (call);                                 \
        if (err__ != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)