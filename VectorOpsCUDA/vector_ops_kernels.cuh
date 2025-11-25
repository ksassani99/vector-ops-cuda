#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// vector add: c = a + b
__global__
void kernel_vec_add(const float* a, const float* b, float* c, int n);

// vector sub: c = a - b
__global__
void kernel_vec_sub(const float* a, const float* b, float* c, int n);

// vector mul: c = a * b (elementwise)
__global__
void kernel_vec_mul(const float* a, const float* b, float* c, int n);

// scalar mul: c = alpha * a
__global__
void kernel_scalar_mul(const float* a, float alpha, float* c, int n);

// saxpy: y = alpha * x + y (y updated in place)
__global__
void kernel_saxpy(const float* x, float* y, float alpha, int n);

// dot product partial: per block partial sums written to 'partial'
__global__
void kernel_dot_partial(const float* a, const float* b, float* partial, int n);