# CUDA-Based Vector Operations
Parallel vector math operations implemented in CUDA, with CPU reference models for correctness validation. Supports elementwise addition, subtraction, multiplication, scalar multiplication, SAXPY, and dot product using shared-memory tree reduction.

## Project Structure
```
VectorOpsCUDA/
├── main.cu                 # Entry point, host memory mgmt, kernel launch, validation
├── vector_ops_cpu.cpp      # CPU reference implementations
├── vector_ops_cpu.hpp      # Function declarations
├── vector_ops_kernels.cu   # GPU kernels
├── vector_ops_kernels.cuh  # Kernel declarations
```

## Supported Operations
| Operation  | Formula       | Description                       |
| ---------- | ------------- | --------------------------------- |
| vec_add    | c = a + b     | Elementwise vector addition       |
| vec_sub    | c = a - b     | Elementwise vector subtraction    |
| vec_mul    | c = a × b     | Elementwise multiplication        |
| scalar_mul | c = α × a     | Scale vector by constant          |
| saxpy      | y = α × x + y | AXPY (y modified in place)        |
| dot        | ∑(aᵢ × bᵢ)    | Dot product using block reduction |

## Design Overview
### CPU Reference Model
All operations are implemented using standard for-loop traversal over input vectors. Used as a correctness baseline to validate GPU results.

### GPU Kernels
Each thread processes one vector element using:
```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```
Bounds are checked to prevent wasted threads.

### Dot Product – Shared Memory Tree Reduction
1. Each thread computes:
   `val = a[idx] * b[idx]`
2. Writes to shared memory:
   `sdata[tid] = val`
3. Reduction inside block:
```cpp
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```
4. Thread 0 stores per-block result into global memory
5. Host accumulates final sum

### Error Checking
A CUDA_CHECK macro wraps CUDA API calls. On failure, it prints the source file, line number, and error string from `cudaGetErrorString()` and terminates.

## Testing Methodology
For each operation:
1. Host allocates input vectors (`h_a`, `h_b`) and initializes values
2. CPU reference implementation runs and stores output in `h_ref`
3. Data is copied to GPU device memory
4. Kernel launched with `<<<grid, block>>>` configuration
5. Results copied back to host
6. `max_abs_error()` identifies maximum deviation between CPU and GPU results
7. First five output entries printed for visual inspection

## Example Output
```
tests started: N = 65536

vec_add max error: 0.000000e+00
i=0, a=0.000, b=0.000, c_gpu=0.000
i=0, a=0.000, b=0.000, c_cpu=0.000

vec_mul max error: 0.000000e+00
i=1, a=1.000, b=2.000, c_gpu=2.000
i=1, a=1.000, b=2.000, c_cpu=2.000

scalar_mul max error: 0.000000e+00
i=3, a=3.000, alpha=3.500, c_gpu=10.500
i=3, a=3.000, alpha=3.500, c_cpu=10.500

saxpy max error: 0.000000e+00
i=2, a=2.000, b=4.000, alpha=3.500, c_gpu=11.000
i=2, a=2.000, b=4.000, alpha=3.500, c_cpu=11.000

dot abs error: 1.192093e-07
dot rel error: 2.465190e-09
dot_gpu=2814479360.000000
dot_cpu=2814479359.500000
```

## Build & Run (Windows, Visual Studio + CUDA Toolkit)
### Visual Studio Setup
1. Create a new **CUDA Runtime Project**
2. Replace default files with project files
3. Build Configuration: **x64 → Debug/Release**
4. Ensure `.cu` files are compiled with CUDA C/C++
5. Build and run directly in Visual Studio

## Tools Used
* CUDA Toolkit
* Visual Studio
* C++ STL

## Key Concepts Demonstrated
* Host-device memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`)
* Grid-block-thread hierarchy in CUDA
* Shared memory and synchronization (`__syncthreads`)
* Reduction algorithm (tree-based)
* Floating-point precision and error analysis
* Cross-validation: GPU vs CPU
