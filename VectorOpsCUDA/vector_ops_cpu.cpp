#include "vector_ops_cpu.hpp"

#include <cmath>

void cpu_vec_add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) {
    std::size_t n = a.size();        // no size n input b/c & is reference to full object

    for (size_t i = 0; i < n; ++i) { // not parallel, uses for loop to do operations on vector entries
        c[i] = a[i] + b[i];
    }
}

void cpu_vec_sub(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) {
    std::size_t n = a.size();

    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] - b[i];
    }
}

void cpu_vec_mul(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c) {
    std::size_t n = a.size();

    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

void cpu_scalar_mul(const std::vector<float>& a, const float alpha, std::vector<float>& c) {
    std::size_t n = a.size();

    for (size_t i = 0; i < n; ++i) {
        c[i] = alpha * a[i];
    }
}

void cpu_saxpy(const std::vector<float>& x, std::vector<float>& y, const float alpha) {
    std::size_t n = x.size();

    for (size_t i = 0; i < n; ++i) {
        y[i] = alpha * x[i] + y[i];
    }
}

float cpu_dot(const std::vector<float>& a, const std::vector<float>& b) {
    float acc = 0.0f; // accumulated value for dot product sum

    std::size_t n = a.size();

    for (size_t i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }

    return acc;
}

float max_abs_error(const std::vector<float>& x, const std::vector<float>& y) { // max error between GPU and CPU output vectors
    float max_err = 0.0f;                                                       // max error value
    std::size_t n = x.size();

    for (size_t i = 0; i < n; ++i) {                                            // from every entry of output vectors, find max error for both vectors
        float diff = std::fabs(x[i] - y[i]);                                    // float abs value
        if (diff > max_err) max_err = diff;                                     // replace max if new max error
    }

    return max_err;                                                             // return float max error
}