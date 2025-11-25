#pragma once

#include <vector>

// CPU reference implementations
void  cpu_vec_add(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c);
void  cpu_vec_sub(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c);
void  cpu_vec_mul(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& c);
void  cpu_scalar_mul(const std::vector<float>& a, float alpha, std::vector<float>& c);
void  cpu_saxpy(const std::vector<float>& x, std::vector<float>& y, float alpha);
float cpu_dot(const std::vector<float>& a, const std::vector<float>& b);

// helper: max absolute error between two vectors
float max_abs_error(const std::vector<float>& x, const std::vector<float>& y);