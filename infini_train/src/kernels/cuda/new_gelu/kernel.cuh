#pragma once

#include <cstddef>

namespace infini_train::kernels::cuda {

__global__ void NewGELUForwardKernel(const float* input, size_t count,
                                     float* output, float* saved_tanh);

__global__ void NewGELUBackwardKernel(const float* input,
                                      const float* saved_tanh,
                                      const float* gradient, size_t count,
                                      float* grad_input);

}  // namespace infini_train::kernels::cuda
