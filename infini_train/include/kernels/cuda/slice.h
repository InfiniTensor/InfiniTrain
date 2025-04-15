#pragma once

#include <memory>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
std::shared_ptr<Tensor> SliceForward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &start,
                                     const std::vector<int64_t> &end, const std::vector<int64_t> &step);
std::shared_ptr<Tensor> SliceBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &output,
                                      const std::vector<int64_t> &start, const std::vector<int64_t> &end,
                                      const std::vector<int64_t> &step);
} // namespace infini_train::kernels::cuda
