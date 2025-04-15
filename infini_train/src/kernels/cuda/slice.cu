#include "infini_train/include/kernels/cuda/slice.h"

#include <cmath>
#include <functional>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
std::shared_ptr<Tensor> SliceForward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &starts,
                                     const std::vector<int64_t> &ends, const std::vector<int64_t> &steps) {
    return nullptr;
}

std::shared_ptr<Tensor> SliceBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                      const std::vector<int64_t> &starts, const std::vector<int64_t> &ends,
                                      const std::vector<int64_t> &steps) {
    return nullptr;
}
} // namespace infini_train::kernels::cuda
