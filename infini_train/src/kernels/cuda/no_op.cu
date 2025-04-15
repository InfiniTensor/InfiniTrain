#include "infini_train/include/kernels/cuda/no_op.h"

#include <memory>
#include <numeric>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
std::shared_ptr<Tensor> NoOpForward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &dims) {
    return nullptr;
}

std::shared_ptr<Tensor> NoOpBackward(const std::vector<int64_t> &dims, const std::shared_ptr<Tensor> &grad_output) {
    return nullptr;
}
} // namespace infini_train::kernels::cuda
