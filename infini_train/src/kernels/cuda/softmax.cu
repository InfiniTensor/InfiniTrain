#include "infini_train/include/kernels/cuda/softmax.h"

#include <cmath>
#include <cstdint>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

std::shared_ptr<Tensor> SoftmaxForward(const std::shared_ptr<Tensor> &input, int64_t dim) { return nullptr; }

std::shared_ptr<Tensor> SoftmaxBackward(const std::shared_ptr<Tensor> &grad_output,
                                        const std::shared_ptr<Tensor> &output, int64_t dim) {
    return nullptr;
}
} // namespace infini_train::kernels::cuda
