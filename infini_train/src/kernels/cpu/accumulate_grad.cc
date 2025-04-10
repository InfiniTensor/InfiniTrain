#include "infini_train/include/kernels/cpu/accumulate_grad.h"

#include <memory>

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        reinterpret_cast<float *>(tensor->DataPtr())[idx]
            += rate * reinterpret_cast<const float *>(gradient->DataPtr())[idx];
    }
}
} // namespace infini_train::kernels::cpu
