#include "infini_train/include/optimizer.h"

#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/kernels/cpu/accumulate_grad.h"
#include "infini_train/include/tensor.h"

#ifdef USE_CUDA
#include "infini_train/include/kernels/cuda/accumulate_grad.h"
#endif

namespace infini_train {
Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>> &params) : params_(params) {}

void Optimizer::ZeroGrad() {
    for (auto param : params_) { param->ZeroGrad(); }
}

namespace optimizers {

SGD::SGD(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate)
    : Optimizer(params), learning_rate_(learning_rate) {}

void SGD::Step() {
    for (auto param : params_) {
        switch (param->GetDevice().Type()) {
        case DeviceType::kCPU: {
            kernels::cpu::AccumulateGrad(param->grad(), -learning_rate_, param);
            break;
        }
#ifdef USE_CUDA
        case DeviceType::kCUDA: {
            kernels::cuda::AccumulateGrad(param->grad(), -learning_rate_, param);
            break;
        }
#endif
        default:
            LOG(FATAL) << "Unsupported device type: " << param->GetDevice();
            break;
        }
    }
}
} // namespace optimizers
} // namespace infini_train
