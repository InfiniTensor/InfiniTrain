#include "infini_train/include/optimizer.h"

#include <cstddef>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
#ifdef USE_CUDA
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)
#endif

Optimizer::Optimizer(const std::vector<Tensor *> &params) : params_(params) {}

void Optimizer::ZeroGrad() {
    for (auto *param : params_) { param->ZeroGrad(); }
}

namespace optimizers {
namespace {
#ifdef USE_CUDA
__global__ void StepKernel(const float *gradient_data, int num_elements, float learning_rate, float *param_data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_elements) {
        param_data[index] -= learning_rate * gradient_data[index];
    }
}
#endif
} // namespace

SGD::SGD(const std::vector<Tensor *> &params, float learning_rate) : Optimizer(params), learning_rate_(learning_rate) {}

void SGD::Step() {
    for (auto *param : params_) {
        switch (param->GetDevice().Type()) {
        case DeviceType::kCPU: {
            for (size_t i = 0; i < param->NumElements(); ++i) {
                reinterpret_cast<float *>(param->DataPtr())[i]
                    -= learning_rate_ * reinterpret_cast<const float *>(param->Gradient()->DataPtr())[i];
            }
            break;
        }
#ifdef USE_CUDA
        case DeviceType::kCUDA: {
            int threads_per_block = 256;
            int num_blocks = (param->NumElements() + threads_per_block - 1) / threads_per_block;
            StepKernel<<<num_blocks, threads_per_block>>>(reinterpret_cast<const float *>(param->Gradient()->DataPtr()),
                                                          param->NumElements(), learning_rate_,
                                                          reinterpret_cast<float *>(param->DataPtr()));
            // FIXME: does this synchronize nessisary?
            CUDA_CHECK(cudaDeviceSynchronize());
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
