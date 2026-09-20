#include <algorithm>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {
__global__ void ReLUKernel(const float *x, const float *dy, float *out, int64_t count) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        out[i] = x[i] <= 0.0f ? 0.0f : (dy ? dy[i] : x[i]);
    }
}

std::shared_ptr<Tensor> ApplyReLU(const std::shared_ptr<Tensor> &x, const std::shared_ptr<Tensor> &dy) {
    const auto device = x->GetDevice();
    core::DeviceGuard guard(device);
    auto out = std::make_shared<Tensor>(x->Dims(), x->Dtype(), device);
    const int64_t count = x->NumElements();
    if (!count) {
        return out;
    }
    auto stream = dynamic_cast<core::cuda::CudaStream *>(core::GetDeviceGuardImpl(device.type())->GetStream(device))
                      ->cuda_stream();
    const int blocks = static_cast<int>(std::min<int64_t>((count + 255) / 256, 65535));
    ReLUKernel<<<blocks, 256, 0, stream>>>(static_cast<const float *>(x->DataPtr()),
                                           dy ? static_cast<const float *>(dy->DataPtr()) : nullptr,
                                           static_cast<float *>(out->DataPtr()), count);
    CUDA_CHECK(cudaGetLastError());
    return out;
}
} // namespace

std::shared_ptr<Tensor> ReLUForward(const std::shared_ptr<Tensor> &x) { return ApplyReLU(x, nullptr); }
std::shared_ptr<Tensor> ReLUBackward(const std::shared_ptr<Tensor> &x, const std::shared_ptr<Tensor> &dy) {
    return ApplyReLU(x, dy);
}
} // namespace infini_train::kernels::cuda

REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, ReLUForward, infini_train::kernels::cuda::ReLUForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, ReLUBackward, infini_train::kernels::cuda::ReLUBackward)
