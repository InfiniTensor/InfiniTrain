#include <algorithm>

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"
#include "infini_train/src/kernels/common/conv2d.h"

namespace infini_train::kernels::cuda {
namespace {
// One thread reduces one output element. No atomics or host staging are needed.
template <int Mode>
__global__ void ConvKernel(float *out, int64_t count, const float *x, const float *w, const float *b, const float *dy,
                           Conv2dShape s) {
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        if constexpr (Mode == 0) {
            out[i] = Conv2dForwardAt(i, x, w, b, s);
        }
        if constexpr (Mode == 1) {
            out[i] = Conv2dInputGradAt(i, w, dy, s);
        }
        if constexpr (Mode == 2) {
            out[i] = Conv2dWeightGradAt(i, x, dy, s);
        }
        if constexpr (Mode == 3) {
            out[i] = Conv2dBiasGradAt(i, dy, s);
        }
    }
}

Conv2dShape Shape(const Tensor &x, const Tensor &w, int64_t stride, int64_t padding) {
    const auto &a = x.Dims();
    const auto &b = w.Dims();
    return {a[0],
            a[1],
            a[2],
            a[3],
            b[0],
            b[2],
            b[3],
            (a[2] + 2 * padding - b[2]) / stride + 1,
            (a[3] + 2 * padding - b[3]) / stride + 1,
            stride,
            padding};
}

template <int Mode>
void Launch(const std::shared_ptr<Tensor> &out, const float *x, const float *w, const float *b, const float *dy,
            Conv2dShape s) {
    const auto device = out->GetDevice();
    core::DeviceGuard guard(device);
    auto stream = dynamic_cast<core::cuda::CudaStream *>(core::GetDeviceGuardImpl(device.type())->GetStream(device))
                      ->cuda_stream();
    const int64_t count = out->NumElements();
    if (!count) {
        return;
    }
    const int blocks = static_cast<int>(std::min<int64_t>((count + 255) / 256, 65535));
    ConvKernel<Mode><<<blocks, 256, 0, stream>>>(static_cast<float *>(out->DataPtr()), count, x, w, b, dy, s);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

std::shared_ptr<Tensor> Conv2dForward(const std::shared_ptr<Tensor> &x, const std::shared_ptr<Tensor> &w,
                                      const std::shared_ptr<Tensor> &b, int64_t stride, int64_t padding) {
    const auto s = Shape(*x, *w, stride, padding);
    auto y = std::make_shared<Tensor>(std::vector<int64_t>{s.n, s.co, s.oh, s.ow}, x->Dtype(), x->GetDevice());
    Launch<0>(y, static_cast<const float *>(x->DataPtr()), static_cast<const float *>(w->DataPtr()),
              b ? static_cast<const float *>(b->DataPtr()) : nullptr, nullptr, s);
    return y;
}

std::vector<std::shared_ptr<Tensor>> Conv2dBackward(const std::shared_ptr<Tensor> &x, const std::shared_ptr<Tensor> &w,
                                                    const std::shared_ptr<Tensor> &dy, int64_t stride, int64_t padding,
                                                    bool need_x, bool need_w, bool need_b) {
    const auto s = Shape(*x, *w, stride, padding);
    const auto xp = static_cast<const float *>(x->DataPtr());
    const auto wp = static_cast<const float *>(w->DataPtr());
    const auto gp = static_cast<const float *>(dy->DataPtr());
    std::vector<std::shared_ptr<Tensor>> grads(3);
    if (need_x) {
        grads[0] = std::make_shared<Tensor>(x->Dims(), x->Dtype(), x->GetDevice());
        Launch<1>(grads[0], nullptr, wp, nullptr, gp, s);
    }
    if (need_w) {
        grads[1] = std::make_shared<Tensor>(w->Dims(), w->Dtype(), w->GetDevice());
        Launch<2>(grads[1], xp, nullptr, nullptr, gp, s);
    }
    if (need_b) {
        grads[2] = std::make_shared<Tensor>(std::vector<int64_t>{s.co}, x->Dtype(), x->GetDevice());
        Launch<3>(grads[2], nullptr, nullptr, nullptr, gp, s);
    }
    return grads;
}
} // namespace infini_train::kernels::cuda

REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, Conv2dForward, infini_train::kernels::cuda::Conv2dForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, Conv2dBackward, infini_train::kernels::cuda::Conv2dBackward)
