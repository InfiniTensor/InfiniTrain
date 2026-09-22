#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/common/dcu/common_dcu.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/dcu/dcu_dispatch.h"
#include "infini_train/src/core/runtime/dcu/dcu_runtime_common.h"

namespace infini_train::kernels::dcu {

template <typename T>
__global__ void ScatterSrcForwardKernel(const T *__restrict__ src, const int64_t *__restrict__ indices,
                                        T *__restrict__ output, int64_t sequence, int64_t rows,
                                        int64_t hidden, int64_t total) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) { return; }
    const int64_t h = idx % hidden;
    const int64_t row = idx / hidden;
    const int64_t b = row / rows;
    output[(b * sequence + indices[row]) * hidden + h] = src[idx];
}

std::shared_ptr<Tensor> ScatterSrcForward(const std::shared_ptr<Tensor> &input, int64_t dim,
                                         const std::shared_ptr<Tensor> &indices,
                                         const std::shared_ptr<Tensor> &src) {
    CHECK_EQ(dim, 1);
    CHECK_EQ(input->Dims().size(), 3);
    CHECK_EQ(src->Dims().size(), 3);
    CHECK_EQ(indices->Dims().size(), 2);
    CHECK(indices->Dtype() == DataType::kINT64);
    CHECK(input->Dtype() == src->Dtype());
    CHECK_EQ(input->Dims()[0], src->Dims()[0]);
    CHECK_EQ(input->Dims()[0], indices->Dims()[0]);
    CHECK_EQ(input->Dims()[2], src->Dims()[2]);
    CHECK_EQ(src->Dims()[1], indices->Dims()[1]);

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    const auto device = input->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::dcu::DcuStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->hip_stream();
    HIP_CHECK(hipMemcpyAsync(output->DataPtr(), input->DataPtr(), input->SizeInBytes(), hipMemcpyDeviceToDevice, stream));
    const int64_t sequence = input->Dims()[1];
    const int64_t rows = src->Dims()[1];
    const int64_t hidden = input->Dims()[2];
    const int64_t total = src->NumElements();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    core::dcu::DispatchDcuFunc<INFINI_ALL_FLOATING_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            ScatterSrcForwardKernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T *>(src->DataPtr()), static_cast<const int64_t *>(indices->DataPtr()),
                static_cast<T *>(output->DataPtr()), sequence, rows, hidden, total);
            HIP_CHECK(hipGetLastError());
        },
        "HIP ScatterSrcForward");
    return output;
}

template <typename T>
__global__ void ScatterSrcBackwardKernel(const T *__restrict__ grad_output,
                                         const int64_t *__restrict__ indices, T *__restrict__ grad_input,
                                         T *__restrict__ grad_src, int64_t sequence, int64_t rows,
                                         int64_t hidden, int64_t total) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) { return; }
    const int64_t h = idx % hidden;
    const int64_t row = idx / hidden;
    const int64_t position = indices[row];
    const int64_t b = row / rows;
    const int64_t text_idx = (b * sequence + position) * hidden + h;
    grad_src[idx] = grad_output[text_idx];
    grad_input[text_idx] = common::dcu::Cast<T>(0.0f);
}

std::vector<std::shared_ptr<Tensor>> ScatterSrcBackward(const std::shared_ptr<Tensor> &grad_output, int64_t dim,
                                                        const std::shared_ptr<Tensor> &indices) {
    CHECK_EQ(dim, 1);
    const int64_t batch = grad_output->Dims()[0];
    const int64_t sequence = grad_output->Dims()[1];
    const int64_t hidden = grad_output->Dims()[2];
    const int64_t rows = indices->Dims()[1];
    auto grad_input
        = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());
    auto grad_src = std::make_shared<Tensor>(std::vector<int64_t>{batch, rows, hidden},
                                            grad_output->Dtype(), grad_output->GetDevice());
    const auto device = grad_output->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::dcu::DcuStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->hip_stream();
    HIP_CHECK(hipMemcpyAsync(grad_input->DataPtr(), grad_output->DataPtr(), grad_output->SizeInBytes(),
                             hipMemcpyDeviceToDevice, stream));
    const int64_t total = grad_src->NumElements();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    core::dcu::DispatchDcuFunc<INFINI_ALL_FLOATING_TYPES>(
        grad_output->Dtype(),
        [=]<typename T>() {
            ScatterSrcBackwardKernel<T><<<blocks, threads, 0, stream>>>(
                static_cast<const T *>(grad_output->DataPtr()), static_cast<const int64_t *>(indices->DataPtr()),
                static_cast<T *>(grad_input->DataPtr()), static_cast<T *>(grad_src->DataPtr()), sequence,
                rows, hidden, total);
            HIP_CHECK(hipGetLastError());
        },
        "HIP ScatterSrcBackward");
    return {grad_input, grad_src};
}

} // namespace infini_train::kernels::dcu

#define REGISTER_HIP_SCATTER_SRC_KERNEL(kernel_name)                                                                  \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kDCU, kernel_name, infini_train::kernels::dcu::kernel_name)

REGISTER_HIP_SCATTER_SRC_KERNEL(ScatterSrcForward)
REGISTER_HIP_SCATTER_SRC_KERNEL(ScatterSrcBackward)

#undef REGISTER_HIP_SCATTER_SRC_KERNEL
