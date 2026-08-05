#include <cstddef>
#include <memory>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/dcu/dcu_dispatch.h"
#include "infini_train/src/core/runtime/dcu/dcu_runtime_common.h"

namespace infini_train::kernels::dcu {

template <typename T> __global__ void FillKernel(T *data, T value, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

// TODO(dcj): refactor Fill kernel with elementwise template
void Fill(std::shared_ptr<Tensor> tensor, Scalar scalar) {
    const int num_tokens = tensor->NumElements();
    const int threads_per_block = 256;
    const int num_blocks = (num_tokens + threads_per_block - 1) / threads_per_block;
    auto device = tensor->GetDevice();
    const auto &hip_stream = dynamic_cast<infini_train::core::dcu::DcuStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->hip_stream();

    core::dcu::DispatchDcuFunc<INFINI_ALL_TYPES>(
        tensor->Dtype(),
        [=]<typename T>() {
            const T casted_value = scalar.to<T>();
            FillKernel<T><<<num_blocks, threads_per_block, 0, hip_stream>>>(static_cast<T *>(tensor->DataPtr()),
                                                                             casted_value, tensor->NumElements());
        },
        "HIP Fill");
}
} // namespace infini_train::kernels::dcu

#define REGISTER_HIP_FILL_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kDCU, kernel_name, infini_train::kernels::dcu::kernel_name)

REGISTER_HIP_FILL_KERNEL(Fill)

#undef REGISTER_HIP_FILL_KERNEL
