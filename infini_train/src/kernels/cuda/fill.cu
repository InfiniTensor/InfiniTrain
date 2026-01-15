#include <cstddef>
#include <memory>

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

template <typename T> __global__ void FillKernel(T *data, T value, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

// TODO(dcj): refactor Fill kernel with elementwise template
void Fill(std::shared_ptr<Tensor> tensor, void *value_ptr) {
    const int num_tokens = tensor->NumElements();
    const int threads_per_block = 256;
    const int num_blocks = (num_tokens + threads_per_block - 1) / threads_per_block;
    const auto *cuda_device = dynamic_cast<const CudaDevice *>(tensor->GetDevice());

    DispatchFunc<INFINI_ALL_TYPES>(
        tensor->Dtype(),
        [=]<typename T>() {
            FillKernel<T><<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                static_cast<T *>(tensor->DataPtr()), *(static_cast<T *>(value_ptr)), tensor->NumElements());
        },
        "CUDA Fill");
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_FILL_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_FILL_KERNEL(Fill)

#undef REGISTER_CUDA_FILL_KERNEL
