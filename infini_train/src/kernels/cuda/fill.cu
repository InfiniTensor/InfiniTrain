#include "glog/logging.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "infini_train/include/common/cuda/common_cuda.cuh"

namespace infini_train::kernels::cuda {
void Fill(std::shared_ptr<Tensor> tensor, void *value_ptr) {
    DispatchFunc<INFINI_ALL_TYPES>(
        tensor->Dtype(),
        [=]<typename T>() {
            thrust::device_ptr<T> dev_ptr(reinterpret_cast<T *>(tensor->DataPtr()));
            thrust::fill(thrust::cuda::par.on(0), dev_ptr, dev_ptr + tensor->NumElements(),
                         *(static_cast<T *>(value_ptr)));
        },
        "Fill");
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_FILL_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_FILL_KERNEL(Fill)

#undef REGISTER_CUDA_FILL_KERNEL
