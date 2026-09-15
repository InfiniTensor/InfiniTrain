#include <cstddef>
#include <memory>

#include "infini_train/include/common/common.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/dtype_dispatch.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

template <typename T> __global__ void ReLUForwardKernel(T *output, const T *input, size_t num_elements, size_t offset) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < num_elements) {
        // Strict `<` keeps NaN and -0 (both compare false), matching clamp_min(x, 0).
        output[idx] = input[idx] < T(0) ? T(0) : input[idx];
    }
}

template <typename T>
__global__ void ReLUBackwardKernel(T *grad_input, const T *input, const T *grad_output, size_t num_elements,
                                   size_t offset) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < num_elements) {
        // `<=` matches threshold_backward: NaN (false) passes grad through, 0/-0/negative mask to 0.
        grad_input[idx] = input[idx] <= T(0) ? T(0) : grad_output[idx];
    }
}

inline size_t ChooseBlockSize(size_t num_elements) {
    if (num_elements < 1024) {
        return 64;
    }
    if (num_elements < 65536) {
        return 128;
    }
    if (num_elements < 1048576) {
        return 256;
    }
    return 512;
}

cudaStream_t CurrentStream(const std::shared_ptr<Tensor> &tensor) {
    auto device = tensor->GetDevice();
    return dynamic_cast<infini_train::core::cuda::CudaStream *>(
               infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
        ->cuda_stream();
}

template <typename T>
void ReLUForwardImpl(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &output) {
    const size_t num_elements = output->NumElements();
    if (num_elements == 0) {
        return;
    }
    cudaStream_t stream = CurrentStream(output);
    T *out_ptr = static_cast<T *>(output->DataPtr());
    const T *in_ptr = static_cast<const T *>(input->DataPtr());

    dim3 block(ChooseBlockSize(num_elements));
    dim3 grid(CEIL_DIV(num_elements, block.x));
    const size_t step = grid.x * block.x;
    for (size_t offset = 0; offset < num_elements; offset += step) {
        ReLUForwardKernel<T><<<grid, block, 0, stream>>>(out_ptr, in_ptr, num_elements, offset);
    }
}

template <typename T>
void ReLUBackwardImpl(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &grad_output,
                      const std::shared_ptr<Tensor> &grad_input) {
    const size_t num_elements = grad_input->NumElements();
    if (num_elements == 0) {
        return;
    }
    cudaStream_t stream = CurrentStream(grad_input);
    T *grad_input_ptr = static_cast<T *>(grad_input->DataPtr());
    const T *input_ptr = static_cast<const T *>(input->DataPtr());
    const T *grad_output_ptr = static_cast<const T *>(grad_output->DataPtr());

    dim3 block(ChooseBlockSize(num_elements));
    dim3 grid(CEIL_DIV(num_elements, block.x));
    const size_t step = grid.x * block.x;
    for (size_t offset = 0; offset < num_elements; offset += step) {
        ReLUBackwardKernel<T>
            <<<grid, block, 0, stream>>>(grad_input_ptr, input_ptr, grad_output_ptr, num_elements, offset);
    }
}

} // namespace

std::shared_ptr<Tensor> ReLUForward(const std::shared_ptr<Tensor> &input) {
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    DISPATCH(input->Dtype(), ReLUForwardImpl<float>(input, output);, DataType::kFLOAT32)
    return output;
}

std::shared_ptr<Tensor> ReLUBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &grad_output) {
    auto grad_input = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    DISPATCH(input->Dtype(), ReLUBackwardImpl<float>(input, grad_output, grad_input);, DataType::kFLOAT32)
    return grad_input;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_RELU_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_RELU_KERNEL(ReLUForward)
REGISTER_CUDA_RELU_KERNEL(ReLUBackward)

#undef REGISTER_CUDA_RELU_KERNEL
