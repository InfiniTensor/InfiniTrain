#include <algorithm>
#include <cstddef>

#include "infini_train/include/common/common.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {
using namespace infini_train::common::cuda;

template <typename T>
__global__ void SwiGLUForwardKernel(T *__restrict__ output, const T *__restrict__ input, int64_t hidden,
                                    size_t num_elements) {
    const size_t grid_stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < num_elements;
         idx += grid_stride) {
        const size_t row = idx / hidden;
        const size_t col = idx % hidden;
        const size_t input_base = row * 2 * hidden;
        const T up = input[input_base + col];
        const T gate = input[input_base + hidden + col];
        output[idx] = Mul(up, Mul(gate, Sigmoid(gate)));
    }
}

template <typename T, typename InputT, typename GradT>
__global__ void SwiGLUBackwardKernel(T *__restrict__ grad_input, const InputT *__restrict__ input,
                                     const GradT *__restrict__ grad_output, int64_t hidden, size_t num_elements) {
    const size_t grid_stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < num_elements;
         idx += grid_stride) {
        const size_t row = idx / hidden;
        const size_t col = idx % hidden;
        const size_t input_base = row * 2 * hidden;
        const T up = Cast<T>(input[input_base + col]);
        const T gate = Cast<T>(input[input_base + hidden + col]);
        const T grad = Cast<T>(grad_output[idx]);
        const T sigmoid = Sigmoid(gate);
        grad_input[input_base + col] = Mul(grad, Mul(gate, sigmoid));
        grad_input[input_base + hidden + col]
            = Mul(grad, Mul(up, Mul(sigmoid, Add(T(1), Mul(gate, Sub(T(1), sigmoid))))));
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
} // namespace

std::shared_ptr<Tensor> SwiGLUForward(const std::shared_ptr<Tensor> &input) {
    CHECK(input->IsContiguous());
    auto output_dims = input->Dims();
    CHECK_GT(output_dims.size(), 0);
    CHECK_EQ(output_dims.back() % 2, 0);
    const int64_t hidden = output_dims.back() / 2;
    CHECK_GT(hidden, 0);
    output_dims.back() = hidden;

    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());
    auto device = output->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();
    const size_t num_elements = output->NumElements();
    const dim3 block(ChooseBlockSize(num_elements));
    const dim3 grid(std::min(CEIL_DIV(num_elements, block.x), static_cast<size_t>(65535)));

    switch (input->Dtype()) {
    case DataType::kFLOAT32:
        SwiGLUForwardKernel<<<grid, block, 0, stream>>>(static_cast<float *>(output->DataPtr()),
                                                        static_cast<const float *>(input->DataPtr()), hidden,
                                                        num_elements);
        break;
    case DataType::kBFLOAT16:
        SwiGLUForwardKernel<<<grid, block, 0, stream>>>(static_cast<nv_bfloat16 *>(output->DataPtr()),
                                                        static_cast<const nv_bfloat16 *>(input->DataPtr()), hidden,
                                                        num_elements);
        break;
    default:
        LOG_LOC(FATAL, "CUDA SwiGLUForward: unsupported data type");
    }
    return output;
}

std::shared_ptr<Tensor> SwiGLUBackward(const std::shared_ptr<Tensor> &input,
                                       const std::shared_ptr<Tensor> &grad_output) {
    CHECK(input->IsContiguous());
    CHECK(grad_output->IsContiguous());
    CHECK_GT(input->Dims().size(), 0);
    const int64_t hidden = input->Dims().back() / 2;
    CHECK_GT(hidden, 0);
    CHECK_EQ(grad_output->NumElements() * 2, input->NumElements());

    const DataType output_dtype = PromoteDataTypes(input->Dtype(), grad_output->Dtype());
    auto grad_input = std::make_shared<Tensor>(input->Dims(), output_dtype, input->GetDevice());
    auto device = input->GetDevice();
    const auto &stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                             infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                             ->cuda_stream();
    const size_t num_elements = grad_output->NumElements();
    const dim3 block(ChooseBlockSize(num_elements));
    const dim3 grid(std::min(CEIL_DIV(num_elements, block.x), static_cast<size_t>(65535)));

    if (input->Dtype() == DataType::kFLOAT32 && grad_output->Dtype() == DataType::kFLOAT32) {
        SwiGLUBackwardKernel<<<grid, block, 0, stream>>>(
            static_cast<float *>(grad_input->DataPtr()), static_cast<const float *>(input->DataPtr()),
            static_cast<const float *>(grad_output->DataPtr()), hidden, num_elements);
    } else if (input->Dtype() == DataType::kBFLOAT16 && grad_output->Dtype() == DataType::kBFLOAT16) {
        SwiGLUBackwardKernel<<<grid, block, 0, stream>>>(
            static_cast<nv_bfloat16 *>(grad_input->DataPtr()), static_cast<const nv_bfloat16 *>(input->DataPtr()),
            static_cast<const nv_bfloat16 *>(grad_output->DataPtr()), hidden, num_elements);
    } else if (input->Dtype() == DataType::kBFLOAT16 && grad_output->Dtype() == DataType::kFLOAT32) {
        SwiGLUBackwardKernel<<<grid, block, 0, stream>>>(
            static_cast<float *>(grad_input->DataPtr()), static_cast<const nv_bfloat16 *>(input->DataPtr()),
            static_cast<const float *>(grad_output->DataPtr()), hidden, num_elements);
    } else if (input->Dtype() == DataType::kFLOAT32 && grad_output->Dtype() == DataType::kBFLOAT16) {
        SwiGLUBackwardKernel<<<grid, block, 0, stream>>>(
            static_cast<float *>(grad_input->DataPtr()), static_cast<const float *>(input->DataPtr()),
            static_cast<const nv_bfloat16 *>(grad_output->DataPtr()), hidden, num_elements);
    } else {
        LOG_LOC(FATAL, "CUDA SwiGLUBackward: unsupported data type combination");
    }
    return grad_input;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_SWIGLU_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_SWIGLU_KERNEL(SwiGLUForward)
REGISTER_CUDA_SWIGLU_KERNEL(SwiGLUBackward)

#undef REGISTER_CUDA_SWIGLU_KERNEL
