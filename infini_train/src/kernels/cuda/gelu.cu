#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "infini_train/include/common/common.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

constexpr int kGeluThreadsPerBlock = 256;
// Cap the grid so the vectorized kernels use a grid-stride loop on large tensors
// instead of launching an unbounded number of blocks.
constexpr size_t kGeluMaxBlocks = 4096;

// Constants of the NewGELU tanh approximation (same as PyTorch's GELU(tanh) CUDA kernel):
//   beta = sqrt(2/pi), kappa = 0.044715.
template <typename T> constexpr T kGeluBeta = T(0.7978845608028654);
template <typename T> constexpr T kGeluKappa = T(0.044715);

// All arithmetic is carried out in opmath precision (double for double, float for the
// float/half/bf16 cases), matching PyTorch. The single rounding to the storage dtype
// happens at the output store (round-to-nearest-even via common::cuda::Cast).
template <typename T> using GeluOpMath = std::conditional_t<std::is_same_v<T, double>, double, float>;

// Forward: y = 0.5 * x * (1 + tanh(beta * (x + kappa * x^3)))
template <typename T, typename OpMathT> __device__ __forceinline__ T NewGeluForwardOp(const T &x) {
    const OpMathT xf = common::cuda::Cast<OpMathT>(x);
    const OpMathT x_cube = xf * xf * xf;
    const OpMathT inner = kGeluBeta<OpMathT> * (xf + kGeluKappa<OpMathT> * x_cube);
    const OpMathT tanh_inner = common::cuda::Tanh(inner);
    return common::cuda::Cast<T>(OpMathT(0.5) * xf * (OpMathT(1) + tanh_inner));
}

// Backward: dx = dy * [0.5 * (1 + t) + 0.5 * x * (1 - t^2) * beta * (1 + 3 * kappa * x^2)]
// where t = tanh(beta * (x + kappa * x^3)). Same analytic derivative as PyTorch.
template <typename T, typename OpMathT>
__device__ __forceinline__ T NewGeluBackwardOp(const T &grad_output, const T &x) {
    const OpMathT dy = common::cuda::Cast<OpMathT>(grad_output);
    const OpMathT xf = common::cuda::Cast<OpMathT>(x);
    const OpMathT x_sq = xf * xf;
    const OpMathT inner = kGeluBeta<OpMathT> * (xf + kGeluKappa<OpMathT> * x_sq * xf);
    const OpMathT tanh_inner = common::cuda::Tanh(inner);

    const OpMathT left = OpMathT(0.5) * xf;
    const OpMathT left_derivative = OpMathT(0.5) * (OpMathT(1) + tanh_inner);
    const OpMathT tanh_derivative = OpMathT(1) - tanh_inner * tanh_inner;
    const OpMathT inner_derivative = kGeluBeta<OpMathT> * (OpMathT(1) + OpMathT(3) * kGeluKappa<OpMathT> * x_sq);
    const OpMathT right_derivative = left * tanh_derivative * inner_derivative;

    return common::cuda::Cast<T>(dy * (left_derivative + right_derivative));
}

// Aligned vector type for vectorized loads/stores (128-bit).
template <typename T, int N> struct __align__(sizeof(T) * N) aligned_vector { T val[N]; };

// Scalar grid-stride fallback for misaligned buffers.
template <typename T>
__global__ void NewGeluForwardKernel(T *__restrict__ output, const T *__restrict__ input, size_t num_elements) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements; idx += stride) {
        output[idx] = NewGeluForwardOp<T, GeluOpMath<T>>(input[idx]);
    }
}

template <typename T>
__global__ void NewGeluBackwardKernel(T *__restrict__ grad_input, const T *__restrict__ grad_output,
                                      const T *__restrict__ input, size_t num_elements) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements; idx += stride) {
        grad_input[idx] = NewGeluBackwardOp<T, GeluOpMath<T>>(grad_output[idx], input[idx]);
    }
}

// Vectorized kernels: each thread processes kElems elements through one 128-bit load and
// one 128-bit store. The trailing (< kElems) elements are handled by the first few threads.
template <typename T, int kElems>
__global__ void NewGeluForwardKernelVec(T *__restrict__ output, const T *__restrict__ input, size_t num_elements) {
    using VecT = aligned_vector<T, kElems>;
    const size_t num_vecs = num_elements / kElems;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vecs; v += stride) {
        const VecT in_vec = reinterpret_cast<const VecT *>(input)[v];
        VecT out_vec;
#pragma unroll
        for (int i = 0; i < kElems; ++i) { out_vec.val[i] = NewGeluForwardOp<T, GeluOpMath<T>>(in_vec.val[i]); }
        reinterpret_cast<VecT *>(output)[v] = out_vec;
    }

    const size_t tail_idx = num_vecs * kElems + blockIdx.x * blockDim.x + threadIdx.x;
    if (tail_idx < num_elements) {
        output[tail_idx] = NewGeluForwardOp<T, GeluOpMath<T>>(input[tail_idx]);
    }
}

template <typename T, int kElems>
__global__ void NewGeluBackwardKernelVec(T *__restrict__ grad_input, const T *__restrict__ grad_output,
                                         const T *__restrict__ input, size_t num_elements) {
    using VecT = aligned_vector<T, kElems>;
    const size_t num_vecs = num_elements / kElems;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vecs; v += stride) {
        const VecT grad_vec = reinterpret_cast<const VecT *>(grad_output)[v];
        const VecT in_vec = reinterpret_cast<const VecT *>(input)[v];
        VecT out_vec;
#pragma unroll
        for (int i = 0; i < kElems; ++i) {
            out_vec.val[i] = NewGeluBackwardOp<T, GeluOpMath<T>>(grad_vec.val[i], in_vec.val[i]);
        }
        reinterpret_cast<VecT *>(grad_input)[v] = out_vec;
    }

    const size_t tail_idx = num_vecs * kElems + blockIdx.x * blockDim.x + threadIdx.x;
    if (tail_idx < num_elements) {
        grad_input[tail_idx] = NewGeluBackwardOp<T, GeluOpMath<T>>(grad_output[tail_idx], input[tail_idx]);
    }
}

inline size_t GeluGridSize(size_t work_items) {
    return std::max<size_t>(std::min(CEIL_DIV(work_items, kGeluThreadsPerBlock), kGeluMaxBlocks), 1);
}

} // namespace

std::shared_ptr<Tensor> NewGELUForward(const std::shared_ptr<Tensor> &input) {
    CHECK(input->IsContiguous()) << "CUDA NewGELUForward: only contiguous input is supported";
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    auto device = input->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    const size_t num_elements = input->NumElements();
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        input->Dtype(),
        [=]<typename T>() {
            T *output_ptr = static_cast<T *>(output->DataPtr());
            const T *input_ptr = static_cast<const T *>(input->DataPtr());

            // 128-bit vectorized path: float -> 4 elems, half/bf16 -> 8 elems, double -> 2 elems.
            constexpr int kElems = static_cast<int>(16 / sizeof(T));
            const bool aligned = (reinterpret_cast<uintptr_t>(output_ptr) % 16 == 0)
                              && (reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0);
            if (aligned && num_elements >= static_cast<size_t>(kElems)) {
                NewGeluForwardKernelVec<T, kElems>
                    <<<static_cast<unsigned int>(GeluGridSize(num_elements / kElems)), kGeluThreadsPerBlock, 0,
                       cuda_stream>>>(output_ptr, input_ptr, num_elements);
            } else {
                NewGeluForwardKernel<T>
                    <<<static_cast<unsigned int>(GeluGridSize(num_elements)), kGeluThreadsPerBlock, 0, cuda_stream>>>(
                        output_ptr, input_ptr, num_elements);
            }
        },
        "CUDA NewGELUForward");

    return output;
}

std::shared_ptr<Tensor> NewGELUBackward(const std::shared_ptr<Tensor> &grad_output,
                                        const std::shared_ptr<Tensor> &input) {
    CHECK(input->IsContiguous() && grad_output->IsContiguous())
        << "CUDA NewGELUBackward: only contiguous tensors are supported";
    CHECK_EQ(input->NumElements(), grad_output->NumElements());
    // Forward output dtype equals input dtype, so the incoming grad matches it; convert the
    // saved input in the rare case of a dtype-mismatched graph instead of failing.
    auto input_matched
        = input->Dtype() == grad_output->Dtype() ? input : std::make_shared<Tensor>(input->To(grad_output->Dtype()));

    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());
    auto device = grad_output->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    const size_t num_elements = grad_output->NumElements();
    core::cuda::DispatchCudaFunc<INFINI_ALL_FLOATING_TYPES>(
        grad_output->Dtype(),
        [=]<typename T>() {
            T *grad_input_ptr = static_cast<T *>(grad_input->DataPtr());
            const T *grad_output_ptr = static_cast<const T *>(grad_output->DataPtr());
            const T *input_ptr = static_cast<const T *>(input_matched->DataPtr());

            constexpr int kElems = static_cast<int>(16 / sizeof(T));
            const bool aligned = (reinterpret_cast<uintptr_t>(grad_input_ptr) % 16 == 0)
                              && (reinterpret_cast<uintptr_t>(grad_output_ptr) % 16 == 0)
                              && (reinterpret_cast<uintptr_t>(input_ptr) % 16 == 0);
            if (aligned && num_elements >= static_cast<size_t>(kElems)) {
                NewGeluBackwardKernelVec<T, kElems>
                    <<<static_cast<unsigned int>(GeluGridSize(num_elements / kElems)), kGeluThreadsPerBlock, 0,
                       cuda_stream>>>(grad_input_ptr, grad_output_ptr, input_ptr, num_elements);
            } else {
                NewGeluBackwardKernel<T>
                    <<<static_cast<unsigned int>(GeluGridSize(num_elements)), kGeluThreadsPerBlock, 0, cuda_stream>>>(
                        grad_input_ptr, grad_output_ptr, input_ptr, num_elements);
            }
        },
        "CUDA NewGELUBackward");

    return grad_input;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_GELU_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_GELU_KERNEL(NewGELUForward)
REGISTER_CUDA_GELU_KERNEL(NewGELUBackward)

#undef REGISTER_CUDA_GELU_KERNEL
