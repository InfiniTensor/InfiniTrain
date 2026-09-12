#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>

#include <curand_kernel.h>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/generator_impl.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_generator_impl.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;

template <typename RandomT> __device__ RandomT UniformSample(curandStatePhilox4_32_10_t *state) {
    if constexpr (std::is_same_v<RandomT, double>) {
        double sample;
        do {
            const double2 values = curand_uniform2_double(state);
            sample = 1.0 - values.x;
            // Subtraction can round a tiny positive draw to 1.0.
        } while (sample >= 1.0);
        return sample;
    } else {
        return static_cast<float>(curand(state)) * 0x1p-32f;
    }
}

template <typename StorageT, typename RandomT>
__global__ void DropoutForwardKernel(StorageT *output, uint8_t *mask, const StorageT *input, int64_t n, RandomT p,
                                     uint64_t seed, uint64_t subsequence) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    if (p == static_cast<RandomT>(0)) {
        mask[index] = 1;
        output[index] = input[index];
        return;
    }
    if (p == static_cast<RandomT>(1)) {
        mask[index] = 0;
        output[index] = common::cuda::Cast<StorageT>(0.0f);
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, subsequence + static_cast<uint64_t>(index), 0, &state);
    const bool keep = UniformSample<RandomT>(&state) >= p;
    const RandomT scale = static_cast<RandomT>(1) / (static_cast<RandomT>(1) - p);
    mask[index] = keep ? 1 : 0;
    output[index] = keep ? common::cuda::Cast<StorageT>(common::cuda::Cast<RandomT>(input[index]) * scale)
                         : common::cuda::Cast<StorageT>(0.0f);
}

template <typename StorageT, typename RandomT>
__global__ void DropoutBackwardKernel(StorageT *grad_input, const StorageT *grad_output, const uint8_t *mask, int64_t n,
                                      RandomT p) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const RandomT scale = p == static_cast<RandomT>(1) ? static_cast<RandomT>(0)
                                                       : static_cast<RandomT>(1) / (static_cast<RandomT>(1) - p);
    grad_input[index] = mask[index]
                          ? common::cuda::Cast<StorageT>(common::cuda::Cast<RandomT>(grad_output[index]) * scale)
                          : common::cuda::Cast<StorageT>(0.0f);
}

const core::cuda::CudaStream *GetCudaStream(const Device &device) {
    return dynamic_cast<const core::cuda::CudaStream *>(core::GetDeviceGuardImpl(device.type())->GetStream(device));
}

} // namespace

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
DropoutForward(const std::shared_ptr<Tensor> input, double p, const std::optional<Generator> gen) {
    const Device device = input->GetDevice();
    CHECK(device.IsCUDA());
    CHECK(IsFloatingPointDType(input->Dtype())) << "Dropout supports floating-point tensors only";
    CHECK_GE(p, 0.0) << "dropout probability has to be between 0 and 1, but got " << p;
    CHECK_LE(p, 1.0) << "dropout probability has to be between 0 and 1, but got " << p;
    core::DeviceGuard guard(device);

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), device);
    auto mask = std::make_shared<Tensor>(input->Dims(), DataType::kUINT8, device);

    const int64_t n = input->NumElements();
    if (n == 0) {
        return {output, mask};
    }

    uint64_t seed = 0;
    uint64_t subsequence = 0;
    if (p > 0.0 && p < 1.0) {
        auto &cuda_generator = GetGeneratorOrDefault<core::cuda::CUDAGeneratorImpl>(
            gen, core::cuda::GetDefaultCudaGenerator(device.index()), device);
        std::lock_guard<std::mutex> lock(cuda_generator.mutex_);
        seed = cuda_generator.current_seed();
        subsequence = cuda_generator.ReservePhiloxSubsequence(static_cast<uint64_t>(n));
    }

    const int blocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
    const auto *stream = GetCudaStream(device);
    core::cuda::DispatchCudaFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        input->Dtype(),
        [&]<typename StorageT>() {
            using RandomT = std::conditional_t<std::is_same_v<StorageT, double>, double, float>;
            DropoutForwardKernel<StorageT, RandomT><<<blocks, kThreadsPerBlock, 0, stream->cuda_stream()>>>(
                static_cast<StorageT *>(output->DataPtr()), static_cast<uint8_t *>(mask->DataPtr()),
                static_cast<const StorageT *>(input->DataPtr()), n, static_cast<RandomT>(p), seed, subsequence);
        },
        "CUDA dropout forward");
    CUDA_CHECK(cudaGetLastError());
    return {output, mask};
}

std::shared_ptr<Tensor> DropoutBackward(const std::shared_ptr<Tensor> grad_output, const std::shared_ptr<Tensor> mask,
                                        double p) {
    const Device device = grad_output->GetDevice();
    CHECK(device.IsCUDA());
    core::DeviceGuard guard(device);

    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), device);

    const int64_t n = grad_output->NumElements();
    if (n == 0) {
        return grad_input;
    }

    const int blocks = static_cast<int>((n + kThreadsPerBlock - 1) / kThreadsPerBlock);
    const auto *stream = GetCudaStream(device);
    core::cuda::DispatchCudaFunc<DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64>(
        grad_output->Dtype(),
        [&]<typename StorageT>() {
            using RandomT = std::conditional_t<std::is_same_v<StorageT, double>, double, float>;
            DropoutBackwardKernel<StorageT, RandomT><<<blocks, kThreadsPerBlock, 0, stream->cuda_stream()>>>(
                static_cast<StorageT *>(grad_input->DataPtr()), static_cast<const StorageT *>(grad_output->DataPtr()),
                static_cast<const uint8_t *>(mask->DataPtr()), n, static_cast<RandomT>(p));
        },
        "CUDA dropout backward");
    CUDA_CHECK(cudaGetLastError());
    return grad_input;
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_DROPOUT_KERNEL(kernel_name)                                                                      \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_DROPOUT_KERNEL(DropoutForward)
REGISTER_CUDA_DROPOUT_KERNEL(DropoutBackward)

#undef REGISTER_CUDA_DROPOUT_KERNEL
