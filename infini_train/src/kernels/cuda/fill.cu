#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/scalar.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cuda/cuda_dispatch.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernels::cuda {

template <typename T> __global__ void FillKernel(T *data, T value, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = value;
    }
}

// A Scalar is memset-compatible iff its stored bit pattern is all zeros, which for every
// dtype InfiniTrain supports (bool / intN / uintN / fp16 / bf16 / fp32 / fp64) coincides
// with the numeric value +0. Note that -0.0 has the sign bit set and is NOT all-zero bits,
// so we deliberately reject it here even though it compares == 0.0 numerically; callers
// that pass -0.0 fall through to FillKernel and get the correct bit pattern.
static bool IsZeroBitsScalar(const Scalar &s) {
    switch (s.kind) {
    case Scalar::Kind::kBool:
    case Scalar::Kind::kUInt64:
        return s.u == 0;
    case Scalar::Kind::kInt64:
        return s.i == 0;
    case Scalar::Kind::kDouble: {
        uint64_t bits = 0;
        std::memcpy(&bits, &s.d, sizeof(bits));
        return bits == 0;
    }
    default:
        return false;
    }
}

// TODO(dcj): refactor Fill kernel with elementwise template
void Fill(std::shared_ptr<Tensor> tensor, Scalar scalar) {
    const size_t num_elements = tensor->NumElements();
    if (num_elements == 0) { return; }

    auto device = tensor->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    // Fast path: Fill(0) on a contiguous tensor is a pure byte-zeroing operation. Route it to
    // cudaMemsetAsync, which runs on the copy/DMA engine and does not consume an SM kernel
    // launch slot. On llama3.2-1B this covers ~500 launches/step (ZeroGrad + scatter backward
    // zero-init) that were previously 3-4 us FillKernels each - see docs/kernel 优化.md §3.3.
    //
    // Preconditions:
    //   1. scalar is bit-exactly zero (see IsZeroBitsScalar; rejects -0.0 which has a nonzero
    //      sign bit and would produce a different bit pattern than memset).
    //   2. tensor is contiguous (IsContiguous() is currently unconditionally true in Tensor;
    //      the check is kept for when strided views land).
    //   3. dtype size is known (kDataTypeToSize covers every enum value).
    // Any of these failing -> fall through to FillKernel, which handles the general case.
    if (IsZeroBitsScalar(scalar) && tensor->IsContiguous()) {
        auto size_it = kDataTypeToSize.find(tensor->Dtype());
        if (size_it != kDataTypeToSize.end()) {
            const size_t bytes = num_elements * size_it->second;
            cudaMemsetAsync(tensor->DataPtr(), 0, bytes, cuda_stream);
            return;
        }
    }

    const int num_tokens = static_cast<int>(num_elements);
    const int threads_per_block = 256;
    const int num_blocks = (num_tokens + threads_per_block - 1) / threads_per_block;

    core::cuda::DispatchCudaFunc<INFINI_ALL_NUMERIC_TYPES>(
        tensor->Dtype(),
        [=]<typename T>() {
            const T casted_value = scalar.to<T>();
            FillKernel<T><<<num_blocks, threads_per_block, 0, cuda_stream>>>(static_cast<T *>(tensor->DataPtr()),
                                                                             casted_value, tensor->NumElements());
        },
        "CUDA Fill");
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_FILL_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_FILL_KERNEL(Fill)

#undef REGISTER_CUDA_FILL_KERNEL
