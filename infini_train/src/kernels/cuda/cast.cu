#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>

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

constexpr int kCastThreadsPerBlock = 256;
// Cap the grid so the vectorized kernel uses a grid-stride loop on large tensors
// instead of launching an unbounded number of blocks.
constexpr size_t kCastMaxBlocks = 4096;

// Unsigned chunk types for wide (up to 128-bit) vectorized loads/stores.
template <int Bytes> struct CastChunk;
template <> struct CastChunk<16> {
    using type = uint4;
};
template <> struct CastChunk<8> {
    using type = uint2;
};
template <> struct CastChunk<4> {
    using type = uint32_t;
};
template <> struct CastChunk<2> {
    using type = uint16_t;
};
template <> struct CastChunk<1> {
    using type = uint8_t;
};

// Scalar grid-stride fallback for misaligned buffers.
template <typename Tdst, typename Tsrc> __global__ void CastKernel(Tdst *dst, const Tsrc *src, size_t num_elements) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements; idx += stride) {
        dst[idx] = common::cuda::Cast<Tdst>(src[idx]);
    }
}

// Vectorized cast: each thread converts kElems elements through one wide load and
// one wide store (each up to 128-bit). The trailing (< kElems) elements are
// handled by the first few threads. Elementwise conversion goes through
// common::cuda::Cast, identical to the scalar kernel.
template <typename Tdst, typename Tsrc, int kElems>
__global__ void CastKernelVec(Tdst *__restrict__ dst, const Tsrc *__restrict__ src, size_t num_elements) {
    using LoadChunk = typename CastChunk<kElems * sizeof(Tsrc)>::type;
    using StoreChunk = typename CastChunk<kElems * sizeof(Tdst)>::type;

    const size_t num_vecs = num_elements / kElems;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vecs; v += stride) {
        const LoadChunk load_chunk = reinterpret_cast<const LoadChunk *>(src)[v];
        Tsrc in[kElems];
        Tdst out[kElems];
        memcpy(in, &load_chunk, sizeof(in));
#pragma unroll
        for (int i = 0; i < kElems; ++i) { out[i] = common::cuda::Cast<Tdst>(in[i]); }
        StoreChunk store_chunk;
        memcpy(&store_chunk, out, sizeof(store_chunk));
        reinterpret_cast<StoreChunk *>(dst)[v] = store_chunk;
    }

    const size_t tail_idx = num_vecs * kElems + blockIdx.x * blockDim.x + threadIdx.x;
    if (tail_idx < num_elements) {
        dst[tail_idx] = common::cuda::Cast<Tdst>(src[tail_idx]);
    }
}

} // namespace

std::shared_ptr<Tensor> Cast(std::shared_ptr<Tensor> input, DataType dtype) {
    auto dst_tensor = std::make_shared<Tensor>(input->Dims(), dtype, input->GetDevice());
    auto device = input->GetDevice();
    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    const size_t num_elements = input->NumElements();

    core::cuda::DispatchCudaFunc<DataTypeList<INFINI_ALL_NUMERIC_TYPES, INFINI_LOGICAL_TYPES>,
                                 DataTypeList<INFINI_ALL_NUMERIC_TYPES, INFINI_LOGICAL_TYPES>>(
        {dtype, input->Dtype()},
        [=]<typename Tdst, typename Tsrc>() {
            auto dst = static_cast<Tdst *>(dst_tensor->DataPtr());
            auto src = static_cast<const Tsrc *>(input->DataPtr());

            // Wide enough that both the load and the store chunk are at most 128-bit.
            constexpr int kElems = static_cast<int>(16 / std::max(sizeof(Tdst), sizeof(Tsrc)));
            const bool aligned
                = (reinterpret_cast<uintptr_t>(dst) % 16 == 0) && (reinterpret_cast<uintptr_t>(src) % 16 == 0);
            if (aligned && num_elements >= static_cast<size_t>(kElems)) {
                const size_t num_vecs = num_elements / kElems;
                const size_t blocks = std::max<size_t>(
                    std::min((num_vecs + kCastThreadsPerBlock - 1) / kCastThreadsPerBlock, kCastMaxBlocks), 1);
                CastKernelVec<Tdst, Tsrc, kElems>
                    <<<static_cast<unsigned int>(blocks), kCastThreadsPerBlock, 0, cuda_stream>>>(dst, src,
                                                                                                  num_elements);
            } else {
                const size_t blocks = std::max<size_t>(
                    std::min((num_elements + kCastThreadsPerBlock - 1) / kCastThreadsPerBlock, kCastMaxBlocks), 1);
                CastKernel<<<static_cast<unsigned int>(blocks), kCastThreadsPerBlock, 0, cuda_stream>>>(dst, src,
                                                                                                        num_elements);
            }
        },
        "CUDA Cast");

    return {dst_tensor};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_CAST_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_CAST_KERNEL(Cast)

#undef REGISTER_CUDA_CAST_KERNEL
