// Flash Attention CUDA kernel integration.
// Wraps the custom flash attention kernels from my-flash-attention/ into the
// InfiniTrain Dispatcher framework.
//
// Only nv_bfloat16 (DataType::kBFLOAT16) is supported.
// Input tensor layout:  Q/K/V: [bs, num_heads, seq_len, head_dim]
// head_dim must be 64.

#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

// Declarations of flash attention entry points.
// The implementations (attention_v6.cu / attention_v6_bp.cu) are compiled as
// separate translation units and linked in via CMake.
#include "my-flash-attention/flash_attention_interface.h"

namespace infini_train::kernels::cuda {

// FlashAttentionForward
//   Inputs:  Q [bs, q_head, q_len, head_dim]
//            K [bs, kv_head, kv_len, head_dim]
//            V [bs, kv_head, kv_len, head_dim]
//   Outputs: O [bs, q_head, q_len, head_dim]
//            L [bs * q_head * q_len]  (logsumexp, bf16, needed for backward)
std::vector<std::shared_ptr<Tensor>> FlashAttentionForward(const std::shared_ptr<Tensor> &Q,
                                                            const std::shared_ptr<Tensor> &K,
                                                            const std::shared_ptr<Tensor> &V,
                                                            bool is_causal) {
    CHECK(Q->Dtype() == DataType::kBFLOAT16)
        << "FlashAttentionForward only supports kBFLOAT16, got dtype=" << static_cast<int>(Q->Dtype());
    CHECK(K->Dtype() == DataType::kBFLOAT16)
        << "K dtype must be kBFLOAT16";
    CHECK(V->Dtype() == DataType::kBFLOAT16)
        << "V dtype must be kBFLOAT16";

    const auto &q_dims = Q->Dims();
    CHECK_EQ(q_dims.size(), 4) << "Q must be 4-D [bs, q_head, q_len, head_dim]";
    const int64_t bs = q_dims[0];
    const int64_t q_head = q_dims[1];
    const int64_t q_len = q_dims[2];
    const int64_t head_dim = q_dims[3];

    const auto &k_dims = K->Dims();
    CHECK_EQ(k_dims.size(), 4) << "K must be 4-D [bs, kv_head, kv_len, head_dim]";
    const int64_t kv_head = k_dims[1];
    const int64_t kv_len = k_dims[2];

    CHECK_EQ(head_dim, 64) << "FlashAttention currently only supports head_dim=64";
    CHECK_EQ(q_head % kv_head, 0) << "q_head must be divisible by kv_head";

    auto device = Q->GetDevice();
    auto cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                           infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                           ->cuda_stream();

    // Allocate output tensors
    auto O = std::make_shared<Tensor>(q_dims, DataType::kBFLOAT16, device);
    // L shape: [bs * q_head * q_len], stored as bf16
    auto L = std::make_shared<Tensor>(std::vector<int64_t>{bs * q_head * q_len}, DataType::kBFLOAT16, device);

    attention_v6(static_cast<const nv_bfloat16 *>(Q->DataPtr()),
                 static_cast<const nv_bfloat16 *>(K->DataPtr()),
                 static_cast<const nv_bfloat16 *>(V->DataPtr()),
                 static_cast<nv_bfloat16 *>(O->DataPtr()),
                 static_cast<nv_bfloat16 *>(L->DataPtr()),
                 static_cast<int>(bs),
                 static_cast<int>(q_head),
                 static_cast<int>(kv_head),
                 static_cast<int>(q_len),
                 static_cast<int>(kv_len),
                 static_cast<int>(head_dim),
                 nullptr,         // atten_mask: not supported
                 is_causal,
                 0.0f,            // dropout_p: not supported
                 kv_head != q_head, // is_gqa
                 cuda_stream
    );

    return {O, L};
}

// FlashAttentionBackward
//   Inputs:  Q, K, V, O (saved from forward), L (logsumexp, bf16), dO (grad of output)
//   Outputs: [dQ, dK, dV]
std::vector<std::shared_ptr<Tensor>> FlashAttentionBackward(const std::shared_ptr<Tensor> &Q,
                                                             const std::shared_ptr<Tensor> &K,
                                                             const std::shared_ptr<Tensor> &V,
                                                             const std::shared_ptr<Tensor> &O,
                                                             const std::shared_ptr<Tensor> &L,
                                                             const std::shared_ptr<Tensor> &dO,
                                                             bool is_causal) {
    const auto &q_dims = Q->Dims();
    const int64_t bs = q_dims[0];
    const int64_t q_head = q_dims[1];
    const int64_t q_len = q_dims[2];
    const int64_t head_dim = q_dims[3];

    const auto &k_dims = K->Dims();
    const int64_t kv_head = k_dims[1];
    const int64_t kv_len = k_dims[2];

    auto device = Q->GetDevice();
    auto cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                           infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                           ->cuda_stream();

    auto dQ = std::make_shared<Tensor>(q_dims, DataType::kBFLOAT16, device);
    auto dK = std::make_shared<Tensor>(k_dims, DataType::kBFLOAT16, device);
    auto dV = std::make_shared<Tensor>(k_dims, DataType::kBFLOAT16, device);

    attention_v6_backward(static_cast<const nv_bfloat16 *>(Q->DataPtr()),
                          static_cast<const nv_bfloat16 *>(K->DataPtr()),
                          static_cast<const nv_bfloat16 *>(V->DataPtr()),
                          static_cast<const nv_bfloat16 *>(O->DataPtr()),
                          static_cast<const nv_bfloat16 *>(L->DataPtr()),
                          static_cast<const nv_bfloat16 *>(dO->DataPtr()),
                          static_cast<nv_bfloat16 *>(dQ->DataPtr()),
                          static_cast<nv_bfloat16 *>(dK->DataPtr()),
                          static_cast<nv_bfloat16 *>(dV->DataPtr()),
                          static_cast<int>(bs),
                          static_cast<int>(q_head),
                          static_cast<int>(kv_head),
                          static_cast<int>(q_len),
                          static_cast<int>(kv_len),
                          static_cast<int>(head_dim),
                          is_causal,
                          cuda_stream);

    return {dQ, dK, dV};
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_FLASH_ATTENTION_KERNEL(kernel_name)                                          \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name,                         \
                    infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_FLASH_ATTENTION_KERNEL(FlashAttentionForward)
REGISTER_CUDA_FLASH_ATTENTION_KERNEL(FlashAttentionBackward)

#undef REGISTER_CUDA_FLASH_ATTENTION_KERNEL
