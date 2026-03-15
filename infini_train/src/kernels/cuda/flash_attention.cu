#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"
#include "my-flash-attention/flash_attention_interface.h"

namespace infini_train::kernels::cuda {

std::vector<std::shared_ptr<Tensor>> FlashAttentionForward(const std::shared_ptr<Tensor> &q,
                                                           const std::shared_ptr<Tensor> &k,
                                                           const std::shared_ptr<Tensor> &v, bool is_causal,
                                                           float scale = -1.0f) {
    CHECK(q->Dtype() == DataType::kBFLOAT16);
    const auto &q_dims = q->Dims();
    CHECK_EQ(static_cast<int>(q_dims.size()), 4);
    const int64_t head_dim = q_dims[3];
    CHECK_EQ(head_dim, static_cast<int64_t>(64));

    const int64_t B = q_dims[0];
    const int64_t q_head = q_dims[1];
    const int64_t q_len = q_dims[2];
    const int64_t kv_head = k->Dims()[1];
    const int64_t kv_len = k->Dims()[2];
    const bool is_gqa = (q_head != kv_head);

    auto device = q->GetDevice();
    // O is bf16; L is float32 (the kernel outputs float32 logsumexp)
    auto o = std::make_shared<Tensor>(std::vector<int64_t>{B, q_head, q_len, head_dim}, DataType::kBFLOAT16, device);
    auto l = std::make_shared<Tensor>(std::vector<int64_t>{B, q_head, q_len}, DataType::kFLOAT32, device);

    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    attention_v6(static_cast<const nv_bfloat16 *>(q->DataPtr()), static_cast<const nv_bfloat16 *>(k->DataPtr()),
                 static_cast<const nv_bfloat16 *>(v->DataPtr()), static_cast<nv_bfloat16 *>(o->DataPtr()),
                 static_cast<float *>(l->DataPtr()), static_cast<int>(B), static_cast<int>(q_head),
                 static_cast<int>(kv_head), static_cast<int>(q_len), static_cast<int>(kv_len),
                 static_cast<int>(head_dim),
                 /*atten_mask=*/nullptr, is_causal,
                 /*dropout_p=*/0.0f, is_gqa, scale, cuda_stream);

    return {o, l};
}

std::vector<std::shared_ptr<Tensor>>
FlashAttentionBackward(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                       const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &o,
                       const std::shared_ptr<Tensor> &l, const std::shared_ptr<Tensor> &do_, bool is_causal,
                       float scale = -1.0f) {
    CHECK(q->Dtype() == DataType::kBFLOAT16);
    CHECK(l->Dtype() == DataType::kFLOAT32);
    const auto &q_dims = q->Dims();
    CHECK_EQ(static_cast<int>(q_dims.size()), 4);
    const int64_t head_dim = q_dims[3];
    CHECK_EQ(head_dim, static_cast<int64_t>(64));

    const int64_t B = q_dims[0];
    const int64_t q_head = q_dims[1];
    const int64_t q_len = q_dims[2];
    const int64_t kv_head = k->Dims()[1];
    const int64_t kv_len = k->Dims()[2];

    auto device = q->GetDevice();

    // The backward kernel needs float32 dO. Convert if incoming grad is bf16.
    auto do_f32 = do_->Dtype() == DataType::kFLOAT32 ? do_ : std::make_shared<Tensor>(do_->To(DataType::kFLOAT32));

    // Allocate float32 gradient tensors
    auto dq_f32 = std::make_shared<Tensor>(q->Dims(), DataType::kFLOAT32, device);
    auto dk_f32 = std::make_shared<Tensor>(k->Dims(), DataType::kFLOAT32, device);
    auto dv_f32 = std::make_shared<Tensor>(v->Dims(), DataType::kFLOAT32, device);

    const auto &cuda_stream = dynamic_cast<infini_train::core::cuda::CudaStream *>(
                                  infini_train::core::GetDeviceGuardImpl(device.type())->GetStream(device))
                                  ->cuda_stream();

    attention_v6_backward(
        static_cast<const nv_bfloat16 *>(q->DataPtr()), static_cast<const nv_bfloat16 *>(k->DataPtr()),
        static_cast<const nv_bfloat16 *>(v->DataPtr()), static_cast<const nv_bfloat16 *>(o->DataPtr()),
        static_cast<const float *>(l->DataPtr()), static_cast<const float *>(do_f32->DataPtr()),
        static_cast<float *>(dq_f32->DataPtr()), static_cast<float *>(dk_f32->DataPtr()),
        static_cast<float *>(dv_f32->DataPtr()), static_cast<int>(B), static_cast<int>(q_head),
        static_cast<int>(kv_head), static_cast<int>(q_len), static_cast<int>(kv_len), static_cast<int>(head_dim),
        is_causal, scale, cuda_stream);

    // Convert float32 gradients back to bf16 to match the input parameter dtype
    auto dq = std::make_shared<Tensor>(dq_f32->To(DataType::kBFLOAT16));
    auto dk = std::make_shared<Tensor>(dk_f32->To(DataType::kBFLOAT16));
    auto dv = std::make_shared<Tensor>(dv_f32->To(DataType::kBFLOAT16));

    return {dq, dk, dv};
}

} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_FLASH_ATTENTION_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_FLASH_ATTENTION_KERNEL(FlashAttentionForward)
REGISTER_CUDA_FLASH_ATTENTION_KERNEL(FlashAttentionBackward)

#undef REGISTER_CUDA_FLASH_ATTENTION_KERNEL
