#include "infini_train/include/autograd/scaled_dot_product_attention.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
namespace {

constexpr char kForwardKernel[] = "ScaledDotProductAttentionForward";
constexpr char kBackwardKernel[] = "ScaledDotProductAttentionBackward";

DataType SelectFlashAttentionDtype(const std::shared_ptr<Tensor> &q) {
    if (q->Dtype() == DataType::kBFLOAT16 || q->Dtype() == DataType::kFLOAT16) {
        return q->Dtype();
    }
    if (tls_autocast_context.enabled
        && (tls_autocast_context.autocast_dtype == DataType::kBFLOAT16
            || tls_autocast_context.autocast_dtype == DataType::kFLOAT16)) {
        return tls_autocast_context.autocast_dtype;
    }
    LOG(FATAL) << "FlashAttention 2 supports fp16/bf16 only. Use --dtype=bfloat16 for --attention_backend=flash.";
    return DataType::kBFLOAT16;
}

void CheckQKVHeads(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                   const std::shared_ptr<Tensor> &v) {
    CHECK_EQ(q->Dims().size(), 4) << "Q must use (B, H, T, D) layout";
    CHECK_EQ(k->Dims().size(), 4) << "K must use (B, H, T, D) layout";
    CHECK_EQ(v->Dims().size(), 4) << "V must use (B, H, T, D) layout";
    CHECK(k->Dims() == v->Dims()) << "K and V must have the same shape";

    const auto query_heads = q->Dims()[1];
    const auto kv_heads = k->Dims()[1];
    CHECK_GT(query_heads, 0) << "Q must have at least one head";
    CHECK_GT(kv_heads, 0) << "K/V must have at least one head";
    CHECK_EQ(query_heads % kv_heads, 0) << "Q heads must be divisible by KV heads for GQA/MQA";
}

} // namespace

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3);
    const auto &q = input_tensors[0];
    const auto &k = input_tensors[1];
    const auto &v = input_tensors[2];

    CheckQKVHeads(q, k, v);
    const auto device = q->GetDevice();
    CHECK(device.IsCUDA()) << "FlashAttention backend requires CUDA tensors";

    const Dispatcher::KeyT forward_key{device.type(), kForwardKernel};
    CHECK(Dispatcher::Instance().HasKernel(forward_key))
        << "FlashAttention backend is not available in this build; configure with -DUSE_FLASH_ATTENTION=ON";

    const auto flash_dtype = SelectFlashAttentionDtype(q);
    flash_ctx_.reset();
    return {
        Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(forward_key, q, k, v, scale_, flash_dtype, &flash_ctx_)};
}

void ScaledDotProductAttention::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                             const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    CHECK_EQ(input_tensors.size(), 3);
    CHECK_EQ(output_tensors.size(), 1);
    ctx_.SaveForBackward({input_tensors[0], input_tensors[1], input_tensors[2], output_tensors[0]});
}

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttention::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto saved_tensors = ctx_.GetSavedTensors();
    CHECK_EQ(saved_tensors.size(), 4);
    const auto &q = saved_tensors[0];
    const auto &k = saved_tensors[1];
    const auto &v = saved_tensors[2];
    const auto &out = saved_tensors[3];

    const Dispatcher::KeyT backward_key{q->GetDevice().type(), kBackwardKernel};
    CHECK(Dispatcher::Instance().HasKernel(backward_key))
        << "FlashAttention backward kernel is not available in this build; configure with "
           "-DUSE_FLASH_ATTENTION=ON";

    return Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(backward_key, grad_output, q, k, v, out,
                                                                             scale_, flash_ctx_);
}

} // namespace infini_train::autograd
