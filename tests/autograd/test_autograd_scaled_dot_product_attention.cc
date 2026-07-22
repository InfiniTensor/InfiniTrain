#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {

struct AttentionResult {
    std::vector<float> output;
    std::vector<float> dq;
    std::vector<float> dk;
    std::vector<float> dv;
    DataType dq_dtype;
    DataType dk_dtype;
    DataType dv_dtype;
};

struct PackedAttentionResult {
    std::vector<float> output;
    std::vector<float> dqkv;
};

std::vector<float> MakeValues(size_t count, float frequency, float scale) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) { values[i] = std::sin(static_cast<float>(i) * frequency) * scale; }
    return values;
}

std::vector<float> ToFloatVector(const std::shared_ptr<Tensor> &tensor) {
    auto float_tensor = tensor->Dtype() == DataType::kFLOAT32 ? *tensor : tensor->To(DataType::kFLOAT32);
    auto cpu_tensor = float_tensor.GetDevice().IsCPU() ? float_tensor : float_tensor.To(Device());
    const auto *data = static_cast<const float *>(cpu_tensor.DataPtr());
    return {data, data + cpu_tensor.NumElements()};
}

class CaptureGradHook : public autograd::PreAccumulateGradHook {
public:
    void operator()(const std::shared_ptr<Tensor> &) override {}

    bool TryBypassAccumulate(const std::shared_ptr<Tensor> &, const std::shared_ptr<Tensor> &grad_output, bool,
                             float) override {
        dtype_ = grad_output->Dtype();
        values_ = ToFloatVector(grad_output);
        return true;
    }

    const std::vector<float> &Values() const { return values_; }
    DataType Dtype() const { return dtype_; }

private:
    std::vector<float> values_;
    DataType dtype_ = DataType::kFLOAT32;
};

std::shared_ptr<Tensor> MakeTensor(const std::vector<float> &values, const std::vector<int64_t> &dims, Device device,
                                   DataType dtype = DataType::kBFLOAT16) {
    auto cpu_float = std::make_shared<Tensor>(values.data(), dims, DataType::kFLOAT32);
    auto cpu_tensor = std::make_shared<Tensor>(cpu_float->To(dtype));
    return std::make_shared<Tensor>(cpu_tensor->To(device));
}

AttentionResult RunFlashAttention(Device device, bool expand_kv, float input_scale = 0.2F, float grad_scale = 0.1F) {
    constexpr int64_t batch = 1;
    constexpr int64_t query_heads = 8;
    constexpr int64_t kv_heads = 2;
    constexpr int64_t seqlen = 64;
    constexpr int64_t head_dim = 64;
    constexpr int64_t groups = query_heads / kv_heads;

    const std::vector<int64_t> q_dims{batch, query_heads, seqlen, head_dim};
    const std::vector<int64_t> kv_dims{batch, kv_heads, seqlen, head_dim};
    const auto q_values = MakeValues(batch * query_heads * seqlen * head_dim, 0.013F, input_scale);
    const auto k_values = MakeValues(batch * kv_heads * seqlen * head_dim, 0.017F, input_scale);
    const auto v_values = MakeValues(batch * kv_heads * seqlen * head_dim, 0.019F, input_scale);
    const auto grad_values = MakeValues(batch * query_heads * seqlen * head_dim, 0.023F, grad_scale);

    auto q = MakeTensor(q_values, q_dims, device);
    auto k = MakeTensor(k_values, kv_dims, device);
    auto v = MakeTensor(v_values, kv_dims, device);
    q->RequiresGrad();
    k->RequiresGrad();
    v->RequiresGrad();
    auto q_grad = std::make_shared<CaptureGradHook>();
    auto k_grad = std::make_shared<CaptureGradHook>();
    auto v_grad = std::make_shared<CaptureGradHook>();
    q->RegisterPreAccumulateGradHook(q_grad);
    k->RegisterPreAccumulateGradHook(k_grad);
    v->RegisterPreAccumulateGradHook(v_grad);

    auto k_input = expand_kv ? k->RepeatInterleave(groups, 1) : k;
    auto v_input = expand_kv ? v->RepeatInterleave(groups, 1) : v;
    auto output = nn::function::ScaledDotProductAttention(q, k_input, v_input, 1.0F / std::sqrt(head_dim));
    auto grad = MakeTensor(grad_values, q_dims, device);
    output->Backward(grad);

    return {
        .output = ToFloatVector(output),
        .dq = q_grad->Values(),
        .dk = k_grad->Values(),
        .dv = v_grad->Values(),
        .dq_dtype = q_grad->Dtype(),
        .dk_dtype = k_grad->Dtype(),
        .dv_dtype = v_grad->Dtype(),
    };
}

AttentionResult RunUnfusedAttention(Device device, bool upcast = false, float input_scale = 0.2F,
                                    float grad_scale = 0.1F) {
    constexpr int64_t batch = 1;
    constexpr int64_t query_heads = 8;
    constexpr int64_t kv_heads = 2;
    constexpr int64_t seqlen = 64;
    constexpr int64_t head_dim = 64;
    constexpr int64_t groups = query_heads / kv_heads;

    const std::vector<int64_t> q_dims{batch, query_heads, seqlen, head_dim};
    const std::vector<int64_t> kv_dims{batch, kv_heads, seqlen, head_dim};
    const auto q_values = MakeValues(batch * query_heads * seqlen * head_dim, 0.013F, input_scale);
    const auto k_values = MakeValues(batch * kv_heads * seqlen * head_dim, 0.017F, input_scale);
    const auto v_values = MakeValues(batch * kv_heads * seqlen * head_dim, 0.019F, input_scale);
    const auto grad_values = MakeValues(batch * query_heads * seqlen * head_dim, 0.023F, grad_scale);

    auto q = MakeTensor(q_values, q_dims, device);
    auto k = MakeTensor(k_values, kv_dims, device);
    auto v = MakeTensor(v_values, kv_dims, device);
    if (upcast) {
        q = std::make_shared<Tensor>(q->To(DataType::kFLOAT32));
        k = std::make_shared<Tensor>(k->To(DataType::kFLOAT32));
        v = std::make_shared<Tensor>(v->To(DataType::kFLOAT32));
    }
    q->RequiresGrad();
    k->RequiresGrad();
    v->RequiresGrad();
    auto q_grad = std::make_shared<CaptureGradHook>();
    auto k_grad = std::make_shared<CaptureGradHook>();
    auto v_grad = std::make_shared<CaptureGradHook>();
    q->RegisterPreAccumulateGradHook(q_grad);
    k->RegisterPreAccumulateGradHook(k_grad);
    v->RegisterPreAccumulateGradHook(v_grad);

    auto k_expanded = k->RepeatInterleave(groups, 1);
    auto v_expanded = v->RepeatInterleave(groups, 1);
    auto scores = q->Matmul(k_expanded->Transpose(-2, -1)) * (1.0F / std::sqrt(head_dim));
    std::vector<float> mask_values(seqlen * seqlen, 0.0F);
    for (int64_t row = 0; row < seqlen; ++row) {
        for (int64_t column = row + 1; column < seqlen; ++column) { mask_values[row * seqlen + column] = 1.0F; }
    }
    auto mask = MakeTensor(mask_values, {1, 1, seqlen, seqlen}, device, DataType::kBOOL);
    auto probabilities = nn::function::Softmax(scores->MaskedFill(mask, std::numeric_limits<float>::lowest()), -1);
    auto output = probabilities->Matmul(v_expanded);
    auto grad = MakeTensor(grad_values, q_dims, device);
    if (upcast) {
        grad = std::make_shared<Tensor>(grad->To(DataType::kFLOAT32));
    }
    output->Backward(grad);

    return {
        .output = ToFloatVector(output),
        .dq = q_grad->Values(),
        .dk = k_grad->Values(),
        .dv = v_grad->Values(),
    };
}

PackedAttentionResult RunPackedFlashAttention(Device device, bool expand_kv) {
    constexpr int64_t batch = 1;
    constexpr int64_t query_heads = 8;
    constexpr int64_t kv_heads = 2;
    constexpr int64_t seqlen = 64;
    constexpr int64_t head_dim = 64;
    constexpr int64_t groups = query_heads / kv_heads;
    constexpr int64_t query_width = query_heads * head_dim;
    constexpr int64_t kv_width = kv_heads * head_dim;
    constexpr int64_t packed_width = query_width + 2 * kv_width;

    const std::vector<int64_t> packed_dims{batch, seqlen, packed_width};
    const std::vector<int64_t> output_dims{batch, seqlen, query_width};
    const auto packed_values = MakeValues(batch * seqlen * packed_width, 0.013F, 0.2F);
    const auto grad_values = MakeValues(batch * seqlen * query_width, 0.023F, 0.1F);

    auto packed = MakeTensor(packed_values, packed_dims, device);
    packed->RequiresGrad();
    auto packed_grad = std::make_shared<CaptureGradHook>();
    packed->RegisterPreAccumulateGradHook(packed_grad);
    auto q = packed->Slice(2, 0, query_width)->View({batch, seqlen, query_heads, head_dim});
    auto k = packed->Slice(2, query_width, query_width + kv_width)->View({batch, seqlen, kv_heads, head_dim});
    auto v = packed->Slice(2, query_width + kv_width, packed_width)->View({batch, seqlen, kv_heads, head_dim});
    if (expand_kv) {
        k = k->RepeatInterleave(groups, 2);
        v = v->RepeatInterleave(groups, 2);
    }
    q = q->Transpose(1, 2);
    k = k->Transpose(1, 2);
    v = v->Transpose(1, 2);

    auto output = nn::function::ScaledDotProductAttention(q, k, v, 1.0F / std::sqrt(head_dim));
    output = output->Transpose(1, 2)->Contiguous()->View(output_dims);
    auto grad = MakeTensor(grad_values, output_dims, device);
    output->Backward(grad);

    return {
        .output = ToFloatVector(output),
        .dqkv = packed_grad->Values(),
    };
}

PackedAttentionResult RunPackedUnfusedAttention(Device device) {
    constexpr int64_t batch = 1;
    constexpr int64_t query_heads = 8;
    constexpr int64_t kv_heads = 2;
    constexpr int64_t seqlen = 64;
    constexpr int64_t head_dim = 64;
    constexpr int64_t groups = query_heads / kv_heads;
    constexpr int64_t query_width = query_heads * head_dim;
    constexpr int64_t kv_width = kv_heads * head_dim;
    constexpr int64_t packed_width = query_width + 2 * kv_width;

    const std::vector<int64_t> packed_dims{batch, seqlen, packed_width};
    const std::vector<int64_t> output_dims{batch, seqlen, query_width};
    const auto packed_values = MakeValues(batch * seqlen * packed_width, 0.013F, 0.2F);
    const auto grad_values = MakeValues(batch * seqlen * query_width, 0.023F, 0.1F);

    auto packed = MakeTensor(packed_values, packed_dims, device);
    packed->RequiresGrad();
    auto packed_grad = std::make_shared<CaptureGradHook>();
    packed->RegisterPreAccumulateGradHook(packed_grad);
    auto q = packed->Slice(2, 0, query_width)->View({batch, seqlen, query_heads, head_dim});
    auto k = packed->Slice(2, query_width, query_width + kv_width)->View({batch, seqlen, kv_heads, head_dim});
    auto v = packed->Slice(2, query_width + kv_width, packed_width)->View({batch, seqlen, kv_heads, head_dim});
    k = k->RepeatInterleave(groups, 2);
    v = v->RepeatInterleave(groups, 2);
    q = q->Transpose(1, 2);
    k = k->Transpose(1, 2);
    v = v->Transpose(1, 2);

    auto scores = q->Matmul(k->Transpose(-2, -1)) * (1.0F / std::sqrt(head_dim));
    std::vector<float> mask_values(seqlen * seqlen, 0.0F);
    for (int64_t row = 0; row < seqlen; ++row) {
        for (int64_t column = row + 1; column < seqlen; ++column) { mask_values[row * seqlen + column] = 1.0F; }
    }
    auto mask = MakeTensor(mask_values, {1, 1, seqlen, seqlen}, device, DataType::kBOOL);
    auto probabilities = nn::function::Softmax(scores->MaskedFill(mask, std::numeric_limits<float>::lowest()), -1);
    auto output = probabilities->Matmul(v);
    output = output->Transpose(1, 2)->Contiguous()->View(output_dims);
    auto grad = MakeTensor(grad_values, output_dims, device);
    output->Backward(grad);

    return {
        .output = ToFloatVector(output),
        .dqkv = packed_grad->Values(),
    };
}

void ExpectClose(const std::vector<float> &actual, const std::vector<float> &expected, float max_relative_l2,
                 float min_cosine, const std::string &name) {
    ASSERT_EQ(actual.size(), expected.size()) << name;
    double diff_squared = 0.0;
    double actual_squared = 0.0;
    double expected_squared = 0.0;
    double dot = 0.0;
    float max_abs = 0.0F;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double diff = static_cast<double>(actual[i]) - expected[i];
        diff_squared += diff * diff;
        actual_squared += static_cast<double>(actual[i]) * actual[i];
        expected_squared += static_cast<double>(expected[i]) * expected[i];
        dot += static_cast<double>(actual[i]) * expected[i];
        max_abs = std::max(max_abs, std::abs(actual[i] - expected[i]));
    }
    const double relative_l2 = std::sqrt(diff_squared) / std::max(std::sqrt(expected_squared), 1e-30);
    const double cosine = dot / std::max(std::sqrt(actual_squared * expected_squared), 1e-30);
    EXPECT_LE(relative_l2, max_relative_l2) << name << " max_abs=" << max_abs << " cosine=" << cosine;
    EXPECT_GE(cosine, min_cosine) << name << " max_abs=" << max_abs << " relative_l2=" << relative_l2;
}

float MaxAbsDiff(const std::vector<float> &actual, const std::vector<float> &expected) {
    EXPECT_EQ(actual.size(), expected.size());
    float max_abs = 0.0F;
    for (size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) {
        max_abs = std::max(max_abs, std::abs(actual[i] - expected[i]));
    }
    return max_abs;
}

void ExpectErrorBoundedByBfloat16Reference(const std::vector<float> &flash, const std::vector<float> &bfloat16,
                                           const std::vector<float> &float32, const std::string &name) {
    const float flash_error = MaxAbsDiff(flash, float32);
    const float bfloat16_error = MaxAbsDiff(bfloat16, float32);
    EXPECT_LE(flash_error, 3.0F * bfloat16_error + 1e-5F)
        << name << " flash_max_error=" << flash_error << " bfloat16_max_error=" << bfloat16_error;
}

} // namespace

class AutogradScaledDotProductAttentionTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradScaledDotProductAttentionTest, NativeGqaMatchesExpandedKv) {
    ONLY_CUDA();

    const auto native_gqa = RunFlashAttention(GetDevice(), false);
    const auto expanded_kv = RunFlashAttention(GetDevice(), true);

    ExpectClose(native_gqa.output, expanded_kv.output, 0.001F, 0.9999F, "output");
    ExpectClose(native_gqa.dq, expanded_kv.dq, 0.001F, 0.9999F, "dQ");
    ExpectClose(native_gqa.dk, expanded_kv.dk, 0.01F, 0.9999F, "dK");
    ExpectClose(native_gqa.dv, expanded_kv.dv, 0.01F, 0.9999F, "dV");
}

TEST_P(AutogradScaledDotProductAttentionTest, Bfloat16BackwardPropagatesFloat32Gradients) {
    ONLY_CUDA();

    const auto result = RunFlashAttention(GetDevice(), false);

    EXPECT_EQ(result.dq_dtype, DataType::kFLOAT32);
    EXPECT_EQ(result.dk_dtype, DataType::kFLOAT32);
    EXPECT_EQ(result.dv_dtype, DataType::kFLOAT32);
}

TEST_P(AutogradScaledDotProductAttentionTest, NativeGqaMatchesExpandedKvWithPackedInput) {
    ONLY_CUDA();

    const auto native_gqa = RunPackedFlashAttention(GetDevice(), false);
    const auto expanded_kv = RunPackedFlashAttention(GetDevice(), true);

    ExpectClose(native_gqa.output, expanded_kv.output, 0.001F, 0.9999F, "output");
    ExpectClose(native_gqa.dqkv, expanded_kv.dqkv, 0.02F, 0.9999F, "packed dQKV");
}

TEST_P(AutogradScaledDotProductAttentionTest, NativeGqaMatchesUnfusedReferenceWithPackedInput) {
    ONLY_CUDA();

    const auto native_gqa = RunPackedFlashAttention(GetDevice(), false);
    const auto unfused = RunPackedUnfusedAttention(GetDevice());

    ExpectClose(native_gqa.output, unfused.output, 0.01F, 0.9995F, "output");
    ExpectClose(native_gqa.dqkv, unfused.dqkv, 0.03F, 0.9995F, "packed dQKV");
}

TEST_P(AutogradScaledDotProductAttentionTest, NativeGqaMatchesUnfusedReference) {
    ONLY_CUDA();

    const auto native_gqa = RunFlashAttention(GetDevice(), false);
    const auto unfused = RunUnfusedAttention(GetDevice());

    ExpectClose(native_gqa.output, unfused.output, 0.01F, 0.9995F, "output");
    ExpectClose(native_gqa.dq, unfused.dq, 0.01F, 0.9995F, "dQ");
    ExpectClose(native_gqa.dk, unfused.dk, 0.03F, 0.9995F, "dK");
    ExpectClose(native_gqa.dv, unfused.dv, 0.01F, 0.9995F, "dV");
}

TEST_P(AutogradScaledDotProductAttentionTest, NativeGqaErrorIsBoundedByUnfusedBfloat16Error) {
    ONLY_CUDA();

    const auto native_gqa = RunFlashAttention(GetDevice(), false, 1.0F, 1.0F);
    const auto bfloat16 = RunUnfusedAttention(GetDevice(), false, 1.0F, 1.0F);
    const auto float32 = RunUnfusedAttention(GetDevice(), true, 1.0F, 1.0F);

    ExpectErrorBoundedByBfloat16Reference(native_gqa.output, bfloat16.output, float32.output, "output");
    ExpectErrorBoundedByBfloat16Reference(native_gqa.dq, bfloat16.dq, float32.dq, "dQ");
    ExpectErrorBoundedByBfloat16Reference(native_gqa.dk, bfloat16.dk, float32.dk, "dK");
    ExpectErrorBoundedByBfloat16Reference(native_gqa.dv, bfloat16.dv, float32.dv, "dV");
}

INFINI_TRAIN_REGISTER_TEST(AutogradScaledDotProductAttentionTest);
