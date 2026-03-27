#include "infini_train/include/nn/functional.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/autograd/attention.h"
#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/autograd/misc.h"
#include "infini_train/include/autograd/reduction.h"
#include "infini_train/include/autograd/softmax.h"
#include "infini_train/include/autograd/transform.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::function {
std::shared_ptr<Tensor> ScaledDotProductAttention(const std::shared_ptr<Tensor> &query,
                                                  const std::shared_ptr<Tensor> &key,
                                                  const std::shared_ptr<Tensor> &value,
                                                  const std::shared_ptr<Tensor> &attn_mask, double dropout_p,
                                                  bool is_causal, std::optional<double> scale, bool enable_gqa) {
    CHECK(query);
    CHECK(key);
    CHECK(value);
    CHECK_GE(dropout_p, 0.0);
    CHECK_LT(dropout_p, 1.0);

    CHECK_EQ(query->Dims().size(), 4) << "ScaledDotProductAttention expects query shape (B, Hq, Tq, D).";
    CHECK_EQ(key->Dims().size(), 4) << "ScaledDotProductAttention expects key shape (B, Hk, Tk, D).";
    CHECK_EQ(value->Dims().size(), 4) << "ScaledDotProductAttention expects value shape (B, Hk, Tk, Dv).";

    const auto &q_dims = query->Dims();
    const auto &k_dims = key->Dims();
    const auto &v_dims = value->Dims();

    CHECK_EQ(q_dims[0], k_dims[0]);
    CHECK_EQ(q_dims[0], v_dims[0]);
    CHECK_EQ(k_dims[1], v_dims[1]);
    CHECK_EQ(k_dims[2], v_dims[2]);
    CHECK_EQ(q_dims[3], k_dims[3]);

    CHECK(query->Dtype() == key->Dtype());
    CHECK(query->Dtype() == value->Dtype());
    CHECK(query->GetDevice() == key->GetDevice());
    CHECK(query->GetDevice() == value->GetDevice());

    if (attn_mask) {
        CHECK(attn_mask->GetDevice() == query->GetDevice());
        CHECK(attn_mask->Dtype() == query->Dtype()) << "attn_mask dtype must match query/key/value dtype.";
    }

    if (enable_gqa) {
        CHECK_GE(q_dims[1], k_dims[1]);
        CHECK_EQ(q_dims[1] % k_dims[1], 0) << "For GQA, q_heads must be divisible by kv_heads.";
    } else {
        CHECK_EQ(q_dims[1], k_dims[1]);
    }

    CHECK_GE(query->Dims().size(), 1);
    const auto head_dim = static_cast<double>(query->Dims().back());
    CHECK_GT(head_dim, 0.0);
    const double scale_value = scale.value_or(1.0 / std::sqrt(head_dim));
    auto function
        = std::make_shared<autograd::ScaledDotProductAttention>(dropout_p, is_causal, scale_value, enable_gqa);

    if (attn_mask) {
        return function->Apply({query, key, value, attn_mask})[0];
    }
    return function->Apply({query, key, value})[0];
}

std::shared_ptr<Tensor> Tril(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    return std::make_shared<autograd::Tril>(diagonal)->Apply({input})[0];
}

std::shared_ptr<Tensor> Triu(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    return std::make_shared<autograd::Triu>(diagonal)->Apply({input})[0];
}

std::shared_ptr<Tensor> Ones(const std::vector<int64_t> size) {
    auto ones = std::make_shared<Tensor>(size, DataType::kFLOAT32);
    return init::Ones(ones);
}

std::shared_ptr<Tensor> Reciprocal(const std::shared_ptr<Tensor> &input) { return input->Reciprocal(); }

std::shared_ptr<Tensor> Sin(const std::shared_ptr<Tensor> &input) { return input->Sin(); }

std::shared_ptr<Tensor> Cos(const std::shared_ptr<Tensor> &input) { return input->Cos(); }

std::shared_ptr<Tensor> Tanh(const std::shared_ptr<Tensor> &input) { return input->Tanh(); }

std::shared_ptr<Tensor> Pow(const std::shared_ptr<Tensor> &input, float exponent) { return input->Pow(exponent); }

std::shared_ptr<Tensor> Pow(float base, const std::shared_ptr<Tensor> &input) {
    return std::make_shared<autograd::Pow>(base, true)->Apply({input})[0];
}

std::shared_ptr<Tensor> Rsqrt(const std::shared_ptr<Tensor> &input) { return input->Rsqrt(); }

std::shared_ptr<Tensor> Mean(const std::shared_ptr<Tensor> &input, int64_t dim, bool keep_dim) {
    return std::make_shared<autograd::Mean>(dim, keep_dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Sum(const std::shared_ptr<Tensor> &input, int64_t dim, bool keep_dim) {
    return std::make_shared<autograd::Sum>(dim, keep_dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Min(const std::shared_ptr<Tensor> &input, int64_t dim, bool keep_dim) {
    return std::make_shared<autograd::Min>(dim, keep_dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Max(const std::shared_ptr<Tensor> &input, int64_t dim, bool keep_dim) {
    return std::make_shared<autograd::Max>(dim, keep_dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Slice(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &starts,
                              const std::vector<int64_t> &ends, const std::vector<int64_t> &steps) {
    return input->Slice(starts, ends, steps);
}

std::shared_ptr<Tensor> Stack(const std::vector<std::shared_ptr<Tensor>> &inputs, int64_t dim) {
    return std::make_shared<autograd::Stack>(dim)->Apply(inputs)[0];
}

std::shared_ptr<Tensor> Concat(const std::vector<std::shared_ptr<Tensor>> &inputs, int64_t dim) {
    return std::make_shared<autograd::Concat>(dim)->Apply(inputs)[0];
}

std::shared_ptr<Tensor> Softmax(const std::shared_ptr<Tensor> &input, int64_t dim) {
    return std::make_shared<autograd::Softmax>(dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Sigmoid(const std::shared_ptr<Tensor> &input) {
    return std::make_shared<autograd::Sigmoid>()->Apply({input})[0];
}
} // namespace infini_train::nn::function
