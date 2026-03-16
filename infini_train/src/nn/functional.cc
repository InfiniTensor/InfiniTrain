#include "infini_train/include/nn/functional.h"

#include <cstdint>
#include <optional>
#include <limits>
#include <cmath>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/autograd/misc.h"
#include "infini_train/include/autograd/reduction.h"
#include "infini_train/include/autograd/softmax.h"
#include "infini_train/include/autograd/transform.h"
#include "infini_train/include/autograd/sdpa.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::function {
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


namespace {
std::shared_ptr<Tensor> RepeatHeads(const std::shared_ptr<Tensor> &x, int64_t n_rep) {
    // x: (B, H, T, D)
    if (n_rep == 1) {
        return x;
    }
    CHECK_EQ(x->Dims().size(), 4);
    return x->RepeatInterleave(n_rep, 1)->Contiguous();
}

std::shared_ptr<Tensor> MakeCausalMask(int64_t T, Device device) {
    // Returns mask of shape (1, 1, T, T) with 1s in upper triangle (excluding diagonal).
    // NOTE: Ones() currently creates a CPU tensor; move it to target device.
    auto ones_cpu = nn::function::Ones({T, T});
    auto ones = std::make_shared<Tensor>(ones_cpu->To(device));
    return nn::function::Triu(ones, 1)->View({1, 1, T, T});
}
} // namespace

std::shared_ptr<Tensor> ScaledDotProductAttention(
    const std::shared_ptr<Tensor> &query, const std::shared_ptr<Tensor> &key,
    const std::shared_ptr<Tensor> &value, const std::shared_ptr<Tensor> &attn_mask,
    double dropout_p, bool is_causal, std::optional<double> scale, bool enable_gqa) {
    CHECK(query != nullptr);
    CHECK(key != nullptr);
    CHECK(value != nullptr);

    if (dropout_p == 0.0) {
        return std::make_shared<autograd::ScaledDotProductAttention>(attn_mask, dropout_p, is_causal, scale, enable_gqa)
            ->Apply({query, key, value})[0];
    }

    const auto &q_shape = query->Dims();
    const auto &k_shape = key->Dims();
    const auto &v_shape = value->Dims();
    CHECK_EQ(q_shape.size(), 4);
    CHECK_EQ(k_shape.size(), 4);
    CHECK_EQ(v_shape.size(), 4);

    const int64_t B = q_shape[0];
    const int64_t Hq = q_shape[1];
    const int64_t Tq = q_shape[2];
    const int64_t D = q_shape[3];

    CHECK_EQ(k_shape[0], B);
    CHECK_EQ(v_shape[0], B);
    CHECK_EQ(k_shape[2], Tq);
    CHECK_EQ(v_shape[2], Tq);
    CHECK_EQ(k_shape[3], D);
    CHECK_EQ(v_shape[3], D);

    auto k = key;
    auto v = value;

    if (enable_gqa) {
        const int64_t Hk = k_shape[1];
        const int64_t Hv = v_shape[1];
        CHECK_EQ(Hk, Hv) << "GQA expects key/value to have the same #heads";
        CHECK_EQ(Hq % Hk, 0) << "query heads must be divisible by kv heads when enable_gqa";
        const int64_t n_rep = Hq / Hk;
        if (n_rep != 1) {
            k = RepeatHeads(k, n_rep);
            v = RepeatHeads(v, n_rep);
        }
    }

    const double scale_value = scale.has_value() ? *scale : (1.0 / std::sqrt(static_cast<double>(D)));

    // att: (B, H, T, T)
    auto att = query->Matmul(k->Transpose(-2, -1)) * static_cast<float>(scale_value);

    std::shared_ptr<Tensor> mask = attn_mask;
    if (is_causal) {
        auto causal = MakeCausalMask(Tq, query->GetDevice());
        if (mask != nullptr) {
            mask = (mask > 0) | (causal > 0);
        } else {
            mask = causal;
        }
    }

    if (mask != nullptr) {
        att = att->MaskedFill(mask, std::numeric_limits<float>::lowest());
    }

    att = nn::function::Softmax(att, -1);

    if (dropout_p > 0.0) {
        CHECK_GE(dropout_p, 0.0);
        CHECK_LT(dropout_p, 1.0);
        const double keep_prob = 1.0 - dropout_p;

        // Random uniform in [0, 1)
        auto r = std::make_shared<Tensor>(att->Dims(), DataType::kFLOAT32, att->GetDevice());
        nn::init::Uniform(r, 0.0f, 1.0f);

        auto keep = r < static_cast<float>(keep_prob);
        att = att * keep * static_cast<float>(1.0 / keep_prob);
    }

    return att->Matmul(v);
}

} // namespace infini_train::nn::function
