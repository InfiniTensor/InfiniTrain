#include "infini_train/include/nn/modules/transformer/mlp.h"

#include <cmath>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

MLP::MLP(const TransformerConfig &config) : CloneableModule(kType) {
    // Compute hidden dimension
    // Base dimension: n_embd * ffn_expansion_ratio
    int64_t ffn_hidden = static_cast<int64_t>(config.n_embd * config.ffn_expansion_ratio);

    // Apply SwiGLU adjustment
    if (config.activation_type == MLPType::kSwiGLU) {
        ffn_hidden = int(2 * ffn_hidden) / 3; // SwiGLU intermediate
    }

    // Apply multiplier
    if (config.ffn_dim_multiplier.has_value()) {
        ffn_hidden
            = static_cast<int64_t>(std::llround(static_cast<double>(ffn_hidden) * config.ffn_dim_multiplier.value()));
    }

    // Round up to multiple_of
    int64_t before_round = ffn_hidden;
    ffn_hidden = (ffn_hidden + config.multiple_of - 1) / config.multiple_of * config.multiple_of;

    // c_fc: ColumnParallel (input full, output parallel)
    modules_[kCFcLayerName] = std::make_shared<parallel::ColumnParallelLinear>(
        /*in_features=*/config.n_embd, /*out_features=*/ffn_hidden,
        /*bias=*/config.add_bias_linear,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/parallel::global::GetSequenceParallelEnabled());

    // For SwiGLU, add second projection
    if (config.activation_type == MLPType::kSwiGLU) {
        modules_[kCFc2LayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
            /*in_features=*/config.n_embd, /*out_features=*/ffn_hidden,
            /*bias=*/config.add_bias_linear,
            /*gather_output=*/false,
            /*input_is_parallel=*/false,
            /*skip_bias_add=*/false,
            /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());
    }

    // Activation: check for GELU or SwiGLU
    if (config.activation_type == MLPType::kGELU) {
        modules_[kGeluLayerName] = std::make_shared<NewGELU>();
    } else if (config.activation_type == MLPType::kSwiGLU) {
        modules_[kSiluLayerName] = std::make_shared<SwiGLU>();
    }

    // c_proj: RowParallel (input parallel, output full)
    modules_[kCProjLayerName] = std::make_shared<nn::parallel::RowParallelLinear>(
        /*in_features=*/ffn_hidden, /*out_features=*/config.n_embd,
        /*bias=*/config.add_bias_linear,
        /*reduce_output=*/true,
        /*input_is_parallel=*/true,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MLP::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    bool is_swiglu = modules_.contains(kCFc2LayerName) && modules_.contains(kSiluLayerName);

    if (is_swiglu) {
        // SwiGLU forward pass
        // (B, T, C) -> ColumnParallelLinear(C, hidden_dim) -> (B, T, hidden_dim)
        auto x1 = (*modules_[kCFcLayerName])(x)[0];
        // (B, T, C) -> ColumnParallelLinear(C, hidden_dim) -> (B, T, hidden_dim)
        auto x2 = (*modules_[kCFc2LayerName])(x)[0];
        // (B, T, hidden_dim) -> SiLU -> (B, T, hidden_dim)
        x2 = (*modules_[kSiluLayerName])({x2})[0];
        // (B, T, hidden_dim) -> element-wise mul -> (B, T, hidden_dim)
        auto x3 = x1 * x2;
        // (B, T, hidden_dim) -> RowParallelLinear(hidden_dim, C) -> (B, T, C)
        auto x4 = (*modules_[kCProjLayerName])({x3});
        return x4;
    } else {
        // GELU forward pass (standard)
        // (B, T, C) -> ColumnParallelLinear(C, 4*C) -> (B, T, 4*C_local)
        auto x1 = (*modules_[kCFcLayerName])(x);
        // (B, T, 4*C_local) -> GELU -> (B, T, 4*C_local)
        auto x2 = (*modules_[kGeluLayerName])(x1);
        // (B, T, 4*C_local) -> RowParallelLinear(4*C, C) -> (B, T, C)
        auto x3 = (*modules_[kCProjLayerName])(x2);
        return x3;
    }
}

} // namespace infini_train::nn
