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
#include "infini_train/include/nn/modules/transformer/moe/moe_utils.h"
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
    ffn_hidden = (ffn_hidden + config.multiple_of - 1) / config.multiple_of * config.multiple_of;

    if (config.ffn_type == FFNType::kMoE) {
        const auto &moe_config = moe::RequireMoEConfig(config);
        CHECK_GT(moe_config.moe_ffn_hidden_size, 0);

        ffn_hidden = moe_config.moe_ffn_hidden_size;
    }
    CHECK_GT(ffn_hidden, 0);

    // c_fc: ColumnParallel (input full, output parallel)
    modules_[kCFcLayerName] = std::make_shared<parallel::ColumnParallelLinear>(
        /*in_features=*/config.n_embd,
        /*out_features=*/config.activation_type == MLPType::kSwiGLU ? 2 * ffn_hidden : ffn_hidden,
        /*bias=*/config.add_bias_linear,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/parallel::global::GetSequenceParallelEnabled());

    // Activation: check for GELU or SwiGLU
    if (config.activation_type == MLPType::kGELU) {
        modules_[kGeluLayerName] = std::make_shared<NewGELU>();
    } else if (config.activation_type == MLPType::kSwiGLU) {
        modules_[kSwiGLULayerName] = std::make_shared<SwiGLU>();
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
    if (modules_.contains(kSwiGLULayerName)) {
        // (B, T, C) -> ColumnParallelLinear(C, 2*H) -> (B, T, 2*H_local)
        auto packed = (*modules_[kCFcLayerName])(x)[0];
        // (B, T, 2*H_local) [up, gate] -> SwiGLU -> (B, T, H_local)
        auto activated = (*modules_[kSwiGLULayerName])({packed});
        // (B, T, H_local) -> RowParallelLinear(H, C) -> (B, T, C)
        return (*modules_[kCProjLayerName])(activated);
    }

    // GELU forward pass (standard)
    // (B, T, C) -> ColumnParallelLinear(C, 4*C) -> (B, T, 4*C_local)
    auto x1 = (*modules_[kCFcLayerName])(x);
    // (B, T, 4*C_local) -> GELU -> (B, T, 4*C_local)
    auto x2 = (*modules_[kGeluLayerName])(x1);
    // (B, T, 4*C_local) -> RowParallelLinear(4*C, C) -> (B, T, C)
    return (*modules_[kCProjLayerName])(x2);
}

} // namespace infini_train::nn
