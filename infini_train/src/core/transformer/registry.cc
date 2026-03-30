#include "infini_train/include/nn/modules/transformer.h"

#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_builders.h"
#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/core/transformer/transformer_model.h"
#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/causal_self_attention.h"
#include "infini_train/include/nn/modules/mlp.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
// ========== Module Registration using INFINI_TRAIN_REGISTER_MODULE macro ==========
INFINI_TRAIN_REGISTER_MODULE(TransformerLayer);
INFINI_TRAIN_REGISTER_MODULE(CausalSelfAttention);
INFINI_TRAIN_REGISTER_MODULE(MLP);

// NewGELU
INFINI_TRAIN_REGISTER_MODULE_CUSTOM(NewGELU, [](const TransformerConfig &config, const ModuleSpec &) {
    return std::make_shared<NewGELU>();
});

// SwiGLU
INFINI_TRAIN_REGISTER_MODULE_CUSTOM(SwiGLU, [](const TransformerConfig &config, const ModuleSpec &) {
    return std::make_shared<SwiGLU>();
});

INFINI_TRAIN_REGISTER_MODULE_CUSTOM(RMSNorm, [](const TransformerConfig &config, const ModuleSpec &spec) {
    int64_t dim = GetRequiredParam<int>(spec, kDim);
    float eps = GetRequiredParam<float>(spec, kEps);
    return std::make_shared<RMSNorm>(dim, eps);
});

// LayerNorm registration with custom config
INFINI_TRAIN_REGISTER_MODULE_CUSTOM(LayerNorm, [](const TransformerConfig &config, const ModuleSpec &spec) {
    auto normalized_shape = GetRequiredParam<std::vector<int64_t>>(spec, kNormalizedShape);
    return std::make_shared<LayerNorm>(normalized_shape);
});

// Embedding registration with params from spec
INFINI_TRAIN_REGISTER_MODULE_CUSTOM(Embedding, [](const TransformerConfig &config, const ModuleSpec &spec) {
    int num_embeddings = GetRequiredParam<int>(spec, kNumEmbeddings);
    int embedding_dim = GetRequiredParam<int>(spec, kEmbeddingDim);
    return std::make_shared<Embedding>(num_embeddings, embedding_dim);
});

namespace parallel {
// ColumnParallelLinear registration with params from spec
INFINI_TRAIN_REGISTER_MODULE_CUSTOM(ColumnParallelLinear, [](const TransformerConfig &config, const ModuleSpec &spec) {
    int in = GetRequiredParam<int>(spec, kInFeatures);
    int out = GetRequiredParam<int>(spec, kOutFeatures);
    bool bias = GetRequiredParam<bool>(spec, kBias);
    return std::make_shared<ColumnParallelLinear>(
        /*in_features=*/in,
        /*out_features=*/out,
        /*bias=*/bias,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/global::GetSequenceParallelEnabled());
});

// RowParallelLinear registration with params from spec
INFINI_TRAIN_REGISTER_MODULE_CUSTOM(RowParallelLinear, [](const TransformerConfig &config, const ModuleSpec &spec) {
    int in = GetRequiredParam<int>(spec, kInFeatures);
    int out = GetRequiredParam<int>(spec, kOutFeatures);
    bool bias = GetRequiredParam<bool>(spec, kBias);
    return std::make_shared<RowParallelLinear>(
        /*in_features=*/in,
        /*out_features=*/out,
        /*bias=*/bias,
        /*reduce_output=*/true,
        /*input_is_parallel=*/true,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/global::GetSequenceParallelEnabled());
});

// VocabParallelEmbedding registration with params from spec
INFINI_TRAIN_REGISTER_MODULE_CUSTOM(VocabParallelEmbedding, [](const TransformerConfig &config,
                                                               const ModuleSpec &spec) {
    int num_embeddings = GetRequiredParam<int>(spec, kNumEmbeddings);
    int embedding_dim = GetRequiredParam<int>(spec, kEmbeddingDim);
    return std::make_shared<VocabParallelEmbedding>(num_embeddings, embedding_dim,
                                                    /*reduce_scatter_embeddings=*/global::GetSequenceParallelEnabled());
});
} // namespace parallel
} // namespace infini_train::nn
