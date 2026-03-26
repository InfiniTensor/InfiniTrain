#include "infini_train/include/core/transformer/transformer_layer.h"

#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/core/transformer/transformer_model.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"
namespace infini_train::nn {
TransformerLayer::TransformerLayer(const nn::TransformerConfig &config, const ModuleSpec &spec)
    : CloneableModule(kType), attention_type_(config.attention_type) {
    modules_[kLn1LayerName] = build_module(config, spec.submodules_.at(kLn1LayerName));
    modules_[kAttnLayerName] = build_module(config, spec.submodules_.at(kAttnLayerName));
    modules_[kLn2LayerName] = build_module(config, spec.submodules_.at(kLn2LayerName));
    modules_[kMlpLayerName] = build_module(config, spec.submodules_.at(kMlpLayerName));
}

std::vector<std::shared_ptr<infini_train::Tensor>>
TransformerLayer::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd)
    auto ln1_out = (*modules_[kLn1LayerName])({x[0]})[0];

    std::shared_ptr<infini_train::Tensor> x1;
    // Build attention input
    if (attention_type_ == AttentionType::kRoPE) {
        // LLaMA3: {ln1_out, freqs_cis, start_pos, mask}
        const auto freqs_cis = x.size() > 1 ? x[1] : nullptr;
        const auto start_pos = x.size() > 2 ? x[2] : nullptr;
        const auto mask = x.size() > 3 ? x[3] : nullptr;
        auto attn_out = (*modules_[kAttnLayerName])({ln1_out, freqs_cis, start_pos, mask})[0];
        x1 = x[0] + attn_out;
    } else {
        // GPT2: {ln1_out}
        auto attn_out = (*modules_[kAttnLayerName])({ln1_out})[0];
        x1 = x[0] + attn_out;
    }

    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> MLP -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x2 = x1 + (*modules_[kMlpLayerName])((*modules_[kLn2LayerName])({x1}))[0];

    // (bs, seq_len, n_embd)
    return {x2};
}
} // namespace infini_train::nn
