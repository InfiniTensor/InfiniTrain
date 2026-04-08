#include "infini_train/include/nn/modules/transformer/transformer.h"

#include <cmath>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/spec_utils.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
TransformerLayer::TransformerLayer(const nn::TransformerConfig &config, const ModuleSpec &spec)
    : CloneableModule(kType) {
    modules_[kLn1LayerName] = BuildModule(config, spec.submodules_.at(kLn1LayerName));
    modules_[kAttnLayerName] = BuildModule(config, spec.submodules_.at(kAttnLayerName));
    modules_[kLn2LayerName] = BuildModule(config, spec.submodules_.at(kLn2LayerName));
    modules_[kMlpLayerName] = BuildModule(config, spec.submodules_.at(kMlpLayerName));
}

std::vector<std::shared_ptr<Tensor>> TransformerLayer::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd)
    auto ln1_out = (*modules_[kLn1LayerName])({x[0]})[0];

    std::vector<std::shared_ptr<Tensor>> attn_input = {ln1_out};
    if (x.size() > 1) {
        attn_input.push_back(x[1]); // freqs_cis
    }
    if (x.size() > 2) {
        attn_input.push_back(x[2]); // start_pos
    }
    if (x.size() > 3) {
        attn_input.push_back(x[3]); // mask
    }

    auto attn_out = (*modules_[kAttnLayerName])(attn_input)[0];
    auto x1 = x[0] + attn_out;

    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> MLP -> (bs, seq_len, n_embd) -> Add -> (bs,
    // seq_len, n_embd)
    auto x2 = x1 + (*modules_[kMlpLayerName])((*modules_[kLn2LayerName])({x1}))[0];

    // (bs, seq_len, n_embd)
    return {x2};
}

} // namespace infini_train::nn
