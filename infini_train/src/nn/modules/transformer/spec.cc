#include "infini_train/include/nn/modules/transformer/spec.h"

#include "example/common/utils.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/config.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
TransformerFirstStageABI::TransformerFirstStageABI(const TransformerConfig &config,
                                                   std::shared_ptr<TransformerKernel> kernel)
    : config_(config), kernel_(kernel) {
    // HF ABI: transformer.wte
    modules_[kWTELayerName] = std::make_shared<nn::parallel::VocabParallelEmbedding>(
        config_.vocab_size, config_.n_embd, nn::parallel::global::GetSequenceParallelEnabled());

    // GPT-2 has wpe, LLaMA does not
    if (kernel_->UseAbsolutePositionEmbedding()) {
        modules_[kWPELayerName] = std::make_shared<nn::Embedding>(config_.block_size, config_.n_embd);
    }

    // RoPE / ALiBi etc live in kernel, not ABI
}

std::vector<std::shared_ptr<Tensor>> TransformerFirstStageABI::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    auto tokens = x[0]; // (B, T)

    auto tok_emb = modules_[kWTELayerName]->Forward({tokens})[0]; // (B,T,C)

    std::shared_ptr<Tensor> h = tok_emb;

    if (modules_.count(kWPELayerName)) {
        auto pos = nn::init::Arange(0, tokens->Dims()[1], DataType::kINT64, tokens->GetDevice());
        auto pos_emb = modules_[kWPELayerName]->Forward({pos})[0];
        h = h + pos_emb;
    }

    // Let kernel add RoPE / ALiBi later inside attention
    return {h};
}

TransformerChunkABI::TransformerChunkABI(const TransformerConfig &config, int start_layer, int end_layer,
                                         std::shared_ptr<TransformerKernel> kernel)
    : config_(config) {
    std::vector<std::shared_ptr<nn::Module>> layers;

    for (int i = start_layer; i < end_layer; ++i) { layers.push_back(kernel->MakeBlock(config_)); }

    // HF ABI: transformer.h = ModuleList
    modules_[kHLayerName] = std::make_shared<nn::ModuleList>(std::move(layers));
}

std::vector<std::shared_ptr<Tensor>> TransformerChunkABI::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    auto h = x[0];
    auto &layers = *std::dynamic_pointer_cast<nn::ModuleList>(modules_[kHLayerName]);

    for (auto &layer : layers) { h = layer->Forward({h})[0]; }
    return {h};
}

TransformerLastStageABI::TransformerLastStageABI(const TransformerConfig &config,
                                                 std::shared_ptr<TransformerKernel> kernel)
    : config_(config) {
    // HF ABI: transformer.ln_f
    modules_[kLnFLayerName] = kernel->MakeFinalNorm(config_);

    // HF ABI: lm_head
    modules_[kLMHeadLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
        config_.n_embd, config_.vocab_size, false, false, false, false,
        nn::parallel::global::GetSequenceParallelEnabled());
}

std::vector<std::shared_ptr<Tensor>> TransformerLastStageABI::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    auto h = modules_[kLnFLayerName]->Forward(x);
    return modules_[kLMHeadLayerName]->Forward(h);
}

} // namespace infini_train::nn