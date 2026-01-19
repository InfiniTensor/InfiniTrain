#pragma once

#include <functional>
#include <memory>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/config.h"
#include "infini_train/include/nn/modules/transformer/transformer_kernel.h"

namespace infini_train::nn {

class TransformerKernel;

class TransformerFirstStageABI : public nn::CloneableModule<TransformerFirstStageABI> {
public:
    static constexpr char kWTELayerName[] = "wte";
    static constexpr char kWPELayerName[] = "wpe"; // GPT-2 only

    TransformerFirstStageABI(const TransformerConfig &config, std::shared_ptr<TransformerKernel> kernel);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &x) override;

private:
    const TransformerConfig config_;
    std::shared_ptr<TransformerKernel> kernel_;
};

class TransformerChunkABI : public nn::CloneableModule<TransformerChunkABI> {
public:
    static constexpr char kHLayerName[] = "h"; // HF ABI

    TransformerChunkABI(const TransformerConfig &config, int start_layer, int end_layer,
                        std::shared_ptr<TransformerKernel> kernel);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &x) override;

private:
    const TransformerConfig config_;
};

class TransformerLastStageABI : public nn::CloneableModule<TransformerLastStageABI> {
public:
    static constexpr char kLnFLayerName[] = "ln_f";
    static constexpr char kLMHeadLayerName[] = "lm_head";

    TransformerLastStageABI(const TransformerConfig &config, std::shared_ptr<TransformerKernel> kernel);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &x) override;

private:
    const TransformerConfig config_;
};

// struct TransformerSpec {
//     // embedding
//     std::function<std::shared_ptr<Module>(const TransformerConfig &)> make_token_embedding;
//     std::function<std::shared_ptr<Module>(const TransformerConfig &)> make_position_embedding;

//     // per-layer
//     std::function<std::shared_ptr<Module>(const TransformerConfig &)> make_norm;
//     std::function<std::shared_ptr<Module>(const TransformerConfig &)> make_attention;
//     std::function<std::shared_ptr<Module>(const TransformerConfig &)> make_mlp;

//     // output
//     std::function<std::shared_ptr<Module>(const TransformerConfig &)> make_final_norm;
//     std::function<std::shared_ptr<Module>(const TransformerConfig &)> make_lm_head;
// };
} // namespace infini_train::nn
