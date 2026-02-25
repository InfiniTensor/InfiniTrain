#pragma once

#include <functional>
#include <memory>

#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn {

class TransformerKernel {
public:
    virtual ~TransformerKernel() = default;

    // === Embedding / Position ===
    virtual bool UseAbsolutePositionEmbedding() const = 0;

    // === Block ===
    virtual std::shared_ptr<Module> MakeBlock(const TransformerConfig &config) = 0;

    // === Norm ===
    virtual std::shared_ptr<Module> MakeFinalNorm(const TransformerConfig &config) = 0;
};

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

/* Transformer:  spec_utils.h */
class Module;
struct BuildContext;

using ModuleBuilderFn = std::function<std::shared_ptr<Module>(const BuildContext &)>;

struct ModuleSpec {
    std::string name;
    ModuleBuilderFn builder;
    std::unordered_map<std::string, ModuleSpec> submodules;
};

std::shared_ptr<Module> build_module(const ModuleSpec &spec, const BuildContext &ctx);

} // namespace infini_train::nn
