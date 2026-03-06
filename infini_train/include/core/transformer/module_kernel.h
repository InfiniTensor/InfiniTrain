#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_block.h"
#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/core/transformer/transformer_layer.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;
namespace nn = infini_train::nn;

// ---------- spec for GPT2 ----------
// 为模型定制的spec，考虑单独放置在模型专属目录
inline nn::ModuleSpec BuildGPT2LayerSpec(const nn::TransformerConfig &config) {
    nn::ModuleSpec layer;

    // ===== first stage =====
    nn::ModuleSpec first_stage;

    first_stage.submodules_[nn::TransformerFirstStage::kWTELayerName]
        = nn::ModuleSpec(typeid(nn::parallel::VocabParallelEmbedding));
    first_stage.submodules_[nn::TransformerFirstStage::kWPELayerName] = nn::ModuleSpec(typeid(nn::Embedding));

    // ===== block =====
    nn::ModuleSpec block;
    block.submodules_[nn::TransformerBlock::kLn1LayerName] = nn::ModuleSpec(typeid(nn::LayerNorm));

    nn::ModuleSpec attn(typeid(nn::CausalSelfAttention));
    {
        nn::ModuleSpec attnColumnParallelLinearSpec = nn::ModuleSpec(typeid(nn::parallel::ColumnParallelLinear));
        {
            attnColumnParallelLinearSpec.params_["in"] = static_cast<int>(config.n_embd);
            attnColumnParallelLinearSpec.params_["out"] = static_cast<int>(3 * config.n_embd);
        }
        attn.submodules_[nn::CausalSelfAttention::kCAttnLayerName] = attnColumnParallelLinearSpec;
        nn::ModuleSpec attnRowParallelLinearSpec = nn::ModuleSpec(typeid(nn::parallel::RowParallelLinear));
        {
            attnRowParallelLinearSpec.params_["in"] = static_cast<int>(config.n_embd);
            attnRowParallelLinearSpec.params_["out"] = static_cast<int>(config.n_embd);
        }
        attn.submodules_[nn::CausalSelfAttention::kCProjLayerName] = attnRowParallelLinearSpec;
    }
    block.submodules_[nn::TransformerBlock::kAttnLayerName] = attn;

    block.submodules_[nn::TransformerBlock::kLn2LayerName] = nn::ModuleSpec(typeid(nn::LayerNorm));

    nn::ModuleSpec mlp(typeid(nn::MLP));
    {
        nn::ModuleSpec mlpColumnParallelLinearSpec = nn::ModuleSpec(typeid(nn::parallel::ColumnParallelLinear));
        {
            mlpColumnParallelLinearSpec.params_["in"] = static_cast<int>(config.n_embd);
            mlpColumnParallelLinearSpec.params_["out"] = static_cast<int>(4 * config.n_embd);
        }
        mlp.submodules_[nn::MLP::kCFcLayerName] = mlpColumnParallelLinearSpec;
        mlp.submodules_[nn::MLP::kGeluLayerName] = nn::ModuleSpec(typeid(nn::NewGELU));
        nn::ModuleSpec mlpRowParallelLinearSpec = nn::ModuleSpec(typeid(nn::parallel::RowParallelLinear));
        {
            mlpRowParallelLinearSpec.params_["in"] = static_cast<int>(4 * config.n_embd);
            mlpRowParallelLinearSpec.params_["out"] = static_cast<int>(config.n_embd);
        }
        mlp.submodules_[nn::MLP::kCProjLayerName] = mlpRowParallelLinearSpec;
    }
    block.submodules_[nn::TransformerBlock::kMlpLayerName] = mlp;

    // ===== last stage =====
    nn::ModuleSpec last_stage;
    {
        last_stage.submodules_[nn::TransformerLastStage::kLnFLayerName] = nn::ModuleSpec(typeid(nn::LayerNorm));
        nn::ModuleSpec lastStageColumnParallelLinearSpec = nn::ModuleSpec(typeid(nn::parallel::ColumnParallelLinear));
        {
            lastStageColumnParallelLinearSpec.params_["in"] = static_cast<int>(config.n_embd);
            lastStageColumnParallelLinearSpec.params_["out"] = static_cast<int>(config.vocab_size);
        }

        last_stage.submodules_[nn::TransformerLastStage::kLMHeadLayerName] = lastStageColumnParallelLinearSpec;
    }
    layer.submodules_["first_stage"] = first_stage;
    layer.submodules_["block"] = block;
    layer.submodules_["last_stage"] = last_stage;

    return layer;
}

// 考虑放在example/net.h
class GPT2 : public nn::TransformerLayer {
public:
    static constexpr char kType[] = "GPT2";
    static constexpr char kTransformerLayerName[] = "transformer";

    enum class ModelType : int8_t {
        kGPT2,
        kGPT2Medium,
        kGPT2Large,
        kGPT2XL,
    };

    explicit GPT2(const nn::TransformerConfig &config)
        : TransformerLayer(config, BuildGPT2LayerSpec(config)),
          stage_info_(nn::parallel::PipelineParallel::GetStageInfo(
              config_.n_layer, nn::parallel::global::GetPipelineParallelSize(), nn::parallel::pp_rank,
              nn::parallel::global::GetVirtualPipelineParallelSize())) {}

    static std::shared_ptr<GPT2> FromPretrained(ModelType model_type);
    static std::shared_ptr<GPT2> FromLLMC(const std::string &filepath);

    int GetChunkSize() const;

private:
    const infini_train::nn::parallel::StageInfo stage_info_;
};

// ---------- spec for LLaMA3 ----------
inline nn::ModuleSpec BuildLLaMA3LayerSpec() {
    nn::ModuleSpec block_subs;

    // ===== attention =====
    block_subs.submodules_["ln1"] = nn::ModuleSpec(typeid(nn::RMSNorm));

    block_subs.submodules_["attn"] = nn::ModuleSpec(typeid(nn::CausalSelfAttention));

    block_subs.submodules_["ln2"] = nn::ModuleSpec(typeid(nn::RMSNorm));

    block_subs.submodules_["mlp"] = nn::ModuleSpec(typeid(nn::MLP));

    nn::ModuleSpec layer_subs;
    layer_subs.submodules_["block"] = block_subs;

    nn::ModuleSpec spec;
    spec.submodules_["subs"] = layer_subs;

    return spec;
}

class LLaMA3 : public nn::TransformerLayer {
public:
    static constexpr char kType[] = "LLaMA3";
    static constexpr char kTransformerLayerName[] = "transformer";

    enum class ModelType : int8_t {
        // TODO(zbl): more model type from huggingface
        kLLaMA3_1_8B,
        kLLaMA3_1_70B,
        kLLaMA3_2_1B,
        kLLaMA3_2_3B,
        kLLaMA3_3_70B,
    };

    explicit LLaMA3(const nn::TransformerConfig &config)
        : TransformerLayer(config, BuildLLaMA3LayerSpec()),
          stage_info_(nn::parallel::PipelineParallel::GetStageInfo(
              config_.n_layer, nn::parallel::global::GetPipelineParallelSize(), nn::parallel::pp_rank,
              nn::parallel::global::GetVirtualPipelineParallelSize())) {}

    static std::shared_ptr<LLaMA3> FromPretrained(ModelType model_type);
    static std::shared_ptr<LLaMA3> FromLLMC(const std::string &filepath);

    int GetChunkSize() const { return stage_info_.layer_ranges_per_chunk.size(); }

private:
    const infini_train::nn::parallel::StageInfo stage_info_;
};
