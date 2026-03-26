#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/models/decode_only_transformer/layer_specs.h"
#include "infini_train/include/core/transformer/activations/gelu.h"
#include "infini_train/include/core/transformer/activations/swiglu.h"
#include "infini_train/include/core/transformer/norms/layer_norm.h"
#include "infini_train/include/core/transformer/norms/rms_norm.h"
#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_builders.h"
#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/core/transformer/transformer_layer.h"
#include "infini_train/include/core/transformer/transformer_model.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;
namespace nn = infini_train::nn;

// ========== GPT2 Model Definition ==========
// Uses LayerNorm, GELU activation, standard multi-head attention
class GPT2 : public nn::TransformerModel {
public:
    static constexpr char kType[] = "GPT2";
    static constexpr char kTransformerModelName[] = "transformer";

    enum class ModelType : int8_t {
        kGPT2,
        kGPT2Medium,
        kGPT2Large,
        kGPT2XL,
    };

    explicit GPT2(const nn::TransformerConfig &config)
        : TransformerModel(config, BuildGPT2Spec(config)),
          stage_info_(nn::parallel::PipelineParallel::GetStageInfo(
              config_.n_layer, nn::parallel::global::GetPipelineParallelSize(), nn::parallel::pp_rank,
              nn::parallel::global::GetVirtualPipelineParallelSize())) {}

    static std::shared_ptr<GPT2> FromPretrained(ModelType model_type);
    static std::shared_ptr<GPT2> FromLLMC(const std::string &filepath);

    int GetChunkSize() const;

private:
    const infini_train::nn::parallel::StageInfo stage_info_;
};

// ========== LLaMA3 Model Definition ==========
// Uses RMSNorm, SwiGLU activation, GQA attention, RoPE positional encoding
class LLaMA3 : public nn::TransformerModel {
public:
    static constexpr char kType[] = "LLaMA3";
    static constexpr char kTransformerModelName[] = "transformer";

    enum class ModelType : int8_t {
        // TODO(zbl): more model type from huggingface
        kLLaMA3_1_8B,
        kLLaMA3_1_70B,
        kLLaMA3_2_1B,
        kLLaMA3_2_3B,
        kLLaMA3_3_70B,
    };

    explicit LLaMA3(const nn::TransformerConfig &config)
        : TransformerModel(config, BuildLLaMA3Spec(config)),
          stage_info_(nn::parallel::PipelineParallel::GetStageInfo(
              config_.n_layer, nn::parallel::global::GetPipelineParallelSize(), nn::parallel::pp_rank,
              nn::parallel::global::GetVirtualPipelineParallelSize())) {}

    static std::shared_ptr<LLaMA3> FromPretrained(ModelType model_type);
    static std::shared_ptr<LLaMA3> FromLLMC(const std::string &filepath);

    int GetChunkSize() const { return stage_info_.layer_ranges_per_chunk.size(); }

private:
    const infini_train::nn::parallel::StageInfo stage_info_;
};
