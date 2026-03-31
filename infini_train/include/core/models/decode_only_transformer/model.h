#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/core/models/decode_only_transformer/layer_specs.h"
#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/core/transformer/transformer_model.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;
namespace nn = infini_train::nn;

class DecoderOnlyTransformer : public nn::TransformerModel {
public:
public:
    static constexpr char kType[] = "DecoderOnlyTransformer";
    static constexpr char kTransformerModelName[] = "transformer";

    enum class ModelType : int8_t {
        kGPT2,
        kGPT2Medium,
        kGPT2Large,
        kGPT2XL,
        // TODO(zbl): more model type from huggingface
        kLLaMA3_1_8B,
        kLLaMA3_1_70B,
        kLLaMA3_2_1B,
        kLLaMA3_2_3B,
        kLLaMA3_3_70B,
    };

    explicit DecoderOnlyTransformer(const nn::TransformerConfig &config)
        : TransformerModel(config, BuildModelSpec(config)),
          stage_info_(nn::parallel::PipelineParallel::GetStageInfo(
              Config().n_layer, nn::parallel::global::GetPipelineParallelSize(), nn::parallel::pp_rank,
              nn::parallel::global::GetVirtualPipelineParallelSize())) {}

    static std::shared_ptr<DecoderOnlyTransformer> FromPretrained(ModelType model_type);

    static std::shared_ptr<DecoderOnlyTransformer> FromLLMC_GPT2(const std::string &filepath);
    static std::shared_ptr<DecoderOnlyTransformer> FromLLMC_LLaMA3(const std::string &filepath);
    static void LoadWeightsFromLLMC(const std::string &filepath, DecoderOnlyTransformer *model,
                                    const std::string &weight_prefix);

    int GetChunkSize() const;

private:
    static nn::ModuleSpec BuildModelSpec(const nn::TransformerConfig &config) {
        return (config.model_type == nn::TransformerConfig::kGPT2Name) ? BuildGPT2Spec(config)
                                                                       : BuildLLaMA3Spec(config);
    }

    const infini_train::nn::parallel::StageInfo stage_info_;
};
