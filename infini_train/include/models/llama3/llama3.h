#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/config.h"
#include "infini_train/include/nn/modules/transformer/transformer_kernel.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/tensor.h"

class SwiGLU : public infini_train::nn::CloneableModule<SwiGLU> {
public:
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

// TODO(zbl): implement fused kernel
class RMSNorm : public infini_train::nn::CloneableModule<RMSNorm> {
public:
    static constexpr char kParamWeightName[] = "weight";

    explicit RMSNorm(int64_t dim, float eps = 1e-6f,
                     const infini_train::Device *device = infini_train::DeviceManager::Instance()->GetDefaultDevice());

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    float eps_ = 1e-5f;
};

class CausalSelfAttention : public infini_train::nn::CloneableModule<CausalSelfAttention> {
public:
    static constexpr char kCAttnLayerName[] = "c_attn";
    static constexpr char kCProjLayerName[] = "c_proj";

    explicit CausalSelfAttention(const infini_train::nn::TransformerConfig &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    infini_train::nn::TransformerConfig config_;
    int64_t n_head_ = 0;
    int64_t n_embd_ = 0;
    int64_t n_kv_head_ = 0;
    int64_t n_rep_ = 0;
    int64_t head_dim_ = 0;
};

class MLP : public infini_train::nn::CloneableModule<MLP> {
public:
    static constexpr char kCFcLayerName[] = "c_fc";
    static constexpr char kCFc2LayerName[] = "c_fc2";
    static constexpr char kSiluLayerName[] = "silu";
    static constexpr char kCProjLayerName[] = "c_proj";

    explicit MLP(const infini_train::nn::TransformerConfig &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    int64_t hidden_dim_ = 0;
};

class Block : public infini_train::nn::CloneableModule<Block> {
public:
    static constexpr char kLn1LayerName[] = "ln_1";
    static constexpr char kAttnLayerName[] = "attn";
    static constexpr char kLn2LayerName[] = "ln_2";
    static constexpr char kMlpLayerName[] = "mlp";

    explicit Block(const infini_train::nn::TransformerConfig &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class LLaMA3Kernel : public infini_train::nn::TransformerKernel {
public:
    bool UseAbsolutePositionEmbedding() const override { return false; }

    std::shared_ptr<infini_train::nn::Module> MakeBlock(const infini_train::nn::TransformerConfig &config) override;

    std::shared_ptr<infini_train::nn::Module> MakeFinalNorm(const infini_train::nn::TransformerConfig &config) override;
};

class LLaMA3 : public infini_train::nn::CloneableModule<LLaMA3> {
public:
    static constexpr char kTransformerLayerName[] = "transformer";

    enum class ModelType : int8_t {
        // TODO(zbl): more model type from huggingface
        kLLaMA3_1_8B,
        kLLaMA3_1_70B,
        kLLaMA3_2_1B,
        kLLaMA3_2_3B,
        kLLaMA3_3_70B,
    };

    explicit LLaMA3(const infini_train::nn::TransformerConfig &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

    static std::shared_ptr<LLaMA3> FromPretrained(ModelType model_type);
    static std::shared_ptr<LLaMA3> FromLLMC(const std::string &filepath);

    int GetChunkSize() const { return stage_info_.layer_ranges_per_chunk.size(); }

private:
    const infini_train::nn::TransformerConfig config_;
    const infini_train::nn::parallel::StageInfo stage_info_;
};
