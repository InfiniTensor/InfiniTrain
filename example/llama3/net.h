#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/tensor.h"

struct LLaMA3Config {
    // ref: https://huggingface.co/meta-llama/Llama-3.2-1B
    // Model basic config
    int64_t block_size = 8192;   // Max seq_len
    int64_t vocab_size = 128256; // Vocab size
    int64_t n_layer = 16;        // Num of transformer layers
    int64_t n_head = 32;         // Num of heads in MHA
    int64_t n_kv_head = 8;       // Num of Key/Value heads（< n_head if using GQA）
    int64_t n_embd = 2048;       // Hidden size

    // FFN config
    std::optional<float> ffn_dim_multiplier = 1.5f; // FFN dim multiplier
    int64_t multiple_of = 256;                      // FFN dims must be multiple of this number

    // Pos embedding
    float rope_theta = 500000.0f; // theta in RoPE
    bool use_scaled_rope = false; // scaled RoPE

    // RMSNorm
    float norm_eps = 1e-5f; // epsilon in RMSNorm

    // Inference
    bool use_kv = false;            // kv cache
    bool flash = false;             // flash attention
    int64_t max_gen_batch_size = 4; // max batch size during inference
};

class SwiGLU : public infini_train::nn::CloneableModule<SwiGLU> {
public:
    static constexpr char kType[] = "SwiGLU";
    SwiGLU() : CloneableModule(kType) {}

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

// TODO(zbl): implement fused kernel
class RMSNorm : public infini_train::nn::CloneableModule<RMSNorm> {
public:
    static constexpr char kType[] = "RMSNorm";
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
    static constexpr char kType[] = "CausalSelfAttention";
    static constexpr char kCAttnLayerName[] = "c_attn";
    static constexpr char kCProjLayerName[] = "c_proj";

    explicit CausalSelfAttention(const LLaMA3Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    LLaMA3Config config_;
    int64_t n_head_ = 0;
    int64_t n_embd_ = 0;
    int64_t n_kv_head_ = 0;
    int64_t n_rep_ = 0;
    int64_t head_dim_ = 0;
};

class MLP : public infini_train::nn::CloneableModule<MLP> {
public:
    static constexpr char kType[] = "MLP";
    static constexpr char kCFcLayerName[] = "c_fc";
    static constexpr char kCFc2LayerName[] = "c_fc2";
    static constexpr char kSiluLayerName[] = "silu";
    static constexpr char kCProjLayerName[] = "c_proj";

    explicit MLP(const LLaMA3Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    int64_t hidden_dim_ = 0;
};

class Block : public infini_train::nn::CloneableModule<Block> {
public:
    static constexpr char kType[] = "Block";
    static constexpr char kLn1LayerName[] = "ln_1";
    static constexpr char kAttnLayerName[] = "attn";
    static constexpr char kLn2LayerName[] = "ln_2";
    static constexpr char kMlpLayerName[] = "mlp";

    explicit Block(const LLaMA3Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class LLaMA3FirstStage : public infini_train::nn::CloneableModule<LLaMA3FirstStage> {
public:
    static constexpr char kType[] = "LLaMA3FirstStage";
    static constexpr char kWTELayerName[] = "wte";

    explicit LLaMA3FirstStage(const LLaMA3Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    const LLaMA3Config config_;
};

class LLaMA3Chunk : public infini_train::nn::CloneableModule<LLaMA3Chunk> {
public:
    static constexpr char kType[] = "LLaMA3Chunk";
    static constexpr char kHLayerName[] = "h";
    static constexpr char kFreqsCisName[] = "freqs_cis";

    LLaMA3Chunk(const LLaMA3Config &config, int start_layer, int end_layer);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    const LLaMA3Config config_;
};

class LLaMA3LastStage : public infini_train::nn::CloneableModule<LLaMA3LastStage> {
public:
    static constexpr char kType[] = "LLaMA3LastStage";
    static constexpr char kLnFLayerName[] = "ln_f";
    static constexpr char kLMHeadLayerName[] = "lm_head";

    explicit LLaMA3LastStage(const LLaMA3Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    const LLaMA3Config config_;
};

class LLaMA3 : public infini_train::nn::CloneableModule<LLaMA3> {
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

    explicit LLaMA3(const LLaMA3Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

    static std::shared_ptr<LLaMA3> FromPretrained(ModelType model_type);
    static std::shared_ptr<LLaMA3> FromLLMC(const std::string &filepath);

    int GetChunkSize() const { return stage_info_.layer_ranges_per_chunk.size(); }

private:
    const LLaMA3Config config_;
    const infini_train::nn::parallel::StageInfo stage_info_;
};
