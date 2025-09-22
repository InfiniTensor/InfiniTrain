#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

struct TPGPT2Config {
    int64_t block_size = 1024;
    int64_t vocab_size = 50304;
    int64_t original_vocab_size = 50257;
    int64_t n_layer = 12;
    int64_t n_head = 12;
    int64_t n_embd = 768;

    infini_train::nn::parallel::TensorParallelGroup tp_group;
};

class NewGELU : public infini_train::nn::CloneableModule<NewGELU> {
public:
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class TPCausalSelfAttention : public infini_train::nn::CloneableModule<TPCausalSelfAttention> {
public:
    static constexpr char kCAttnLayerName[] = "c_attn";
    static constexpr char kCProjLayerName[] = "c_proj";

    static constexpr char kParamBiasName[] = "bias";

    explicit TPCausalSelfAttention(const TPGPT2Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

private:
    TPGPT2Config config_;
    int64_t n_head_ = 0;
    int64_t n_embd_ = 0;

    int64_t local_n_head_ = 0;
};

class TPMLP : public infini_train::nn::CloneableModule<TPMLP> {
public:
    static constexpr char kCFcLayerName[] = "c_fc";
    static constexpr char kGeluLayerName[] = "gelu";
    static constexpr char kCProjLayerName[] = "c_proj";

    explicit TPMLP(const TPGPT2Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class TPBlock : public infini_train::nn::CloneableModule<TPBlock> {
public:
    static constexpr char kLn1LayerName[] = "ln_1";
    static constexpr char kAttnLayerName[] = "attn";
    static constexpr char kLn2LayerName[] = "ln_2";
    static constexpr char kMlpLayerName[] = "mlp";

    explicit TPBlock(const TPGPT2Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};

class TensorParallelGPT2 : public infini_train::nn::CloneableModule<TensorParallelGPT2> {
public:
    static constexpr char kWTELayerName[] = "wte";
    static constexpr char kWPELayerName[] = "wpe";
    static constexpr char kHLayerName[] = "h";
    static constexpr char kLnFLayerName[] = "ln_f";
    static constexpr char kTransformerLayerName[] = "transformer";
    static constexpr char kLMHeadLayerName[] = "lm_head";

    enum class ModelType : int8_t {
        kGPT2,
        kGPT2Medium,
        kGPT2Large,
        kGPT2XL,
    };

    explicit TensorParallelGPT2(const TPGPT2Config &config);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

    static std::shared_ptr<TensorParallelGPT2> FromPretrained(ModelType model_type);
    static std::shared_ptr<TensorParallelGPT2> FromLLMC(const std::string &filepath,
                                                        infini_train::nn::parallel::TensorParallelGroup tp_group);

private:
    TPGPT2Config config_;
};
