#include <cmath>
#include <memory>
#include <optional>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"
#include "infini_train/include/nn/modules/transformer/mla_self_attention.h"
#include "infini_train/include/nn/modules/transformer/mlp.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/modules/transformer/utils.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;
namespace nn = infini_train::nn;

class TransformerModuleTest : public infini_train::test::InfiniTrainTest {};

TEST_P(TransformerModuleTest, Embedding) {
    SKIP_CPU();
    auto embedding = std::make_shared<nn::Embedding>(1000, 128, GetDevice());
    EXPECT_EQ(embedding->Parameters().size(), 1);

    auto weight = embedding->parameter(nn::Embedding::kParamWeightName);
    EXPECT_EQ(weight->Dims(), (std::vector<int64_t>{1000, 128}));

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 16}, DataType::kINT64, GetDevice());
    auto output = (*embedding)({input});
    EXPECT_EQ(output.size(), 1);
    EXPECT_EQ(output[0]->Dims(), (std::vector<int64_t>{2, 16, 128}));
}

TEST_P(TransformerModuleTest, LayerNorm) {
    SKIP_CPU();
    auto layernorm = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{64});
    layernorm->To(GetDevice());
    EXPECT_EQ(layernorm->Parameters().size(), 2);

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 8, 64}, DataType::kFLOAT32, GetDevice());
    auto output = (*layernorm)({input});
    EXPECT_EQ(output[0]->Dims(), input->Dims());
}

TEST_P(TransformerModuleTest, RMSNorm) {
    SKIP_CPU();
    auto rmsnorm = std::make_shared<nn::RMSNorm>(64);
    rmsnorm->To(GetDevice());
    EXPECT_EQ(rmsnorm->Parameters().size(), 1);

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 8, 64}, DataType::kFLOAT32, GetDevice());
    auto output = (*rmsnorm)({input});
    EXPECT_EQ(output[0]->Dims(), input->Dims());
}

TEST_P(TransformerModuleTest, GPT2MLP) {
    SKIP_CPU();
    nn::TransformerConfig config;
    config.n_embd = 64;
    config.activation_type = nn::MLPType::kGELU;
    config.ffn_expansion_ratio = 4.0f;
    config.add_bias_linear = true;

    auto mlp = std::make_shared<nn::MLP>(config);
    mlp->To(GetDevice());
    EXPECT_EQ(mlp->Parameters().size(), 4);

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 8, 64}, DataType::kFLOAT32, GetDevice());
    auto output = (*mlp)({input});
    EXPECT_EQ(output[0]->Dims()[2], 64);
}

TEST_P(TransformerModuleTest, SwiGLUMLP) {
    SKIP_CPU();
    nn::TransformerConfig config;
    config.n_embd = 64;
    config.activation_type = nn::MLPType::kSwiGLU;
    config.ffn_expansion_ratio = 4.0f;
    config.add_bias_linear = false;
    config.ffn_dim_multiplier = 1.5f;
    config.multiple_of = 256;

    auto mlp = std::make_shared<nn::MLP>(config);
    mlp->To(GetDevice());
    EXPECT_EQ(mlp->Parameters().size(), 3);

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 8, 64}, DataType::kFLOAT32, GetDevice());
    auto output = (*mlp)({input});
    EXPECT_EQ(output[0]->Dims()[2], 64);
}

TEST_P(TransformerModuleTest, StandardAttention) {
    SKIP_CPU();
    nn::TransformerConfig config;
    config.n_embd = 64;
    config.n_head = 4;
    config.n_kv_head = 4;
    config.attention_type = nn::AttentionType::kStandard;
    config.add_bias_linear = true;

    auto attn = std::make_shared<nn::CausalSelfAttention>(config);
    attn->To(GetDevice());
    EXPECT_EQ(attn->Parameters().size(), 4);

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 8, 64}, DataType::kFLOAT32, GetDevice());
    auto output = (*attn)({input});
    EXPECT_EQ(output[0]->Dims(), input->Dims());
}

TEST_P(TransformerModuleTest, MLAAttention) {
    SKIP_CPU();
    nn::TransformerConfig config;
    config.n_embd = 64;
    config.n_head = 4;
    config.block_size = 16;
    config.attention_type = nn::AttentionType::kStandard;
    config.add_bias_linear = true;
    config.multi_latent_attention = true;
    config.q_lora_rank = 32;
    config.kv_lora_rank = 32;
    config.qk_nope_head_dim = 8;
    config.qk_rope_head_dim = 8;
    config.v_head_dim = 16;

    auto attn = std::make_shared<nn::MLASelfAttention>(config);
    attn->To(GetDevice());
    EXPECT_FALSE(attn->Parameters().empty());
    EXPECT_EQ(attn->module(nn::MLASelfAttention::kLinearQDownProjLayerName).type(), nn::Linear::kType);
    EXPECT_EQ(attn->module(nn::MLASelfAttention::kLinearKVDownProjLayerName).type(), nn::Linear::kType);

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 8, 64}, DataType::kFLOAT32, GetDevice());
    auto output = (*attn)({input});
    EXPECT_EQ(output[0]->Dims(), input->Dims());

    auto tp_down_config = config;
    tp_down_config.q_down_proj_use_tp = true;
    tp_down_config.kv_down_proj_use_tp = true;
    auto tp_down_attn = std::make_shared<nn::MLASelfAttention>(tp_down_config);
    tp_down_attn->To(GetDevice());
    EXPECT_EQ(tp_down_attn->module(nn::MLASelfAttention::kLinearQDownProjLayerName).type(),
              nn::parallel::ColumnParallelLinear::kType);
    EXPECT_EQ(tp_down_attn->module(nn::MLASelfAttention::kLinearKVDownProjLayerName).type(),
              nn::parallel::ColumnParallelLinear::kType);
    output = (*tp_down_attn)({input});
    EXPECT_EQ(output[0]->Dims(), input->Dims());

    auto direct_q_config = config;
    direct_q_config.q_lora_rank = std::nullopt;
    auto direct_q_attn = std::make_shared<nn::MLASelfAttention>(direct_q_config);
    direct_q_attn->To(GetDevice());
    EXPECT_EQ(direct_q_attn->module(nn::MLASelfAttention::kLinearQProjLayerName).type(),
              nn::parallel::ColumnParallelLinear::kType);
    output = (*direct_q_attn)({input});
    EXPECT_EQ(output[0]->Dims(), input->Dims());
}

TEST_P(TransformerModuleTest, GPT2TransformerLayer) {
    SKIP_CPU();
    nn::TransformerConfig config;
    config.n_embd = 64;
    config.n_head = 4;
    config.n_kv_head = 4;
    config.n_layer = 1;

    auto layer = std::make_shared<nn::TransformerLayer>(config);
    layer->To(GetDevice());
    EXPECT_FALSE(layer->Parameters().empty());

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 8, 64}, DataType::kFLOAT32, GetDevice());
    auto output = (*layer)({input});
    EXPECT_EQ(output[0]->Dims(), input->Dims());
}

TEST_P(TransformerModuleTest, GPT2Model) {
    SKIP_CPU();
    nn::TransformerConfig config;
    config.n_layer = 2;
    config.n_head = 4;
    config.n_kv_head = 4;
    config.n_embd = 64;

    auto model = std::make_shared<nn::TransformerModel>(config);
    model->To(GetDevice());
    EXPECT_FALSE(model->Parameters().empty());
}

TEST_P(TransformerModuleTest, LLaMA3Model) {
    SKIP_CPU();
    nn::TransformerConfig config;
    config.n_layer = 2;
    config.n_head = 4;
    config.n_kv_head = 2;
    config.n_embd = 64;
    config.attention_type = nn::AttentionType::kRoPE;
    config.activation_type = nn::MLPType::kSwiGLU;
    config.norm_type = nn::NormType::kRMSNorm;
    config.add_bias_linear = false;
    config.tie_weights = false;

    auto model = std::make_shared<nn::TransformerModel>(config);
    model->To(GetDevice());
    EXPECT_FALSE(model->Parameters().empty());
}

TEST_P(TransformerModuleTest, RoPEUtils) {
    auto freqs_cis = PrecomputeFreqsCis(64, 128);
    EXPECT_EQ(freqs_cis->Dims().size(), 3);
    EXPECT_EQ(freqs_cis->Dims()[0], 128);
    EXPECT_EQ(freqs_cis->Dims()[1], 32);
    EXPECT_EQ(freqs_cis->Dims()[2], 2);
}

TEST_P(TransformerModuleTest, StateDict) {
    SKIP_CPU();
    nn::TransformerConfig config;
    config.n_layer = 1;
    config.n_head = 2;
    config.n_kv_head = 2;
    config.n_embd = 32;
    config.vocab_size = 1000;
    config.attention_type = nn::AttentionType::kStandard;
    config.activation_type = nn::MLPType::kGELU;
    config.norm_type = nn::NormType::kLayerNorm;
    config.add_bias_linear = true;

    auto model = std::make_shared<nn::TransformerModel>(config);
    model->To(GetDevice());
    auto state_dict = model->StateDict();
    auto params = model->Parameters();
    auto buffers = model->Buffers();

    EXPECT_FALSE(state_dict.empty());
    EXPECT_GE(state_dict.size(), params.size());
}

INFINI_TRAIN_REGISTER_TEST(TransformerModuleTest);
