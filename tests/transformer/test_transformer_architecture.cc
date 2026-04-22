#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "example/gpt2/config.h"
#include "example/llama3/config.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"
#include "infini_train/include/nn/modules/transformer/mlp.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/modules/transformer/utils.h"
#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;
namespace nn = infini_train::nn;

class TransformerConfigTest : public infini_train::test::InfiniTrainTest {};

TEST_P(TransformerConfigTest, GPT2Config) {
    auto config = gpt2::GPT2Config();
    EXPECT_EQ(config.attention_type, nn::AttentionType::kStandard);
    EXPECT_EQ(config.activation_type, nn::MLPType::kGELU);
    EXPECT_EQ(config.norm_type, nn::NormType::kLayerNorm);
    EXPECT_TRUE(config.add_bias_linear);
    EXPECT_TRUE(config.tie_weights);
    EXPECT_FALSE(config.UseGQA());
}

TEST_P(TransformerConfigTest, LLaMA3Config) {
    auto config = llama3::LLaMA3Config();
    EXPECT_EQ(config.attention_type, nn::AttentionType::kRoPE);
    EXPECT_EQ(config.activation_type, nn::MLPType::kSwiGLU);
    EXPECT_EQ(config.norm_type, nn::NormType::kRMSNorm);
    EXPECT_FALSE(config.add_bias_linear);
    EXPECT_FALSE(config.tie_weights);
    EXPECT_TRUE(config.UseGQA());
}

INFINI_TRAIN_REGISTER_TEST(TransformerConfigTest);

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

TEST_P(TransformerModuleTest, GPT2TransformerLayer) {
    SKIP_CPU();
    auto config = gpt2::GPT2Config();
    config.n_embd = 64;
    config.n_head = 4;
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
    auto config = gpt2::GPT2Config();
    config.n_layer = 2;
    config.n_head = 4;
    config.n_embd = 64;

    auto model = std::make_shared<nn::TransformerModel>(config);
    model->To(GetDevice());
    EXPECT_FALSE(model->Parameters().empty());
}

TEST_P(TransformerModuleTest, LLaMA3Model) {
    SKIP_CPU();
    auto config = llama3::LLaMA3Config();
    config.n_layer = 2;
    config.n_head = 4;
    config.n_kv_head = 2;
    config.n_embd = 64;

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
