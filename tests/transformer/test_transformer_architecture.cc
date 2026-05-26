#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/topk.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"
#include "infini_train/include/nn/modules/transformer/mlp.h"
#include "infini_train/include/nn/modules/transformer/moe/moe_layer.h"
#include "infini_train/include/nn/modules/transformer/moe/router.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/modules/transformer/utils.h"
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


TEST_P(TransformerModuleTest, MoELayerTop1) {
    nn::TransformerConfig config;
    config.n_embd = 32;
    config.n_head = 2;
    config.n_kv_head = 2;
    config.activation_type = nn::MLPType::kGELU;
    config.add_bias_linear = true;
    config.ffn_type = nn::FFNType::kMoE;
    config.moe_config = nn::MoEConfig{};
    config.moe_config->num_experts = 2;
    config.moe_config->router_topk = 1;
    config.moe_config->router_pre_softmax = true;

    auto moe = std::make_shared<nn::moe::MoELayer>(config);
    moe->To(GetDevice());
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 4, config.n_embd}, DataType::kFLOAT32, GetDevice());
    input->Uniform();

    auto output = (*moe)({input});
    ASSERT_EQ(output.size(), 1);
    EXPECT_EQ(output[0]->Dims(), input->Dims());
    EXPECT_FALSE(moe->Parameters().empty());
}

TEST_P(TransformerModuleTest, MoELayerTop2SwiGLU) {
    nn::TransformerConfig config;
    config.n_embd = 32;
    config.n_head = 2;
    config.n_kv_head = 2;
    config.activation_type = nn::MLPType::kSwiGLU;
    config.add_bias_linear = false;
    config.ffn_type = nn::FFNType::kMoE;
    config.moe_config = nn::MoEConfig{};
    config.moe_config->num_experts = 4;
    config.moe_config->router_topk = 2;
    config.moe_config->moe_ffn_hidden_size = 48;

    auto moe = std::make_shared<nn::moe::MoELayer>(config);
    moe->To(GetDevice());
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 4, config.n_embd}, DataType::kFLOAT32, GetDevice());
    input->Uniform();

    auto output = (*moe)({input});
    ASSERT_EQ(output.size(), 1);
    EXPECT_EQ(output[0]->Dims(), input->Dims());

    auto state = moe->StateDict();
    ASSERT_TRUE(state.contains("experts.expert_0.c_fc.weight"));
    ASSERT_TRUE(state.contains("experts.expert_0.c_fc2.weight"));
    ASSERT_TRUE(state.contains("experts.expert_0.c_proj.weight"));
    EXPECT_EQ(state.at("experts.expert_0.c_fc.weight")->Dims(), (std::vector<int64_t>{48, config.n_embd}));
    EXPECT_EQ(state.at("experts.expert_0.c_fc2.weight")->Dims(), (std::vector<int64_t>{48, config.n_embd}));
    EXPECT_EQ(state.at("experts.expert_0.c_proj.weight")->Dims(), (std::vector<int64_t>{config.n_embd, 48}));
}

TEST_P(TransformerModuleTest, TopKRouterMegatronOutputs) {
    nn::TransformerConfig config;
    config.n_embd = 32;
    config.add_bias_linear = false;
    config.ffn_type = nn::FFNType::kMoE;
    config.moe_config = nn::MoEConfig{};
    config.moe_config->num_experts = 4;
    config.moe_config->router_topk = 2;

    auto router = std::make_shared<nn::moe::TopKRouter>(config);
    router->To(GetDevice());
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 4, config.n_embd}, DataType::kFLOAT32, GetDevice());
    input->Uniform();

    auto output = (*router)({input});
    ASSERT_EQ(output.size(), 2);
    EXPECT_EQ(output[0]->Dims(), (std::vector<int64_t>{2, 4, 4}));
    EXPECT_EQ(output[1]->Dims(), (std::vector<int64_t>{2, 4, 4}));
    EXPECT_EQ(output[0]->Dtype(), DataType::kFLOAT32);
    EXPECT_EQ(output[1]->Dtype(), DataType::kBOOL);
}

TEST_P(TransformerModuleTest, TopKTorchInterface) {
    ONLY_CPU();
    const float data[] = {1.0f, 5.0f, 2.0f, 4.0f, 3.0f, 0.0f};
    auto input = std::make_shared<Tensor>(data, std::vector<int64_t>{2, 3}, DataType::kFLOAT32);

    auto largest_topk = std::make_shared<autograd::TopK>(2, 1, true, true);
    auto largest_values = largest_topk->Apply({input})[0];
    auto largest_indices = largest_topk->TopIndices();
    ASSERT_EQ(largest_values->Dims(), (std::vector<int64_t>{2, 2}));
    ASSERT_EQ(largest_indices->Dims(), (std::vector<int64_t>{2, 2}));
    const auto *largest_values_ptr = static_cast<const float *>(largest_values->DataPtr());
    const auto *largest_indices_ptr = static_cast<const int64_t *>(largest_indices->DataPtr());
    EXPECT_FLOAT_EQ(largest_values_ptr[0], 5.0f);
    EXPECT_FLOAT_EQ(largest_values_ptr[1], 2.0f);
    EXPECT_FLOAT_EQ(largest_values_ptr[2], 4.0f);
    EXPECT_FLOAT_EQ(largest_values_ptr[3], 3.0f);
    EXPECT_EQ(largest_indices_ptr[0], 1);
    EXPECT_EQ(largest_indices_ptr[1], 2);
    EXPECT_EQ(largest_indices_ptr[2], 0);
    EXPECT_EQ(largest_indices_ptr[3], 1);

    auto smallest_topk = std::make_shared<autograd::TopK>(1, 0, false, true);
    auto smallest_values = smallest_topk->Apply({input})[0];
    auto smallest_indices = smallest_topk->TopIndices();
    ASSERT_EQ(smallest_values->Dims(), (std::vector<int64_t>{1, 3}));
    ASSERT_EQ(smallest_indices->Dims(), (std::vector<int64_t>{1, 3}));
    const auto *smallest_values_ptr = static_cast<const float *>(smallest_values->DataPtr());
    const auto *smallest_indices_ptr = static_cast<const int64_t *>(smallest_indices->DataPtr());
    EXPECT_FLOAT_EQ(smallest_values_ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(smallest_values_ptr[1], 3.0f);
    EXPECT_FLOAT_EQ(smallest_values_ptr[2], 0.0f);
    EXPECT_EQ(smallest_indices_ptr[0], 0);
    EXPECT_EQ(smallest_indices_ptr[1], 1);
    EXPECT_EQ(smallest_indices_ptr[2], 1);
}

TEST_P(TransformerModuleTest, TopKRouterNormalization) {
    ONLY_CPU();
    auto make_router = [](nn::MoEConfig::RouterScoreFunction score_function, bool pre_softmax) {
        nn::TransformerConfig config;
        config.n_embd = 2;
        config.add_bias_linear = false;
        config.ffn_type = nn::FFNType::kMoE;
        config.moe_config = nn::MoEConfig{};
        config.moe_config->num_experts = 3;
        config.moe_config->router_topk = 2;
        config.moe_config->router_score_function = score_function;
        config.moe_config->router_pre_softmax = pre_softmax;
        auto router = std::make_shared<nn::moe::TopKRouter>(config);
        auto weight = router->parameter(nn::moe::TopKRouter::kParamWeightName);
        auto *weight_ptr = static_cast<float *>(weight->DataPtr());
        weight_ptr[0] = 1.0f;
        weight_ptr[1] = 0.0f;
        weight_ptr[2] = 2.0f;
        weight_ptr[3] = 0.0f;
        weight_ptr[4] = 0.0f;
        weight_ptr[5] = 0.0f;
        return router;
    };

    const float input_data[] = {1.0f, 1.0f};
    auto input = std::make_shared<Tensor>(input_data, std::vector<int64_t>{1, 1, 2}, DataType::kFLOAT32);

    auto softmax_router = make_router(nn::MoEConfig::RouterScoreFunction::kSoftmax, false);
    auto softmax_output = (*softmax_router)({input});
    const auto *softmax_probs = static_cast<const float *>(softmax_output[0]->DataPtr());
    EXPECT_NEAR(softmax_probs[0] + softmax_probs[1] + softmax_probs[2], 1.0f, 1e-5f);
    EXPECT_GT(softmax_probs[1], softmax_probs[0]);
    EXPECT_FLOAT_EQ(softmax_probs[2], 0.0f);

    auto sigmoid_router = make_router(nn::MoEConfig::RouterScoreFunction::kSigmoid, true);
    auto sigmoid_output = (*sigmoid_router)({input});
    const auto *sigmoid_probs = static_cast<const float *>(sigmoid_output[0]->DataPtr());
    EXPECT_NEAR(sigmoid_probs[0] + sigmoid_probs[1] + sigmoid_probs[2], 1.0f, 1e-5f);
    EXPECT_GT(sigmoid_probs[1], sigmoid_probs[0]);
    EXPECT_FLOAT_EQ(sigmoid_probs[2], 0.0f);
}

INFINI_TRAIN_REGISTER_TEST(TransformerModuleTest);
