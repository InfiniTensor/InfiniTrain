#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "example/qwen3/config.h"
#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/utils.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;
namespace nn = infini_train::nn;

class Qwen3ArchitectureTest : public infini_train::test::InfiniTrainTest {};

TEST_P(Qwen3ArchitectureTest, ConfigMatchesQwen3Architecture) {
    const auto config = qwen3::Qwen3Config();

    EXPECT_EQ(config.block_size, 40960);
    EXPECT_EQ(config.vocab_size, 151936);
    EXPECT_EQ(config.n_layer, 36);
    EXPECT_EQ(config.n_head, 32);
    EXPECT_EQ(config.n_kv_head, 8);
    EXPECT_EQ(config.n_embd, 4096);
    EXPECT_EQ(config.position_embedding_type, nn::PositionEmbeddingType::kRoPE);
    EXPECT_EQ(config.activation_type, nn::MLPType::kSwiGLU);
    EXPECT_EQ(config.norm_type, nn::NormType::kRMSNorm);
    EXPECT_FALSE(config.add_bias_linear);
    EXPECT_FALSE(config.add_bias_lm_head);
    EXPECT_FALSE(config.tie_weights);
    EXPECT_FLOAT_EQ(config.rope_theta, 1000000.0f);
    EXPECT_FALSE(config.rotary_interleaved);
    EXPECT_TRUE(config.use_qk_norm);
    EXPECT_FLOAT_EQ(config.qk_norm_eps, 1e-6f);
    EXPECT_FLOAT_EQ(config.norm_eps, 1e-6f);
}

TEST_P(Qwen3ArchitectureTest, AttentionRegistersQKNormParameters) {
    SKIP_CPU();
    auto config = qwen3::Qwen3Config();
    config.block_size = 16;
    config.vocab_size = 128;
    config.n_layer = 1;
    config.n_head = 4;
    config.n_kv_head = 2;
    config.n_embd = 32;

    auto attention = std::make_shared<nn::CausalSelfAttention>(config);
    attention->To(GetDevice());
    auto state_dict = attention->StateDict();

    EXPECT_TRUE(state_dict.contains(std::string(nn::CausalSelfAttention::kQNormLayerName) + "."
                                    + nn::RMSNorm::kParamWeightName));
    EXPECT_TRUE(state_dict.contains(std::string(nn::CausalSelfAttention::kKNormLayerName) + "."
                                    + nn::RMSNorm::kParamWeightName));
    EXPECT_EQ(attention->Parameters().size(), 4);
}

TEST_P(Qwen3ArchitectureTest, QwenStyleModelForward) {
    SKIP_CPU();
    auto config = qwen3::Qwen3Config();
    config.block_size = 16;
    config.vocab_size = 128;
    config.n_layer = 1;
    config.n_head = 4;
    config.n_kv_head = 2;
    config.n_embd = 32;

    auto model = std::make_shared<nn::TransformerModel>(config);
    model->To(GetDevice());
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kINT64,
                                          GetDevice());

    auto output = (*model)({input});
    ASSERT_EQ(output.size(), 1);
    EXPECT_EQ(output[0]->Dims(), (std::vector<int64_t>{2, 4, config.vocab_size}));
}

TEST_P(Qwen3ArchitectureTest, RotaryEmbeddingSupportsInterleavedAndHalfSplit) {
    const float q_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const float k_data[] = {9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    const float freqs_data[] = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f};
    const std::vector<int64_t> shape = {1, 2, 1, 4};

    auto q = std::make_shared<Tensor>(q_data, shape, DataType::kFLOAT32, GetDevice());
    auto k = std::make_shared<Tensor>(k_data, shape, DataType::kFLOAT32, GetDevice());
    auto freqs = std::make_shared<Tensor>(freqs_data, std::vector<int64_t>{2, 2, 2}, DataType::kFLOAT32,
                                          GetDevice());

    auto [interleaved_q, interleaved_k] = ApplyRotaryEmbedding(q, k, freqs, true);
    test::ExpectTensorFloatEqual(interleaved_q,
                                 {1.0f, 2.0f, 3.0f, 4.0f, -6.0f, 5.0f, -8.0f, 7.0f});
    test::ExpectTensorFloatEqual(interleaved_k,
                                 {9.0f, 10.0f, 11.0f, 12.0f, -14.0f, 13.0f, -16.0f, 15.0f});

    auto [half_split_q, half_split_k] = ApplyRotaryEmbedding(q, k, freqs, false);
    test::ExpectTensorFloatEqual(half_split_q, {1.0f, 2.0f, 3.0f, 4.0f, -7.0f, -8.0f, 5.0f, 6.0f});
    test::ExpectTensorFloatEqual(half_split_k, {9.0f, 10.0f, 11.0f, 12.0f, -15.0f, -16.0f, 13.0f, 14.0f});
}

INFINI_TRAIN_REGISTER_TEST(Qwen3ArchitectureTest);
