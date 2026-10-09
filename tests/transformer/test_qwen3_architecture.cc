#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "example/qwen3/config.h"
#include "infini_train/include/autocast.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/utils.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;
namespace nn = infini_train::nn;

class Qwen3ArchitectureTest : public infini_train::test::InfiniTrainTest {};

TEST_P(Qwen3ArchitectureTest, AttentionRegistersQKNormParameters) {
    SKIP_CPU();
    nn::TransformerConfig config;
    config.block_size = 16;
    config.vocab_size = 128;
    config.n_layer = 1;
    config.n_head = 4;
    config.n_kv_head = 2;
    config.n_embd = 32;
    config.qk_layernorm = true;

    auto attention = std::make_shared<nn::CausalSelfAttention>(config);
    attention->To(GetDevice());
    auto state_dict = attention->StateDict();

    EXPECT_TRUE(state_dict.contains(std::string(nn::CausalSelfAttention::kQNormLayerName) + "."
                                    + nn::RMSNorm::kParamWeightName));
    EXPECT_TRUE(state_dict.contains(std::string(nn::CausalSelfAttention::kKNormLayerName) + "."
                                    + nn::RMSNorm::kParamWeightName));
    EXPECT_EQ(attention->Parameters().size(), 6);
}

TEST_P(Qwen3ArchitectureTest, QwenStyleModelForward) {
    SKIP_CPU();
    nn::TransformerConfig config;
    config.block_size = 16;
    config.vocab_size = 128;
    config.n_layer = 1;
    config.n_head = 4;
    config.n_kv_head = 2;
    config.n_embd = 32;

    auto model = std::make_shared<nn::TransformerModel>(config);
    model->To(GetDevice());
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kINT64, GetDevice());

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
    auto freqs = std::make_shared<Tensor>(freqs_data, std::vector<int64_t>{2, 2, 2}, DataType::kFLOAT32, GetDevice());

    auto [interleaved_q, interleaved_k] = ApplyRotaryEmbedding(q, k, freqs, true);
    test::ExpectTensorFloatEqual(interleaved_q, {1.0f, 2.0f, 3.0f, 4.0f, -6.0f, 5.0f, -8.0f, 7.0f});
    test::ExpectTensorFloatEqual(interleaved_k, {9.0f, 10.0f, 11.0f, 12.0f, -14.0f, 13.0f, -16.0f, 15.0f});

    auto [half_split_q, half_split_k] = ApplyRotaryEmbedding(q, k, freqs, false);
    test::ExpectTensorFloatEqual(half_split_q, {1.0f, 2.0f, 3.0f, 4.0f, -7.0f, -8.0f, 5.0f, 6.0f});
    test::ExpectTensorFloatEqual(half_split_k, {9.0f, 10.0f, 11.0f, 12.0f, -15.0f, -16.0f, 13.0f, 14.0f});
}

// Exercise Q/K RMSNorm and half-split RoPE before native GQA, including gradients
// to the norm/projection parameters and a second forward after an optimizer update.
TEST_P(Qwen3ArchitectureTest, FlashMatchesUnfusedTraining) {
    if (!GetDevice().IsCUDA()
        || !Dispatcher::Instance().HasKernel({GetDevice().type(), "ScaledDotProductAttentionForward"})) {
        GTEST_SKIP() << "requires the CUDA FlashAttention backend";
    }

    for (const int64_t head_dim : {64, 128}) {
        SCOPED_TRACE(head_dim);
        auto config = qwen3::Qwen3Config();
        config.block_size = 32;
        config.vocab_size = config.original_vocab_size = 128;
        config.n_layer = 1;
        config.n_head = 4;
        config.n_kv_head = 1;
        config.n_embd = config.n_head * head_dim;
        qwen3::SanitizeQwen3Config(config);
        auto unfused = std::make_shared<nn::TransformerModel>(config);
        config.flash = true;
        auto flash = std::make_shared<nn::TransformerModel>(config);

        // Use identical, independent parameters with deterministic nontrivial values.
        std::mt19937 rng(42);
        std::normal_distribution<float> distribution(0.0f, 0.02f);
        for (const auto &[name, param] : unfused->NamedParameters()) {
            auto *data = static_cast<float *>(param->DataPtr());
            for (size_t i = 0; i < param->NumElements(); ++i) {
                data[i] = param->Dims().size() == 1 ? 1.0f : distribution(rng);
            }
        }
        flash->LoadStateDict(unfused->StateDict());
        unfused->To(GetDevice());
        flash->To(GetDevice());
        optimizers::Adam unfused_optimizer(unfused->Parameters(), 1e-4f);
        optimizers::Adam flash_optimizer(flash->Parameters(), 1e-4f);
        nn::CrossEntropyLoss loss_fn;

        constexpr int64_t batch = 2, sequence = 17;
        auto tokens_cpu = std::make_shared<Tensor>(std::vector<int64_t>{batch, sequence}, DataType::kINT64);
        auto targets_cpu = std::make_shared<Tensor>(std::vector<int64_t>{batch, sequence}, DataType::kINT64);
        for (int64_t i = 0; i < batch * sequence; ++i) {
            static_cast<int64_t *>(tokens_cpu->DataPtr())[i] = (i * 7) % config.vocab_size;
            static_cast<int64_t *>(targets_cpu->DataPtr())[i] = (i * 7 + 1) % config.vocab_size;
        }
        auto tokens = std::make_shared<Tensor>(tokens_cpu->To(GetDevice()));
        auto targets = std::make_shared<Tensor>(targets_cpu->To(GetDevice()));
        const auto unfused_params = unfused->NamedParameters();
        const auto flash_params = flash->NamedParameters();
        ASSERT_EQ(unfused_params.size(), flash_params.size());

        auto expect_relative_rmse
            = [](const std::shared_ptr<Tensor> &actual, const std::shared_ptr<Tensor> &expected, double tolerance) {
                  ASSERT_NE(actual, nullptr);
                  ASSERT_NE(expected, nullptr);
                  ASSERT_EQ(actual->Dims(), expected->Dims());
                  auto actual_cpu = actual->To(DataType::kFLOAT32).To(Device());
                  auto expected_cpu = expected->To(DataType::kFLOAT32).To(Device());
                  const auto *a = static_cast<const float *>(actual_cpu.DataPtr());
                  const auto *e = static_cast<const float *>(expected_cpu.DataPtr());
                  double squared_error = 0.0, squared_reference = 0.0;
                  for (size_t i = 0; i < actual_cpu.NumElements(); ++i) {
                      ASSERT_TRUE(std::isfinite(a[i]) && std::isfinite(e[i]));
                      squared_error += std::pow(static_cast<double>(a[i]) - e[i], 2);
                      squared_reference += static_cast<double>(e[i]) * e[i];
                  }
                  ASSERT_GT(squared_reference, 0.0);
                  EXPECT_LE(std::sqrt(squared_error / squared_reference), tolerance);
              };

        for (int step = 0; step < 2; ++step) {
            SCOPED_TRACE(step);
            unfused_optimizer.ZeroGrad();
            flash_optimizer.ZeroGrad();
            AutocastGuard autocast(GetDevice().type(), DataType::kBFLOAT16);
            auto expected = (*unfused)({tokens})[0];
            auto actual = (*flash)({tokens})[0];
            expect_relative_rmse(actual, expected, 0.02);
            auto expected_loss = loss_fn(
                {expected->View({batch * sequence, config.vocab_size}), targets->View({batch * sequence})})[0];
            auto actual_loss
                = loss_fn({actual->View({batch * sequence, config.vocab_size}), targets->View({batch * sequence})})[0];
            auto expected_loss_cpu = expected_loss->To(Device());
            test::ExpectTensorNear(actual_loss, *static_cast<const float *>(expected_loss_cpu.DataPtr()), 0.005f);
            expected_loss->Backward();
            actual_loss->Backward();
            for (size_t i = 0; i < unfused_params.size(); ++i) {
                SCOPED_TRACE(unfused_params[i].first);
                ASSERT_EQ(unfused_params[i].first, flash_params[i].first);
                expect_relative_rmse(flash_params[i].second->grad(), unfused_params[i].second->grad(), 0.08);
            }
            unfused_optimizer.Step();
            flash_optimizer.Step();
        }
    }
}

INFINI_TRAIN_REGISTER_TEST(Qwen3ArchitectureTest);
