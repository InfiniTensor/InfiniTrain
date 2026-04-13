#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "infini_train/include/nn/lora/lora_config.h"
#include "infini_train/include/nn/lora/lora_linear.h"
#include "infini_train/include/nn/lora/lora_utils.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;
using namespace infini_train::nn::lora;

class LoRATest : public infini_train::test::InfiniTrainTestP {};

// Helper: sum tensor values on any device by copying to CPU if needed.
static float TensorSum(const std::shared_ptr<Tensor> &t) {
    if (t->GetDevice().IsCPU()) {
        return t->EigenMatrix().sum();
    }
    auto cpu = std::make_shared<Tensor>(t->Dims(), t->Dtype(), Device(Device::DeviceType::kCPU, 0));
    cpu->CopyFrom(t);
    return cpu->EigenMatrix().sum();
}

TEST_P(LoRATest, LoRAConfigScaling) {
    LoRAConfig config;
    config.rank = 8;
    config.alpha = 16.0f;

    float expected_scaling = 16.0f / 8.0f;
    EXPECT_EQ(config.Scaling(), expected_scaling);
}

TEST_P(LoRATest, LoRAConfigShouldApply) {
    LoRAConfig config;
    config.rank = 8;
    config.alpha = 16.0f;

    EXPECT_TRUE(config.ShouldApplyLoRA("c_attn"));
    EXPECT_TRUE(config.ShouldApplyLoRA("transformer.h.0.attn.c_attn"));
    EXPECT_TRUE(config.ShouldApplyLoRA("c_proj"));
    EXPECT_FALSE(config.ShouldApplyLoRA("c_fc"));
    EXPECT_FALSE(config.ShouldApplyLoRA("random_layer"));
}

TEST_P(LoRATest, LoRALinearFromModel) {
    auto base_linear = std::make_shared<nn::Linear>(64, 128, /*bias=*/true, GetDevice());

    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;
    config.target_modules = {"Linear"};

    auto model = GetLoRAModel(base_linear, config);

    auto *lora_linear = dynamic_cast<LoRALinear *>(model.get());
    ASSERT_NE(lora_linear, nullptr);

    EXPECT_EQ(lora_linear->in_features(), 64);
    EXPECT_EQ(lora_linear->out_features(), 128);
    EXPECT_EQ(lora_linear->rank(), 4);

    auto lora_A = lora_linear->parameter(LoRALinear::kParamLoraAName);
    auto lora_B = lora_linear->parameter(LoRALinear::kParamLoraBName);
    auto weight = lora_linear->parameter(nn::Linear::kParamWeightName);

    EXPECT_EQ(lora_A->Dims()[0], config.rank);
    EXPECT_EQ(lora_A->Dims()[1], 64);
    EXPECT_EQ(lora_B->Dims()[0], 128);
    EXPECT_EQ(lora_B->Dims()[1], config.rank);

    EXPECT_FALSE(weight->requires_grad());
    EXPECT_TRUE(lora_A->requires_grad());
    EXPECT_TRUE(lora_B->requires_grad());

    auto params = lora_linear->LoRAParameters();
    EXPECT_EQ(params.size(), 2);
}

TEST_P(LoRATest, LoRALinearForward) {
    auto base_linear = std::make_shared<nn::Linear>(64, 128, /*bias=*/true, GetDevice());

    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;
    config.target_modules = {"Linear"};

    auto model = GetLoRAModel(base_linear, config);

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 10, 64}, DataType::kFLOAT32, GetDevice());

    auto output = (*model)({input})[0];

    EXPECT_EQ(output->Dims().size(), 3);
    EXPECT_EQ(output->Dims()[0], 2);
    EXPECT_EQ(output->Dims()[1], 10);
    EXPECT_EQ(output->Dims()[2], 128);
}

TEST_P(LoRATest, LoRALinearMerge) {
    auto base_linear = std::make_shared<nn::Linear>(32, 64, /*bias=*/false, GetDevice());

    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;
    config.target_modules = {"Linear"};

    auto model = GetLoRAModel(base_linear, config);

    auto *lora_linear = dynamic_cast<LoRALinear *>(model.get());
    ASSERT_NE(lora_linear, nullptr);

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 5, 32}, DataType::kFLOAT32, GetDevice());
    infini_train::test::FillSequentialTensor(input, 1.0f);

    auto output_before = (*model)({input})[0];
    float output_before_sum = TensorSum(output_before);

    EXPECT_FALSE(lora_linear->IsMerged());
    MergeLoRAWeights(model);
    EXPECT_TRUE(lora_linear->IsMerged());

    auto lora_A = lora_linear->parameter(LoRALinear::kParamLoraAName);
    auto lora_B = lora_linear->parameter(LoRALinear::kParamLoraBName);
    EXPECT_FALSE(lora_A->requires_grad());
    EXPECT_FALSE(lora_B->requires_grad());

    auto output_merged = (*model)({input})[0];
    float output_merged_sum = TensorSum(output_merged);
    EXPECT_NEAR(std::abs(output_before_sum - output_merged_sum), 0.0f, 1e-3);

    UnmergeLoRAWeights(model);
    EXPECT_FALSE(lora_linear->IsMerged());
    EXPECT_TRUE(lora_A->requires_grad());
    EXPECT_TRUE(lora_B->requires_grad());

    auto output_unmerged = (*model)({input})[0];
    EXPECT_EQ(output_before->Dims(), output_unmerged->Dims());
}

TEST_P(LoRATest, LoRAUtils) {
    auto base_linear = std::make_shared<nn::Linear>(32, 64, /*bias=*/true, GetDevice());

    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;
    config.target_modules = {"Linear"};

    auto model = GetLoRAModel(base_linear, config);

    auto lora_params = GetLoRAParameters(model);
    EXPECT_EQ(lora_params.size(), 2);

    int64_t trainable = CountTrainableParameters(model);
    int64_t expected_trainable = config.rank * 32 + 64 * config.rank;
    EXPECT_EQ(trainable, expected_trainable);

    int64_t total = CountTotalParameters(model);
    int64_t expected_total = 64 * 32 + 64 + config.rank * 32 + 64 * config.rank;
    EXPECT_EQ(total, expected_total);
}

TEST_P(LoRATest, ParseLoRATargetModules) {
    auto modules = ParseLoRATargetModules("c_attn");
    EXPECT_EQ(modules.size(), 1);
    EXPECT_TRUE(modules.count("c_attn"));

    modules = ParseLoRATargetModules("c_attn,c_proj,c_fc");
    EXPECT_EQ(modules.size(), 3);
    EXPECT_TRUE(modules.count("c_attn"));
    EXPECT_TRUE(modules.count("c_proj"));
    EXPECT_TRUE(modules.count("c_fc"));

    modules = ParseLoRATargetModules("c_attn, c_proj , c_fc");
    EXPECT_EQ(modules.size(), 3);

    modules = ParseLoRATargetModules("c_attn,,c_proj");
    EXPECT_EQ(modules.size(), 2);
}

TEST_P(LoRATest, ShouldApplyLoRAEdgeCases) {
    {
        LoRAConfig config{8, 16.0f, 0.0f, ParseLoRATargetModules("c_attn,attn.c_proj")};
        EXPECT_TRUE(config.ShouldApplyLoRA("attn.c_proj"));
        EXPECT_TRUE(config.ShouldApplyLoRA("transformer.h.0.attn.c_proj"));
        EXPECT_FALSE(config.ShouldApplyLoRA("mlp.c_proj"));
    }

    {
        LoRAConfig config{8, 16.0f, 0.0f, ParseLoRATargetModules("c_attn,mlp.c_proj")};
        EXPECT_FALSE(config.ShouldApplyLoRA("attn.c_proj"));
        EXPECT_TRUE(config.ShouldApplyLoRA("mlp.c_proj"));
    }

    {
        LoRAConfig config{8, 16.0f, 0.0f, ParseLoRATargetModules("c_attn,c_proj")};
        EXPECT_TRUE(config.ShouldApplyLoRA("transformer.h.0.attn.c_proj"));
        EXPECT_TRUE(config.ShouldApplyLoRA("transformer.h.0.mlp.c_proj"));
    }
}

TEST_P(LoRATest, FreezeUnfreeze) {
    auto base_linear = std::make_shared<nn::Linear>(64, 128, /*bias=*/true, GetDevice());

    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;
    config.target_modules = {"Linear"};

    auto model = GetLoRAModel(base_linear, config);

    auto all_params = model->Parameters();

    int64_t total_trainable = 0;
    for (const auto &p : all_params) {
        if (p->requires_grad()) {
            total_trainable += p->NumElements();
        }
    }
    int64_t expected = config.rank * 64 + 128 * config.rank;
    EXPECT_EQ(total_trainable, expected);

    FreezeBaseModel(model);

    int64_t after_freeze = 0;
    for (const auto &p : all_params) {
        if (p->requires_grad()) {
            after_freeze += p->NumElements();
        }
    }
    EXPECT_EQ(after_freeze, expected);

    UnfreezeModel(model);
    int64_t after_unfreeze = 0;
    for (const auto &p : all_params) {
        if (p->requires_grad()) {
            after_unfreeze += p->NumElements();
        }
    }
    int64_t expected_unfreeze = 64 * 128 + 128 + config.rank * 64 + 128 * config.rank;
    EXPECT_EQ(after_unfreeze, expected_unfreeze);
}

TEST_P(LoRATest, LoRAStateDict) {
    auto base_linear = std::make_shared<nn::Linear>(64, 128, /*bias=*/true, GetDevice());

    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;
    config.target_modules = {"Linear"};

    auto model = GetLoRAModel(base_linear, config);

    auto state_dict = model->StateDict();

    EXPECT_TRUE(state_dict.count("weight"));
    EXPECT_TRUE(state_dict.count("bias"));
    EXPECT_TRUE(state_dict.count("lora_A"));
    EXPECT_TRUE(state_dict.count("lora_B"));

    EXPECT_TRUE(state_dict.at("lora_A")->requires_grad());
    EXPECT_TRUE(state_dict.at("lora_B")->requires_grad());
    EXPECT_FALSE(state_dict.at("weight")->requires_grad());

    EXPECT_EQ(state_dict.at("lora_A")->Dims()[0], config.rank);
    EXPECT_EQ(state_dict.at("lora_A")->Dims()[1], 64);
    EXPECT_EQ(state_dict.at("lora_B")->Dims()[0], 128);
    EXPECT_EQ(state_dict.at("lora_B")->Dims()[1], config.rank);
}

TEST_P(LoRATest, GetLoRAModel) {
    auto base_linear = std::make_shared<nn::Linear>(64, 128, /*bias=*/true, GetDevice());

    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;
    config.target_modules = {"Linear"};

    auto model = GetLoRAModel(base_linear, config);

    EXPECT_NE(model, nullptr);

    auto lora_params = GetLoRAParameters(model);
    EXPECT_EQ(lora_params.size(), 2);

    int64_t total_elements = 0;
    for (const auto &t : lora_params) { total_elements += t->NumElements(); }
    int64_t expected_elements = config.rank * 64 + 128 * config.rank;
    EXPECT_EQ(total_elements, expected_elements);

    MergeLoRAWeights(model);
    auto *lora_mod = dynamic_cast<LoRALinear *>(model.get());
    EXPECT_NE(lora_mod, nullptr);
    EXPECT_FALSE(lora_mod->LoRAParameters()[0]->requires_grad());

    UnmergeLoRAWeights(model);
    EXPECT_TRUE(lora_mod->LoRAParameters()[0]->requires_grad());
}

TEST_P(LoRATest, MergeAndUnload) {
    auto base_linear = std::make_shared<nn::Linear>(64, 128, /*bias=*/true, GetDevice());
    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;
    config.target_modules = {"Linear"};
    auto model = GetLoRAModel(base_linear, config);

    EXPECT_NE(dynamic_cast<LoRALinear *>(model.get()), nullptr);

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 5, 64}, DataType::kFLOAT32, GetDevice());
    infini_train::test::FillSequentialTensor(input, 1.0f);
    auto output_before = (*model)({input})[0];
    float output_before_sum = TensorSum(output_before);

    auto unloaded_model = MergeAndUnload(model);
    EXPECT_NE(unloaded_model, nullptr);
    EXPECT_EQ(dynamic_cast<LoRALinear *>(unloaded_model.get()), nullptr);

    auto state_dict = unloaded_model->StateDict();
    for (const auto &[name, param] : state_dict) {
        EXPECT_EQ(name.find("lora_A"), std::string::npos);
        EXPECT_EQ(name.find("lora_B"), std::string::npos);
    }

    auto output_after = (*unloaded_model)({input})[0];
    float output_after_sum = TensorSum(output_after);
    EXPECT_NEAR(std::abs(output_before_sum - output_after_sum), 0.0f, 1e-3);

    for (const auto &param : unloaded_model->Parameters()) { EXPECT_TRUE(param->requires_grad()); }
}

INFINI_TRAIN_REGISTER_TEST(LoRATest);
