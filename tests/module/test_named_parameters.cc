#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class ModuleNamedParametersTest : public test::InfiniTrainTest {};

TEST_P(ModuleNamedParametersTest, SupportsPrefixAndNonRecursiveLookup) {
    auto linear = std::make_shared<nn::Linear>(2, 3, /*bias=*/true, GetDevice());

    const auto parameters = linear->NamedParameters("model", false);
    const std::unordered_map<std::string, std::shared_ptr<Tensor>> by_name(parameters.begin(), parameters.end());

    ASSERT_EQ(by_name.size(), 2);
    EXPECT_EQ(by_name.at("model.weight"), linear->parameter(nn::Linear::kParamWeightName));
    EXPECT_EQ(by_name.at("model.bias"), linear->parameter(nn::Linear::kParamBiasName));
}

TEST_P(ModuleNamedParametersTest, SupportsRecursionAndSharedParameterDeduplication) {
    auto shared = std::make_shared<nn::Linear>(2, 3, /*bias=*/true, GetDevice());
    auto root = std::make_shared<nn::Sequential>(std::vector<std::shared_ptr<nn::Module>>{shared, shared});

    const auto deduplicated = root->NamedParameters("model");
    ASSERT_EQ(deduplicated.size(), 2);
    EXPECT_EQ(deduplicated[0].first, "model.0.bias");
    EXPECT_EQ(deduplicated[1].first, "model.0.weight");
    std::unordered_set<const Tensor *> tensors;
    for (const auto &[name, parameter] : deduplicated) {
        EXPECT_TRUE(name == "model.0.weight" || name == "model.0.bias" || name == "model.1.weight"
                    || name == "model.1.bias");
        tensors.insert(parameter.get());
    }
    EXPECT_TRUE(tensors.contains(shared->parameter(nn::Linear::kParamWeightName).get()));
    EXPECT_TRUE(tensors.contains(shared->parameter(nn::Linear::kParamBiasName).get()));

    const auto aliases = root->NamedParameters("model", true, false);
    const std::unordered_map<std::string, std::shared_ptr<Tensor>> by_name(aliases.begin(), aliases.end());
    ASSERT_EQ(by_name.size(), 4);
    EXPECT_EQ(by_name.at("model.0.weight"), by_name.at("model.1.weight"));
    EXPECT_EQ(by_name.at("model.0.bias"), by_name.at("model.1.bias"));
}

TEST_P(ModuleNamedParametersTest, ReturnsNestedParametersInStableNameOrder) {
    auto first = std::make_shared<nn::Linear>(2, 3, /*bias=*/false, GetDevice());
    auto second = std::make_shared<nn::Linear>(3, 4, /*bias=*/false, GetDevice());
    auto nested = std::make_shared<nn::Sequential>(std::vector<std::shared_ptr<nn::Module>>{first, second});
    auto root = std::make_shared<nn::Sequential>(
        std::vector<std::shared_ptr<nn::Module>>{std::make_shared<nn::Linear>(2, 2, false, GetDevice()), nested});

    const auto parameters = root->NamedParameters();
    const std::unordered_map<std::string, std::shared_ptr<Tensor>> by_name(parameters.begin(), parameters.end());

    ASSERT_EQ(by_name.size(), 3);
    ASSERT_EQ(parameters.size(), 3);
    EXPECT_EQ(parameters[0].first, "0.weight");
    EXPECT_EQ(parameters[1].first, "1.0.weight");
    EXPECT_EQ(parameters[2].first, "1.1.weight");
    EXPECT_TRUE(by_name.contains("0.weight"));
    EXPECT_TRUE(by_name.contains("1.0.weight"));
    EXPECT_TRUE(by_name.contains("1.1.weight"));
}

TEST_P(ModuleNamedParametersTest, SkipsNullSubmodules) {
    auto root = std::make_shared<nn::Sequential>(std::vector<std::shared_ptr<nn::Module>>{nullptr});

    EXPECT_TRUE(root->NamedParameters().empty());
}

INFINI_TRAIN_REGISTER_TEST(ModuleNamedParametersTest);
