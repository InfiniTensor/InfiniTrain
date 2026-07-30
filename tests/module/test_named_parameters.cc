#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {

class NamedParameterModule final : public nn::Module {
public:
    void AddParameter(const std::string &name, const std::shared_ptr<Tensor> &parameter) {
        parameters_[name] = parameter;
    }

    void AddModule(const std::string &name, const std::shared_ptr<nn::Module> &module) { modules_[name] = module; }
};

std::shared_ptr<Tensor> MakeParameter(Device device) {
    return std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, device);
}

} // namespace

class ModuleNamedParametersTest : public test::InfiniTrainTest {};

TEST_P(ModuleNamedParametersTest, SupportsPrefixRecursionAndSharedParameterDeduplication) {
    auto root = std::make_shared<NamedParameterModule>();
    auto child = std::make_shared<NamedParameterModule>();
    auto grandchild = std::make_shared<NamedParameterModule>();
    auto shared = MakeParameter(GetDevice());
    auto child_weight = MakeParameter(GetDevice());
    auto grandchild_weight = MakeParameter(GetDevice());

    root->AddParameter("root_weight", shared);
    child->AddParameter("alias", shared);
    child->AddParameter("weight", child_weight);
    grandchild->AddParameter("weight", grandchild_weight);
    child->AddModule("grandchild", grandchild);
    root->AddModule("child", child);

    const auto local = root->NamedParameters("model", false);
    ASSERT_EQ(local.size(), 1);
    EXPECT_EQ(local[0].first, "model.root_weight");
    EXPECT_EQ(local[0].second, shared);

    const auto deduplicated = root->NamedParameters("model");
    ASSERT_EQ(deduplicated.size(), 3);
    EXPECT_EQ(deduplicated[0].first, "model.root_weight");
    EXPECT_EQ(deduplicated[1].first, "model.child.weight");
    EXPECT_EQ(deduplicated[2].first, "model.child.grandchild.weight");

    const auto aliases = root->NamedParameters("model", true, false);
    ASSERT_EQ(aliases.size(), 4);
    EXPECT_EQ(aliases[0].first, "model.root_weight");
    EXPECT_EQ(aliases[1].first, "model.child.alias");
    EXPECT_EQ(aliases[1].second, shared);
}

TEST_P(ModuleNamedParametersTest, ProducesDeterministicLexicalOrder) {
    auto root = std::make_shared<NamedParameterModule>();
    auto first_child = std::make_shared<NamedParameterModule>();
    auto second_child = std::make_shared<NamedParameterModule>();

    root->AddParameter("z", MakeParameter(GetDevice()));
    root->AddParameter("a", MakeParameter(GetDevice()));
    first_child->AddParameter("weight", MakeParameter(GetDevice()));
    second_child->AddParameter("weight", MakeParameter(GetDevice()));
    root->AddModule("z_child", second_child);
    root->AddModule("a_child", first_child);

    const auto parameters = root->NamedParameters();
    ASSERT_EQ(parameters.size(), 4);
    EXPECT_EQ(parameters[0].first, "a");
    EXPECT_EQ(parameters[1].first, "z");
    EXPECT_EQ(parameters[2].first, "a_child.weight");
    EXPECT_EQ(parameters[3].first, "z_child.weight");
}

TEST_P(ModuleNamedParametersTest, SkipsNullEntries) {
    auto root = std::make_shared<NamedParameterModule>();
    root->AddParameter("missing", nullptr);
    root->AddModule("missing_child", nullptr);

    EXPECT_TRUE(root->NamedParameters().empty());
}

INFINI_TRAIN_REGISTER_TEST(ModuleNamedParametersTest);
