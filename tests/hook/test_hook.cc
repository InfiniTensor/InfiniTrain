#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/autograd/function.h"
#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/common/hook.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class HookTest : public infini_train::test::InfiniTrainTest {};

class TestModule : public nn::Module {
public:
    TestModule() : Module("TestModule") {}
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override {
        return inputs;
    }
};

TEST_P(HookTest, BasicModuleHooks) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    x->set_requires_grad(true);

    auto module = std::make_shared<TestModule>();
    auto pre_hook = module->RegisterForwardPreHook([](nn::Module *, const std::vector<std::shared_ptr<Tensor>> &) {});
    auto fwd_hook = module->RegisterForwardPostHook([](nn::Module *, const std::vector<std::shared_ptr<Tensor>> &,
                                                       const std::vector<std::shared_ptr<Tensor>> &) {});
    auto bwd_pre_hook
        = module->RegisterBackwardPreHook([](nn::Module *, const std::vector<std::shared_ptr<Tensor>> &) {});
    auto bwd_post_hook = module->RegisterBackwardPostHook([](nn::Module *, const std::vector<std::shared_ptr<Tensor>> &,
                                                             const std::vector<std::shared_ptr<Tensor>> &) {});

    auto outputs = (*module)({x});
    EXPECT_EQ(outputs.size(), 1);
}

TEST_P(HookTest, HookRemove) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice());
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice());
    a->set_requires_grad(true);
    b->set_requires_grad(true);

    int hook1_count = 0, hook2_count = 0, hook3_count = 0;
    auto add_fn = std::make_shared<autograd::Add>();

    auto handle1 = add_fn->RegisterForwardPreHook(
        [&hook1_count](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &) { hook1_count++; });
    auto handle2 = add_fn->RegisterForwardPreHook(
        [&hook2_count](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &) { hook2_count++; });
    auto handle3 = add_fn->RegisterForwardPreHook(
        [&hook3_count](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &) { hook3_count++; });

    add_fn->Apply({a, b});
    EXPECT_EQ(hook1_count, 1);
    EXPECT_EQ(hook2_count, 1);
    EXPECT_EQ(hook3_count, 1);

    handle2->Remove();
    add_fn->Apply({a, b});
    EXPECT_EQ(hook1_count, 2);
    EXPECT_EQ(hook2_count, 1);
    EXPECT_EQ(hook3_count, 2);

    handle1->Remove();
    add_fn->Apply({a, b});
    EXPECT_EQ(hook1_count, 2);
    EXPECT_EQ(hook2_count, 1);
    EXPECT_EQ(hook3_count, 3);
}

TEST_P(HookTest, ModuleRegistriesPreserveInsertionOrder) {
    auto module = std::make_shared<TestModule>();
    auto first = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice());
    auto second = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice());
    auto replacement = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice());

    module->RegisterParameter("first", first);
    module->RegisterParameter("second", second);
    module->RegisterParameter("first", replacement);
    module->RegisterParameter("deferred", nullptr);

    auto direct_params = module->Parameters(/*recurse=*/false);
    ASSERT_EQ(direct_params.size(), 2);
    EXPECT_EQ(direct_params[0], replacement);
    EXPECT_EQ(direct_params[1], second);

    auto child_10 = std::make_shared<TestModule>();
    auto child_2 = std::make_shared<TestModule>();
    auto child_1 = std::make_shared<TestModule>();
    auto child_10_param = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice());
    auto child_2_param = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice());
    auto child_1_param = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice());
    child_10->RegisterParameter("weight", child_10_param);
    child_2->RegisterParameter("weight", child_2_param);
    child_1->RegisterParameter("weight", child_1_param);
    module->RegisterModule("10", child_10);
    module->RegisterModule("2", child_2);
    module->RegisterModule("1", child_1);

    auto modules = module->modules();
    ASSERT_EQ(modules.size(), 4);
    EXPECT_EQ(modules[0], module);
    EXPECT_EQ(modules[1], child_10);
    EXPECT_EQ(modules[2], child_2);
    EXPECT_EQ(modules[3], child_1);

    auto recursive_params = module->Parameters();
    ASSERT_EQ(recursive_params.size(), 5);
    EXPECT_EQ(recursive_params[0], replacement);
    EXPECT_EQ(recursive_params[1], second);
    EXPECT_EQ(recursive_params[2], child_10_param);
    EXPECT_EQ(recursive_params[3], child_2_param);
    EXPECT_EQ(recursive_params[4], child_1_param);
}

TEST_P(HookTest, ModuleBuffersMatchPersistentStateDictSemantics) {
    auto module = std::make_shared<TestModule>();
    auto persistent = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice());
    auto transient = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice());

    module->RegisterBuffer("persistent", persistent);
    module->RegisterBuffer("transient", transient, false);
    module->RegisterBuffer("deferred", nullptr);

    auto buffers = module->Buffers();
    ASSERT_EQ(buffers.size(), 2);
    EXPECT_EQ(buffers[0], persistent);
    EXPECT_EQ(buffers[1], transient);

    auto state = module->StateDict();
    EXPECT_TRUE(state.contains("persistent"));
    EXPECT_FALSE(state.contains("transient"));
    EXPECT_FALSE(state.contains("deferred"));

    module->RegisterBuffer("transient", transient, true);
    EXPECT_TRUE(module->StateDict().contains("transient"));
}

TEST_P(HookTest, ModuleRegistriesRejectInvalidAndConflictingNames) {
    auto module = std::make_shared<TestModule>();
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, GetDevice());

    EXPECT_DEATH(module->RegisterParameter("", tensor), "cannot be empty");
    EXPECT_DEATH(module->RegisterParameter("nested.weight", tensor), "cannot contain");

    module->RegisterParameter("weight", tensor);
    EXPECT_DEATH(module->RegisterBuffer("weight", tensor), "already used");
    EXPECT_DEATH(module->RegisterModule("weight", std::make_shared<TestModule>()), "already used");
}

INFINI_TRAIN_REGISTER_TEST(HookTest);
