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

TEST_P(HookTest, SavedTensorHooksPackAndUnpack) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice(), true);
    x->Fill(1.0f);

    int pack_count = 0;
    int unpack_count = 0;
    autograd::Function::SavedTensorHooks hooks{
        [&pack_count](const std::shared_ptr<Tensor> &tensor) {
            ++pack_count;
            return std::shared_ptr<void>(tensor);
        },
        [&unpack_count](const std::shared_ptr<void> &state) {
            ++unpack_count;
            return std::static_pointer_cast<Tensor>(state);
        },
    };

    auto sin_fn = std::make_shared<autograd::Sin>();
    {
        autograd::Function::SavedTensorHooksGuard guard(std::move(hooks));
        sin_fn->Apply({x});
    }

    EXPECT_EQ(pack_count, 1);
    EXPECT_EQ(unpack_count, 0);
    EXPECT_EQ(sin_fn->GetSavedTensor(0), x);
    EXPECT_EQ(unpack_count, 1);
}

TEST_P(HookTest, SavedTensorHooksCanBeDisabledInNestedScope) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice(), true);
    x->Fill(1.0f);

    int pack_count = 0;
    autograd::Function::SavedTensorHooks hooks{
        [&pack_count](const std::shared_ptr<Tensor> &tensor) {
            ++pack_count;
            return std::shared_ptr<void>(tensor);
        },
        [](const std::shared_ptr<void> &state) { return std::static_pointer_cast<Tensor>(state); },
    };

    autograd::Function::SavedTensorHooksGuard outer_guard(std::move(hooks));
    auto first_fn = std::make_shared<autograd::Sin>();
    first_fn->Apply({x});
    {
        autograd::Function::SavedTensorHooksGuard disable_hooks({nullptr, nullptr});
        auto disabled_fn = std::make_shared<autograd::Sin>();
        disabled_fn->Apply({x});
        EXPECT_EQ(disabled_fn->GetSavedTensor(0), x);
    }
    auto second_fn = std::make_shared<autograd::Sin>();
    second_fn->Apply({x});

    EXPECT_EQ(pack_count, 2);
}

TEST_P(HookTest, SavedTensorHooksSupportNullHookState) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice(), true);
    x->Fill(1.0f);

    int unpack_count = 0;
    autograd::Function::SavedTensorHooks hooks{
        [](const std::shared_ptr<Tensor> &) { return std::shared_ptr<void>{}; },
        [&x, &unpack_count](const std::shared_ptr<void> &state) {
            ++unpack_count;
            EXPECT_EQ(state, nullptr);
            return x;
        },
    };

    auto sin_fn = std::make_shared<autograd::Sin>();
    {
        autograd::Function::SavedTensorHooksGuard guard(std::move(hooks));
        sin_fn->Apply({x});
    }

    EXPECT_EQ(sin_fn->GetSavedTensor(0), x);
    EXPECT_EQ(unpack_count, 1);
}

INFINI_TRAIN_REGISTER_TEST(HookTest);
