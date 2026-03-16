#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/autograd/function.h"
#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/common/hook.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"
#include "test_utils.h"

using namespace infini_train;

class HookTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1);
    }
};

class TestModule : public nn::Module {
public:
    TestModule() : Module("TestModule") {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override {
        return inputs;
    }
};

TEST_F(HookTest, BasicModuleHooks) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32);
    x->set_requires_grad(true);

    auto module = std::make_shared<TestModule>();

    auto pre_hook = module->RegisterForwardPreHook(
        [](nn::Module *mod, const std::vector<std::shared_ptr<Tensor>> &inputs) {});

    auto fwd_hook = module->RegisterForwardPostHook(
        [](nn::Module *mod, const std::vector<std::shared_ptr<Tensor>> &inputs,
           const std::vector<std::shared_ptr<Tensor>> &outputs) {});

    auto bwd_pre_hook = module->RegisterBackwardPreHook(
        [](nn::Module *mod, const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {});

    auto bwd_post_hook = module->RegisterBackwardPostHook(
        [](nn::Module *mod, const std::vector<std::shared_ptr<Tensor>> &grad_inputs,
           const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {});

    std::vector<std::shared_ptr<Tensor>> inputs = {x};
    auto outputs = (*module)(inputs);

    EXPECT_EQ(outputs.size(), 1);
}

TEST_F(HookTest, HookRemove) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32);
    a->set_requires_grad(true);
    b->set_requires_grad(true);

    int hook1_count = 0;
    int hook2_count = 0;
    int hook3_count = 0;

    auto add_fn = std::make_shared<autograd::Add>();

    auto handle1 = add_fn->RegisterForwardPreHook(
        [&hook1_count](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &) {
            hook1_count++;
        });

    auto handle2 = add_fn->RegisterForwardPreHook(
        [&hook2_count](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &) {
            hook2_count++;
        });

    auto handle3 = add_fn->RegisterForwardPreHook(
        [&hook3_count](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &) {
            hook3_count++;
        });

    std::vector<std::shared_ptr<Tensor>> inputs = {a, b};

    add_fn->Apply(inputs);
    EXPECT_EQ(hook1_count, 1);
    EXPECT_EQ(hook2_count, 1);
    EXPECT_EQ(hook3_count, 1);

    handle2->Remove();

    add_fn->Apply(inputs);
    EXPECT_EQ(hook1_count, 2);
    EXPECT_EQ(hook2_count, 1);
    EXPECT_EQ(hook3_count, 2);

    handle1->Remove();

    add_fn->Apply(inputs);
    EXPECT_EQ(hook1_count, 2);
    EXPECT_EQ(hook2_count, 1);
    EXPECT_EQ(hook3_count, 3);
}

TEST_F(HookTest, BasicModuleHooksCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    x->set_requires_grad(true);

    auto module = std::make_shared<TestModule>();

    auto pre_hook = module->RegisterForwardPreHook(
        [](nn::Module *mod, const std::vector<std::shared_ptr<Tensor>> &inputs) {});

    std::vector<std::shared_ptr<Tensor>> inputs = {x};
    auto outputs = (*module)(inputs);

    EXPECT_EQ(outputs.size(), 1);
    EXPECT_TRUE(outputs[0]->IsCUDA());
#endif
}

TEST_F(HookTest, HookRemoveCUDA) {
    REQUIRE_CUDA();
#if defined(USE_CUDA)
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    a->set_requires_grad(true);
    b->set_requires_grad(true);

    int hook_count = 0;
    auto add_fn = std::make_shared<autograd::Add>();

    auto handle = add_fn->RegisterForwardPreHook(
        [&hook_count](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &) {
            hook_count++;
        });

    std::vector<std::shared_ptr<Tensor>> inputs = {a, b};
    add_fn->Apply(inputs);

    EXPECT_EQ(hook_count, 1);
#endif
}

TEST_F(HookTest, DistributedModuleHooks) {
    REQUIRE_DISTRIBUTED();
#if defined(USE_CUDA) && defined(USE_NCCL)
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    x->set_requires_grad(true);

    auto module = std::make_shared<TestModule>();

    auto pre_hook = module->RegisterForwardPreHook(
        [](nn::Module *mod, const std::vector<std::shared_ptr<Tensor>> &inputs) {});

    std::vector<std::shared_ptr<Tensor>> inputs = {x};
    auto outputs = (*module)(inputs);

    EXPECT_EQ(outputs.size(), 1);
    EXPECT_TRUE(outputs[0]->IsCUDA());
#endif
}

TEST_F(HookTest, DistributedHookRemove) {
    REQUIRE_DISTRIBUTED();
#if defined(USE_CUDA) && defined(USE_NCCL)
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    a->set_requires_grad(true);
    b->set_requires_grad(true);

    int hook_count = 0;
    auto add_fn = std::make_shared<autograd::Add>();

    auto handle = add_fn->RegisterForwardPreHook(
        [&hook_count](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &) {
            hook_count++;
        });

    std::vector<std::shared_ptr<Tensor>> inputs = {a, b};
    add_fn->Apply(inputs);

    EXPECT_EQ(hook_count, 1);
#endif
}
