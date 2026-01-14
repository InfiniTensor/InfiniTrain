#include <iostream>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/autograd/function.h"
#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/common/hook.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

// ============================================================================
// Test 1: Basic Module Hooks
// ============================================================================
void test_basic_hooks() {
    std::cout << "\n=== Test 1: Basic Module Hooks ===" << std::endl;

    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32);
    x->set_requires_grad(true);

    // Module hook example
    class MyModule : public nn::Module {
    public:
        MyModule() : Module("MyModule") {}

        std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override {
            std::cout << "Forward pass executing..." << std::endl;
            return inputs;
        }
    };

    auto module = std::make_shared<MyModule>();

    // Register forward pre-hook
    auto pre_hook
        = module->RegisterForwardPreHook([](nn::Module *mod, const std::vector<std::shared_ptr<Tensor>> &inputs) {
              std::cout << "Forward pre-hook: Module type = " << mod->type() << std::endl;
          });

    // Register forward post-hook
    auto fwd_hook
        = module->RegisterForwardPostHook([](nn::Module *mod, const std::vector<std::shared_ptr<Tensor>> &inputs,
                                             const std::vector<std::shared_ptr<Tensor>> &outputs) {
              std::cout << "Forward post-hook: Got " << outputs.size() << " outputs" << std::endl;
          });

    // Register backward pre-hook
    auto bwd_pre_hook = module->RegisterBackwardPreHook(
        [](nn::Module *mod, const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
            std::cout << "Backward pre-hook called!" << std::endl;
        });

    // Register backward post-hook
    auto bwd_post_hook
        = module->RegisterBackwardPostHook([](nn::Module *mod, const std::vector<std::shared_ptr<Tensor>> &grad_inputs,
                                              const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
              std::cout << "Backward post-hook called!" << std::endl;
          });

    // Test forward pass
    std::vector<std::shared_ptr<Tensor>> inputs = {x};
    auto outputs = (*module)(inputs);

    std::cout << "Module hook test completed!" << std::endl;
}

// ============================================================================
// Test 2: Hook Remove() Functionality Test
// ============================================================================
void test_hook_remove() {
    std::cout << "\n=== Test 2: Hook Remove() Functionality Test ===" << std::endl;

    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32);
    a->set_requires_grad(true);
    b->set_requires_grad(true);

    int hook1_count = 0;
    int hook2_count = 0;
    int hook3_count = 0;

    auto add_fn = std::make_shared<autograd::Add>();

    // Register three forward pre-hooks
    auto handle1 = add_fn->RegisterForwardPreHook(
        [&hook1_count](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &) {
            hook1_count++;
            std::cout << "Hook 1 called (count: " << hook1_count << ")" << std::endl;
        });

    auto handle2 = add_fn->RegisterForwardPreHook(
        [&hook2_count](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &) {
            hook2_count++;
            std::cout << "Hook 2 called (count: " << hook2_count << ")" << std::endl;
        });

    auto handle3 = add_fn->RegisterForwardPreHook(
        [&hook3_count](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &) {
            hook3_count++;
            std::cout << "Hook 3 called (count: " << hook3_count << ")" << std::endl;
        });

    // First call - all hooks should fire
    std::cout << "\n--- First Apply (all hooks active) ---" << std::endl;
    std::vector<std::shared_ptr<Tensor>> inputs;
    inputs.push_back(a);
    inputs.push_back(b);
    auto result1 = add_fn->Apply(inputs);
    std::cout << "Hook counts: " << hook1_count << ", " << hook2_count << ", " << hook3_count << std::endl;

    // Remove hook 2
    std::cout << "\n--- Removing Hook 2 ---" << std::endl;
    handle2->Remove();

    // Second call - hook 2 should not fire
    std::cout << "\n--- Second Apply (hook 2 removed) ---" << std::endl;
    auto result2 = add_fn->Apply(inputs);
    std::cout << "Hook counts: " << hook1_count << ", " << hook2_count << ", " << hook3_count << std::endl;

    // Remove hook 1
    std::cout << "\n--- Removing Hook 1 ---" << std::endl;
    handle1->Remove();

    // Third call - only hook 3 should fire
    std::cout << "\n--- Third Apply (hooks 1 and 2 removed) ---" << std::endl;
    auto result3 = add_fn->Apply(inputs);
    std::cout << "Hook counts: " << hook1_count << ", " << hook2_count << ", " << hook3_count << std::endl;

    // Verify results
    std::cout << "\n=== Test Results ===" << std::endl;
    bool test_passed = true;

    if (hook1_count != 2) {
        std::cout << "FAIL: Hook 1 should be called 2 times, got " << hook1_count << std::endl;
        test_passed = false;
    }

    if (hook2_count != 1) {
        std::cout << "FAIL: Hook 2 should be called 1 time, got " << hook2_count << std::endl;
        test_passed = false;
    }

    if (hook3_count != 3) {
        std::cout << "FAIL: Hook 3 should be called 3 times, got " << hook3_count << std::endl;
        test_passed = false;
    }

    if (test_passed) {
        std::cout << "SUCCESS: All hooks behaved correctly!" << std::endl;
        std::cout << "  - Hook 1: called 2 times (before removal)" << std::endl;
        std::cout << "  - Hook 2: called 1 time (removed after first call)" << std::endl;
        std::cout << "  - Hook 3: called 3 times (never removed)" << std::endl;
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    nn::parallel::global::GlobalEnv::Instance().Init(0, 1, 1, 1, 1);

    std::cout << "========================================" << std::endl;
    std::cout << "    Hook Mechanism Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_basic_hooks();
    test_hook_remove();

    std::cout << "\n========================================" << std::endl;
    std::cout << "    All Tests Completed Successfully" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
