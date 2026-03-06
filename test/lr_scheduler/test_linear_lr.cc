#include "infini_train/include/lr_scheduler.h"
#include "test/lr_scheduler/test_helpers.h"

using namespace infini_train;
using namespace infini_train::lr_schedulers;

namespace {
constexpr float kBaseLR = 0.1f;
}

void TestFirstStepFromZero() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<LinearLR>(
        opt, /*start_factor=*/0.2f, /*end_factor=*/1.0f, /*total_iters=*/5);
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.02f);
}

void TestMidpointLR() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<LinearLR>(opt, 0.2f, 1.0f, 5);
    for (int i = 0; i < 3; ++i) sched->Step();
    // last_step_=3 -> 0.1*(0.2 + 0.8*3/5) = 0.068
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.068f);
}

void TestWarmupEnd() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<LinearLR>(opt, 0.2f, 1.0f, 5);
    for (int i = 0; i < 5; ++i) sched->Step();
    // last_step_ >= total_iters -> base_lr * end_factor
    ASSERT_FLOAT_EQ(sched->GetLR(), kBaseLR);
}

void TestBeyondWarmup() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<LinearLR>(opt, 0.2f, 1.0f, 5);
    for (int i = 0; i < 20; ++i) sched->Step();
    ASSERT_FLOAT_EQ(sched->GetLR(), kBaseLR);
}

void TestCustomStartFactor() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<LinearLR>(
        opt, /*start_factor=*/0.25f, /*end_factor=*/1.0f, /*total_iters=*/4);
    sched->Step();  // last_step_=1, lr=0.1*(0.25+0.75*1/4)=0.04375
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.04375f, 1e-6f);
    sched->Step();  // last_step_=2, lr=0.1*(0.25+0.75*2/4)=0.0625
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.0625f, 1e-6f);
}

void TestPyTorchAlignment() {
    const std::vector<float> expected = {
        0.036f, 0.052f, 0.068f, 0.084f, 0.1f, 0.1f, 0.1f};
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<LinearLR>(opt, 0.2f, 1.0f, 5);
    for (size_t i = 0; i < expected.size(); ++i) {
        sched->Step();
        ASSERT_FLOAT_NEAR(sched->GetLR(), expected[i], 1e-7f);
    }
}

void TestChainableAndClosedFormConsistency() {
    auto opt_a = MakeDummyOptimizer(kBaseLR);
    auto chainable = LRScheduler::Create<LinearLR>(opt_a, 0.2f, 1.0f, 5);

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    auto closed_form = LRScheduler::Create<LinearLR>(opt_b, 0.2f, 1.0f, 5);

    for (int epoch = 1; epoch <= 10; ++epoch) {
        chainable->Step();
        closed_form->Step(epoch);
        ASSERT_FLOAT_NEAR(chainable->GetLR(), closed_form->GetLR(), 1e-6f);
    }
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    std::cout << "=== Linear Tests ===" << std::endl;
    TestFirstStepFromZero();
    TestMidpointLR();
    TestWarmupEnd();
    TestBeyondWarmup();
    TestCustomStartFactor();
    TestPyTorchAlignment();
    TestChainableAndClosedFormConsistency();

    std::cout << "========================" << std::endl;
    if (g_fail_count == 0) {
        std::cout << "All Tests PASSED" << std::endl;
    } else {
        std::cout << g_fail_count << " test(s) FAILED" << std::endl;
    }
    return g_fail_count > 0 ? 1 : 0;
}