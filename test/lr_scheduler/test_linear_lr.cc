#include "infini_train/include/lr_scheduler.h"
#include "test/lr_scheduler/test_helpers.h"

using namespace infini_train;
using namespace infini_train::lr_schedulers;

namespace {
constexpr float kBaseLR = 0.1f;
}

void TestFirstStepFromZero() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "linear",
        .linear_start_factor = 0.2f,
        .linear_end_factor = 1.0f,
        .linear_total_iters = 5,
    };
    
    auto sched = CreateLRScheduler(opt, config);
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.02f);
}

void TestMidpointLR() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "linear",
        .linear_start_factor = 0.2f,
        .linear_end_factor = 1.0f,
        .linear_total_iters = 5,
    };
    auto sched = CreateLRScheduler(opt, config);
    for (int i = 0; i < 3; ++i) sched->Step();
    // last_step_=3 -> 0.1*(0.2 + 0.8*3/5) = 0.068
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.068f);
}

void TestWarmupEnd() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "linear",
        .linear_start_factor = 0.2f,
        .linear_end_factor = 1.0f,
        .linear_total_iters = 5,
    };
    auto sched = CreateLRScheduler(opt, config);
    for (int i = 0; i < 5; ++i) sched->Step();
    // last_step_ >= total_iters -> base_lr * end_factor
    ASSERT_FLOAT_EQ(sched->GetLR(), kBaseLR);
}

void TestBeyondWarmup() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "linear",
        .linear_start_factor = 0.2f,
        .linear_end_factor = 1.0f,
        .linear_total_iters = 5,
    };
    auto sched = CreateLRScheduler(opt, config);
    for (int i = 0; i < 20; ++i) sched->Step();
    ASSERT_FLOAT_EQ(sched->GetLR(), kBaseLR);
}

void TestCustomStartFactor() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "linear",
        .linear_start_factor = 0.25f,
        .linear_end_factor = 1.0f,
        .linear_total_iters = 4,
    };
    auto sched = CreateLRScheduler(opt, config);
    sched->Step();  // last_step_=1, lr=0.1*(0.25+0.75*1/4)=0.04375
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.04375f, 1e-6f);
    sched->Step();  // last_step_=2, lr=0.1*(0.25+0.75*2/4)=0.0625
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.0625f, 1e-6f);
}

void TestPyTorchAlignment() {
    const std::vector<float> expected = {
        0.036f, 0.052f, 0.068f, 0.084f, 0.1f, 0.1f, 0.1f};
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "linear",
        .linear_start_factor = 0.2f,
        .linear_end_factor = 1.0f,
        .linear_total_iters = 5,
    };
    auto sched = CreateLRScheduler(opt, config);
    for (size_t i = 0; i < expected.size(); ++i) {
        sched->Step();
        ASSERT_FLOAT_NEAR(sched->GetLR(), expected[i], 1e-7f);
    }
}

void TestChainableAndClosedFormConsistency() {
    auto opt_a = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config_a = {
        .type = "linear",
        .linear_start_factor = 0.2f,
        .linear_end_factor = 1.0f,
        .linear_total_iters = 5,
    };
    auto chainable = CreateLRScheduler(opt_a, config_a);

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config_b = {
        .type = "linear",
        .linear_start_factor = 0.2f,
        .linear_end_factor = 1.0f,
        .linear_total_iters = 5,
    };
    auto closed_form = CreateLRScheduler(opt_b, config_b);

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