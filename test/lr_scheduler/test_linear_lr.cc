#include "infini_train/include/lr_scheduler.h"
#include "test/lr_scheduler/test_helpers.h"

using namespace infini_train;
using namespace infini_train::lr_schedulers;

namespace {
constexpr float kBaseLR = 0.1f;
}

void TestFirstStepFromZero() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<LinearWarmupLR>(opt, /*warmup_steps=*/5, /*start_factor=*/0.0f);
    sched->Step();  // last_step_=1, alpha=1/5=0.2 → lr=0.02
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.02f);
}

void TestMidpointLR() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<LinearWarmupLR>(opt, 5, 0.0f);
    for (int i = 0; i < 3; ++i) sched->Step();
    // last_step_=2, alpha=3/5=0.6 → lr=0.1*0.6=0.06
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.06f);
}

void TestWarmupEnd() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<LinearWarmupLR>(opt, 5, 0.0f);
    for (int i = 0; i < 5; ++i) sched->Step();
    // last_step_=5, 5 >= 5 → base_lr
    ASSERT_FLOAT_EQ(sched->GetLR(), kBaseLR);
}

void TestBeyondWarmup() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<LinearWarmupLR>(opt, 5, 0.0f);
    for (int i = 0; i < 20; ++i) sched->Step();
    ASSERT_FLOAT_EQ(sched->GetLR(), kBaseLR);
}

void TestCustomStartFactor() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<LinearWarmupLR>(opt, 4, /*start_factor=*/0.25f);
    sched->Step();  // last_step_=1, alpha=1/4=0.25 → lr=0.1*(0.25+0.75*0.25)=0.04375
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.04375f, 1e-6f);
    sched->Step();  // last_step_=2, alpha=2/4=0.5 → lr=0.1*(0.25+0.75*0.5)=0.0625
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.0625f, 1e-6f);
}

void TestPyTorchAlignment() {
    const std::vector<float> expected = {
        0.02f, 0.04f, 0.06f, 0.08f, 0.1f, 0.1f, 0.1f};
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<LinearWarmupLR>(opt, 5, 0.0f);
    for (size_t i = 0; i < expected.size(); ++i) {
        sched->Step();
        ASSERT_FLOAT_NEAR(sched->GetLR(), expected[i], 1e-7f);
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

    std::cout << "========================" << std::endl;
    if (g_fail_count == 0) {
        std::cout << "All Tests PASSED" << std::endl;
    } else {
        std::cout << g_fail_count << " test(s) FAILED" << std::endl;
    }
    return g_fail_count > 0 ? 1 : 0;
}