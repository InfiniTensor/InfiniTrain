#include "infini_train/include/lr_scheduler.h"
#include "test/lr_scheduler/test_helpers.h"

using namespace infini_train;
using namespace infini_train::lr_schedulers;

namespace {
constexpr float kBaseLR = 0.1f;
}

void TestWithinFirstPeriod() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "step",
        .step_size = 3,
        .step_gamma = 0.1f,
    };
    auto sched = CreateLRScheduler(opt, config);
    for (int i = 0; i < 2; ++i) {
        sched->Step();
        ASSERT_FLOAT_EQ(sched->GetLR(), kBaseLR); // last_step 1,2 → 指数 0
    }
}

void TestFirstDecay() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "step",
        .step_size = 3,
        .step_gamma = 0.1f,
    };
    auto sched = CreateLRScheduler(opt, config);
    for (int i = 0; i < 3; ++i) { sched->Step(); }
    // last_step=3, 3//3=1 → 0.1^1 = 0.1 → lr=0.01
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.01f);
}

void TestMultipleDecays() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "step",
        .step_size = 3,
        .step_gamma = 0.1f,
    };
    auto sched = CreateLRScheduler(opt, config);
    for (int i = 0; i < 6; ++i) { sched->Step(); }
    // last_step=6, 6//3=2 → 0.1^2 = 0.01 → lr=0.001
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.001f, 1e-7f);
}

void TestPyTorchAlignment() {
    const std::vector<float> expected = {0.1f, 0.1f, 0.01f, 0.01f, 0.01f, 0.001f, 0.001f};
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "step",
        .step_size = 3,
        .step_gamma = 0.1f,
    };
    auto sched = CreateLRScheduler(opt, config);
    for (size_t i = 0; i < expected.size(); ++i) {
        sched->Step();
        ASSERT_FLOAT_NEAR(sched->GetLR(), expected[i], 1e-7f);
    }
}

void TestGammaOne() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "step",
        .step_size = 3,
        .step_gamma = 1.0f,
    };
    auto sched = CreateLRScheduler(opt, config);
    for (int i = 0; i < 20; ++i) {
        sched->Step();
        ASSERT_FLOAT_EQ(sched->GetLR(), kBaseLR);
    }
}

void TestChainableAndClosedFormConsistency() {
    auto opt_a = MakeDummyOptimizer(kBaseLR);
    auto chainable = CreateLRScheduler(opt_a, {
                                                  .type = "step",
                                                  .step_size = 3,
                                                  .step_gamma = 0.1f,
                                              });

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    auto closed_form = CreateLRScheduler(opt_b, {
                                                    .type = "step",
                                                    .step_size = 3,
                                                    .step_gamma = 0.1f,
                                                });

    for (int epoch = 1; epoch <= 12; ++epoch) {
        chainable->Step();
        closed_form->Step(epoch);
        ASSERT_FLOAT_NEAR(chainable->GetLR(), closed_form->GetLR(), 1e-7f);
    }
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    std::cout << "=== Step Tests ===" << std::endl;
    TestWithinFirstPeriod();
    TestFirstDecay();
    TestMultipleDecays();
    TestPyTorchAlignment();
    TestGammaOne();
    TestChainableAndClosedFormConsistency();

    std::cout << "========================" << std::endl;
    if (g_fail_count == 0) {
        std::cout << "All Tests PASSED" << std::endl;
    } else {
        std::cout << g_fail_count << " test(s) FAILED" << std::endl;
    }
    return g_fail_count > 0 ? 1 : 0;
}
