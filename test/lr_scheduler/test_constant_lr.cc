#include "infini_train/include/lr_scheduler.h"
#include "test/lr_scheduler/test_helpers.h"

using namespace infini_train;
using namespace infini_train::lr_schedulers;

namespace {
constexpr float kBaseLR = 0.1f;
} // namespace

void TestInitialState() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "constant",
        .constant_factor = 0.5f,
        .constant_total_iters = 3,
    };
    auto sched = CreateLRScheduler(opt, config);
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.05f);
    ASSERT_TRUE(sched->LastStep() == 0);
    ASSERT_FLOAT_EQ(opt->GetLearningRate(), 0.05f);
}

void TestFirstStepAppliesFactor() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "constant",
        .constant_factor = 0.5f,
        .constant_total_iters = 3,
    };

    auto sched = CreateLRScheduler(opt, config);
    sched->Step(); // last_step_ = 0
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.05f);
    ASSERT_FLOAT_EQ(opt->GetLearningRate(), 0.05f);
    ASSERT_TRUE(sched->LastStep() == 1);
}

void TestWithinTotalIters() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "constant",
        .constant_factor = 0.5f,
        .constant_total_iters = 3,
    };
    auto sched = CreateLRScheduler(opt, config);
    for (int i = 0; i < 2; ++i) { sched->Step(); }
    // last_step_ = 2, still < 3
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.05f);
}

void TestBeyondTotalIters() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "constant",
        .constant_factor = 0.5f,
        .constant_total_iters = 3,
    };
    auto sched = CreateLRScheduler(opt, config);
    for (int i = 0; i < 10; ++i) { sched->Step(); }
    ASSERT_FLOAT_EQ(sched->GetLR(), kBaseLR);
    ASSERT_FLOAT_EQ(opt->GetLearningRate(), kBaseLR);
}

void TestPyTorchAlignment() {
    const std::vector<float> expected = {0.05f, 0.05f, 0.1f, 0.1f, 0.1f};
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "constant",
        .constant_factor = 0.5f,
        .constant_total_iters = 3,
    };
    auto sched = CreateLRScheduler(opt, config);
    for (size_t i = 0; i < expected.size(); ++i) {
        sched->Step();
        ASSERT_FLOAT_EQ(sched->GetLR(), expected[i]);
    }
}

void TestStateRoundTrip() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config = {
        .type = "constant",
        .constant_factor = 0.5f,
        .constant_total_iters = 5,
    };
    auto sched = CreateLRScheduler(opt, config);
    for (int i = 0; i < 3; ++i) { sched->Step(); }
    StateDict saved = sched->State();

    auto opt2 = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config2 = {
        .type = "constant",
        .constant_factor = 0.5f,
        .constant_total_iters = 5,
    };
    auto sched2 = CreateLRScheduler(opt2, config2);
    sched2->LoadState(saved);

    ASSERT_TRUE(sched2->LastStep() == sched->LastStep());
    ASSERT_FLOAT_EQ(sched2->GetLR(), sched->GetLR());
    ASSERT_FLOAT_EQ(opt2->GetLearningRate(), sched->GetLR());
}

void TestResumeConsistency() {
    constexpr int kN = 8;
    constexpr int kK = 3;

    auto opt_ref = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config_ref = {
        .type = "constant",
        .constant_factor = 0.5f,
        .constant_total_iters = 5,
    };
    auto sched_ref = CreateLRScheduler(opt_ref, config_ref);
    for (int i = 0; i < kN; ++i) { sched_ref->Step(); }

    auto opt_a = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config_a = {
        .type = "constant",
        .constant_factor = 0.5f,
        .constant_total_iters = 5,
    };
    auto sched_a = CreateLRScheduler(opt_a, config_a);
    for (int i = 0; i < kK; ++i) { sched_a->Step(); }
    StateDict ckpt = sched_a->State();

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config_b = {
        .type = "constant",
        .constant_factor = 0.5f,
        .constant_total_iters = 5,
    };
    auto sched_b = CreateLRScheduler(opt_b, config_b);
    sched_b->LoadState(ckpt);
    for (int i = 0; i < kN - kK; ++i) { sched_b->Step(); }

    ASSERT_FLOAT_EQ(sched_b->GetLR(), sched_ref->GetLR());
    ASSERT_TRUE(sched_b->LastStep() == sched_ref->LastStep());
}

void TestChainableAndClosedFormConsistency() {
    auto opt_a = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config_a = {
        .type = "constant",
        .constant_factor = 0.5f,
        .constant_total_iters = 5,
    };
    auto chainable = CreateLRScheduler(opt_a, config_a);

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    LRSchedulerConfig config_b = {
        .type = "constant",
        .constant_factor = 0.5f,
        .constant_total_iters = 5,
    };
    auto closed_form = CreateLRScheduler(opt_b, config_b);

    for (int epoch = 1; epoch <= 12; ++epoch) {
        chainable->Step();
        closed_form->Step(epoch);
        ASSERT_FLOAT_NEAR(chainable->GetLR(), closed_form->GetLR(), 1e-7f);
    }
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    std::cout << "=== ConstantLR Tests ===" << std::endl;
    TestInitialState();
    TestFirstStepAppliesFactor();
    TestWithinTotalIters();
    TestBeyondTotalIters();
    TestPyTorchAlignment();
    TestStateRoundTrip();
    TestResumeConsistency();
    TestChainableAndClosedFormConsistency();
    std::cout << "========================" << std::endl;
    if (g_fail_count == 0) {
        std::cout << "All Tests PASSED" << std::endl;
    } else {
        std::cout << g_fail_count << " test(s) FAILED" << std::endl;
    }
    return g_fail_count > 0 ? 1 : 0;
}