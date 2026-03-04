#include "infini_train/include/lr_scheduler.h"
#include "test/lr_scheduler/test_helpers.h"

using namespace infini_train;
using namespace infini_train::lr_schedulers;

namespace {
constexpr float kBaseLR = 0.1f;
} // namespace 


void TestInitialState() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    ConstantLR sched(opt, /*factor=*/0.5f, /*total_iters=*/3);
    ASSERT_FLOAT_EQ(sched.GetLR(), 0.05f);
    ASSERT_TRUE(sched.LastStep() == 0);
    ASSERT_FLOAT_EQ(opt->GetLearningRate(), 0.05f);
}

void TestFirstStepAppliesFactor() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    ConstantLR sched(opt, 0.5f, 3);
    sched.Step();  // last_step_ = 0
    ASSERT_FLOAT_EQ(sched.GetLR(), 0.05f);
    ASSERT_FLOAT_EQ(opt->GetLearningRate(), 0.05f);
    ASSERT_TRUE(sched.LastStep() == 1);
}

void TestWithinTotalIters() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    ConstantLR sched(opt, 0.5f, 3);
    for (int i = 0; i < 2; ++i) sched.Step();
    // last_step_ = 2, still < 3
    ASSERT_FLOAT_EQ(sched.GetLR(), 0.05f);
}

void TestBeyondTotalIters() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    ConstantLR sched(opt, 0.5f, 3);
    for (int i = 0; i < 10; ++i) sched.Step();
    ASSERT_FLOAT_EQ(sched.GetLR(), kBaseLR);
    ASSERT_FLOAT_EQ(opt->GetLearningRate(), kBaseLR);
}

void TestPyTorchAlignment() {
    const std::vector<float> expected = {0.05f, 0.05f, 0.1f, 0.1f, 0.1f};
    auto opt = MakeDummyOptimizer(kBaseLR);
    ConstantLR sched(opt, 0.5f, 3);
    for (size_t i = 0; i < expected.size(); ++i) {
        sched.Step();
        ASSERT_FLOAT_EQ(sched.GetLR(), expected[i]);
    }
}

void TestStateRoundTrip() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    ConstantLR sched(opt, 0.5f, 5);
    for (int i = 0; i < 3; ++i) sched.Step();
    StateDict saved = sched.State();

    auto opt2 = MakeDummyOptimizer(kBaseLR);
    ConstantLR sched2(opt2, 0.5f, 5);
    sched2.LoadState(saved);

    ASSERT_TRUE(sched2.LastStep() == sched.LastStep());
    ASSERT_FLOAT_EQ(sched2.GetLR(), sched.GetLR());
    ASSERT_FLOAT_EQ(opt2->GetLearningRate(), sched.GetLR());
}

void TestResumeConsistency() {
    constexpr int kN = 8;
    constexpr int kK = 3;

    auto opt_ref = MakeDummyOptimizer(kBaseLR);
    ConstantLR sched_ref(opt_ref, 0.5f, 5);
    for (int i = 0; i < kN; ++i) sched_ref.Step();

    auto opt_a = MakeDummyOptimizer(kBaseLR);
    ConstantLR sched_a(opt_a, 0.5f, 5);
    for (int i = 0; i < kK; ++i) sched_a.Step();
    StateDict ckpt = sched_a.State();

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    ConstantLR sched_b(opt_b, 0.5f, 5);
    sched_b.LoadState(ckpt);
    for (int i = 0; i < kN - kK; ++i) sched_b.Step();

    ASSERT_FLOAT_EQ(sched_b.GetLR(), sched_ref.GetLR());
    ASSERT_TRUE(sched_b.LastStep() == sched_ref.LastStep());
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
    std::cout << "========================" << std::endl;
    if (g_fail_count == 0) {
        std::cout << "All Tests PASSED" << std::endl;
    } else {
        std::cout << g_fail_count << " test(s) FAILED" << std::endl;
    }
    return g_fail_count > 0 ? 1 : 0;
}