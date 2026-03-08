#include "infini_train/include/lr_scheduler.h"
#include "test/lr_scheduler/test_helpers.h"

using namespace infini_train;
using namespace infini_train::lr_schedulers;

namespace {
constexpr float kBaseLR = 0.1f;
} // namespace 

void TestIdentityLambda() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = CreateLRScheduler(opt, {
        .type = "lambda",
        .lambda_fn = [](int64_t) { return 1.0f; },
    });
    // 构造器内 Step() → last_step_=0, lr = 0.1 * 1.0 = 0.1
    ASSERT_TRUE(sched->LastStep() == 0);
    ASSERT_FLOAT_NEAR(sched->GetLR(), kBaseLR, kEps);
    ASSERT_FLOAT_NEAR(opt->GetLearningRate(), kBaseLR, kEps);
}

void TestLinearDecayLambda() {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = CreateLRScheduler(opt, {
        .type = "lambda",
        .lambda_fn = [](int64_t step) { return 1.0f - step * 0.1f; },
    });
    // step=0, lambda(0)=1.0, lr=0.1
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.1f, kEps);

    sched->Step();  // step=1, lambda(1)=0.9, lr=0.09
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.09f, kEps);

    sched->Step();  // step=2, lambda(2)=0.8, lr=0.08
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.08f, kEps);

    sched->Step();  // step=3, lambda(3)=0.7, lr=0.07
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.07f, kEps);
}

void TestPyTorchAlignment() {
    // PyTorch: LambdaLR(opt, lr_lambda=lambda epoch: 0.95**epoch)
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = CreateLRScheduler(opt, {
        .type = "lambda",
        .lambda_fn = [](int64_t step) { return static_cast<float>(std::pow(0.95, step)); },
    });
    // step=0, lr = 0.1 * 0.95^0 = 0.1
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.1f, kEps);

    std::vector<float> expected = {0.095f, 0.09025f, 0.0857375f};
    for (size_t i = 0; i < expected.size(); ++i) {
        sched->Step();
        ASSERT_FLOAT_NEAR(sched->GetLR(), expected[i], 1e-5f);
    }
}

void TestStateRoundTrip() {
    auto lambda_fn = [](int64_t step) { return 1.0f - step * 0.05f; };
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = CreateLRScheduler(opt, {
        .type = "lambda",
        .lambda_fn = lambda_fn,
    });
    for (int i = 0; i < 5; ++i) sched->Step();
    StateDict saved = sched->State();

    auto opt2 = MakeDummyOptimizer(kBaseLR);
    auto sched2 = CreateLRScheduler(opt2, {
        .type = "lambda",
        .lambda_fn = lambda_fn,
    });  // same lambda
    sched2->LoadState(saved);

    ASSERT_TRUE(sched2->LastStep() == sched->LastStep());
    ASSERT_FLOAT_NEAR(sched2->GetLR(), sched->GetLR(), kEps);
}

void TestResumeConsistency() {
    auto lambda_fn = [](int64_t step) { return 1.0f - step * 0.05f; };
    constexpr int kN = 10, kK = 4;

    auto opt_ref = MakeDummyOptimizer(kBaseLR);
    auto sched_ref = CreateLRScheduler(opt_ref, {
        .type = "lambda",
        .lambda_fn = lambda_fn,
    });
    for (int i = 0; i < kN; ++i) sched_ref->Step();

    auto opt_a = MakeDummyOptimizer(kBaseLR);
    auto sched_a = CreateLRScheduler(opt_a, {
        .type = "lambda",
        .lambda_fn = lambda_fn,
    });
    for (int i = 0; i < kK; ++i) sched_a->Step();
    StateDict ckpt = sched_a->State();

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    auto sched_b = CreateLRScheduler(opt_b, {
        .type = "lambda",
        .lambda_fn = lambda_fn,
    });
    sched_b->LoadState(ckpt);
    for (int i = 0; i < kN - kK; ++i) sched_b->Step();

    ASSERT_FLOAT_NEAR(sched_b->GetLR(), sched_ref->GetLR(), kEps);
    ASSERT_TRUE(sched_b->LastStep() == sched_ref->LastStep());
}


int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    std::cout << "=== LambdaLR Tests ===" << std::endl;
    TestIdentityLambda();
    TestLinearDecayLambda();
    TestPyTorchAlignment();
    TestStateRoundTrip();
    TestResumeConsistency();
    std::cout << "======================" << std::endl;
    if (g_fail_count == 0) {
        std::cout << "All Tests PASSED" << std::endl;
    } else {
        std::cout << g_fail_count << " test(s) FAILED" << std::endl;
    }
    return g_fail_count > 0 ? 1 : 0;
}