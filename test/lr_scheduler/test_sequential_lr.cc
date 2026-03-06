#include "infini_train/include/lr_scheduler.h"
#include "test/lr_scheduler/test_helpers.h"

using namespace infini_train;
using namespace infini_train::lr_schedulers;
namespace {
constexpr float kBaseLR = 0.1f;
} // namespace

void TestLinearThenConstant() {
    std::cout << "[TC1] TestLinearThenConstant" << std::endl;
    auto opt = MakeDummyOptimizer(kBaseLR);

    auto linear = LRScheduler::Create<LinearLR>(opt, /*start_factor=*/1e-8, /*end_factor=*/1.0f, /*total_iters=*/3);
    auto constant = LRScheduler::Create<ConstantLR>(opt, /*factor=*/1.0f, /*total_iters=*/100);
    auto sched = LRScheduler::Create<SequentialLR>(opt, std::vector<std::shared_ptr<LRScheduler>>{linear, constant}, std::vector<int64_t>{3});

    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.0f, kEps);

    sched->Step();  // global=1, warmup step=1, lr=0.1*(1/3)
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.1f / 3.0f, 1e-5f);

    sched->Step();  // global=2, warmup step=2, lr=0.1*(2/3)
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.2f / 3.0f, 1e-5f);

    sched->Step();  // global=3, constant step=0, lr=0.1*1.0=0.1
    ASSERT_FLOAT_NEAR(sched->GetLR(), kBaseLR, kEps);

    sched->Step();  // global=4, constant step=1, lr=0.1
    ASSERT_FLOAT_NEAR(sched->GetLR(), kBaseLR, kEps);
}

void TestLinearThenStepLR() {
    std::cout << "[TC2] TestLinearThenStepLR" << std::endl;
    auto opt = MakeDummyOptimizer(kBaseLR);

    auto linear = LRScheduler::Create<LinearLR>(opt, /*start_factor=*/1e-8f, /*end_factor=*/1.0f, /*total_iters=*/3);
    auto step_lr = LRScheduler::Create<StepLR>(opt, /*step_size=*/3, /*gamma=*/0.5f);

    auto sched = LRScheduler::Create<SequentialLR>(opt, std::vector<std::shared_ptr<LRScheduler>>{linear, step_lr}, std::vector<int64_t>{3});

    sched->Step();  // global=1
    sched->Step();  // global=2

    sched->Step();  // global=3, StepLR step=0, lr=0.1
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.1f, kEps);

    sched->Step();  // global=4, StepLR step=1
    sched->Step();  // global=5, StepLR step=2
    sched->Step();  // global=6, StepLR step=3, 3//3=1, lr=0.1*0.5=0.05
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.05f, kEps);
}

void TestLinearThenStepThenConstant(){
    std::cout << "[TC3] TestLinearThenStepThenConstant" << std::endl;
    auto opt = MakeDummyOptimizer(kBaseLR);

    auto linear = LRScheduler::Create<LinearLR>(opt, /*start_factor=*/1e-8f, /*end_factor=*/1.0f, /*total_iters=*/3);
    auto step_lr = LRScheduler::Create<StepLR>(opt, /*step_size=*/3, /*gamma=*/0.5f);
    auto constant = LRScheduler::Create<ConstantLR>(opt, /*factor=*/0.5f, /*total_iters=*/2);

    auto sched = LRScheduler::Create<SequentialLR>(opt, std::vector<std::shared_ptr<LRScheduler>>{linear, step_lr, constant}, std::vector<int64_t>{3, 6});
    const std::vector<float> expected = {
        0.033333f, 0.066667f, 0.1f, 0.1f, 0.1f, 0.05f, 0.05f, 0.1f, 0.1f, 0.1f};
    for (size_t i = 0; i < expected.size(); ++i) {
        sched->Step();
        ASSERT_FLOAT_NEAR(sched->GetLR(), expected[i], 1e-5f);
    }
}

void TestStateRoundTrip() {
    std::cout << "[TC4] TestStateRoundTrip" << std::endl;
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto linear = LRScheduler::Create<LinearLR>(opt, /*start_factor=*/1e-8f, /*end_factor=*/1.0f, /*total_iters=*/3);
    auto step_lr = LRScheduler::Create<StepLR>(opt, /*step_size=*/3, /*gamma=*/0.5f);
    auto sched = LRScheduler::Create<SequentialLR>(opt, std::vector<std::shared_ptr<LRScheduler>>{linear, step_lr}, std::vector<int64_t>{3});

    for (int i = 0; i < 5; ++i) sched->Step();
    StateDict saved = sched->State();

    auto opt2 = MakeDummyOptimizer(kBaseLR);
    auto linear2 = LRScheduler::Create<LinearLR>(opt2, /*start_factor=*/1e-8f, /*end_factor=*/1.0f, /*total_iters=*/3);
    auto step_lr2 = LRScheduler::Create<StepLR>(opt2, /*step_size=*/3, /*gamma=*/0.5f);
    auto sched2 = LRScheduler::Create<SequentialLR>(opt2, std::vector<std::shared_ptr<LRScheduler>>{linear2, step_lr2}, std::vector<int64_t>{3});
    sched2->LoadState(saved);

    ASSERT_TRUE(sched2->LastStep() == sched->LastStep());
    ASSERT_FLOAT_NEAR(sched2->GetLR(), sched->GetLR(), kEps);
}

void TestResumeConsistency() {
    std::cout << "[TC5] TestResumeConsistency" << std::endl;
    constexpr int kN = 10, kK = 4;

    auto make_sched = [](std::shared_ptr<Optimizer> opt) {
        auto linear = LRScheduler::Create<LinearLR>(opt, /*start_factor=*/1e-8f, /*end_factor=*/1.0f, /*total_iters=*/3);
        auto step_lr = LRScheduler::Create<StepLR>(opt, /*step_size=*/3, /*gamma=*/0.5f);
        return LRScheduler::Create<SequentialLR>(opt,
            std::vector<std::shared_ptr<LRScheduler>>{linear, step_lr},
            std::vector<int64_t>{3});
    };

    auto opt_ref = MakeDummyOptimizer(kBaseLR);
    auto sched_ref = make_sched(opt_ref);
    for (int i = 0; i < kN; ++i) sched_ref->Step();

    auto opt_a = MakeDummyOptimizer(kBaseLR);
    auto sched_a = make_sched(opt_a);
    for (int i = 0; i < kK; ++i) sched_a->Step();
    StateDict ckpt = sched_a->State();

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    auto sched_b = make_sched(opt_b);
    sched_b->LoadState(ckpt);
    for (int i = 0; i < kN - kK; ++i) sched_b->Step();

    ASSERT_FLOAT_NEAR(sched_b->GetLR(), sched_ref->GetLR(), kEps);
    ASSERT_TRUE(sched_b->LastStep() == sched_ref->LastStep());
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    std::cout << "=== SequentialLR Tests ===" << std::endl;
    TestLinearThenConstant();
    TestLinearThenStepLR();
    TestLinearThenStepThenConstant();
    TestStateRoundTrip();
    TestResumeConsistency();
    if (g_fail_count == 0) {
        std::cout << "All Tests PASSED" << std::endl;
    } else {
        std::cout << g_fail_count << " test(s) FAILED" << std::endl;
    }
    return g_fail_count > 0 ? 1 : 0;
}