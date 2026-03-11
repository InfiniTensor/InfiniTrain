#include "infini_train/include/lr_scheduler.h"
#include "test/lr_scheduler/test_helpers.h"

using namespace infini_train;
using namespace infini_train::lr_schedulers;

namespace {
constexpr float kBaseLR = 0.1f;
}
// TC1: 单子调度器退化
void TestSingleScheduler() {
    std::cout << "[TC1] TestSingleScheduler" << std::endl;
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto step_lr = CreateLRScheduler(opt, {
                                              .type = "step",
                                              .step_size = 3,
                                              .step_gamma = 0.5f,
                                          });
    auto sched = LRScheduler::Create<ChainedScheduler>(opt, std::vector<std::shared_ptr<LRScheduler>>{step_lr});

    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.1f, kEps);
    sched->Step(); // step=1
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.1f, kEps);
}

// TC2: StepLR + LambdaLR 乘法叠加
void TestMultiplicativeChain() {
    std::cout << "[TC2] TestMultiplicativeChain" << std::endl;
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = CreateLRScheduler(
        opt, {
                 .type = "chained",
                 .chained_configs = {{
                                         .type = "step",
                                         .step_size = 2,
                                         .step_gamma = 0.5f,
                                     },
                                     {
                                         .type = "lambda",
                                         .lambda_fn = [](int64_t step) { return 1.0f - 0.1f * step; },
                                     }},
             });

    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.1f, kEps);

    sched->Step();
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.09f, kEps);

    sched->Step();
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.08f, kEps);

    sched->Step();
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.07f, kEps);
}

// TC3: ConstantLR + StepLR 叠加 (无穿插声明)
void TestConstantPlusStep() {
    std::cout << "[TC3] TestConstantPlusStep" << std::endl;
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = CreateLRScheduler(opt, {
                                            .type = "chained",
                                            .chained_configs = {{
                                                                    .type = "constant",
                                                                    .constant_factor = 0.5f,
                                                                    .constant_total_iters = 2,
                                                                },
                                                                {
                                                                    .type = "step",
                                                                    .step_size = 3,
                                                                    .step_gamma = 0.1f,
                                                                }},
                                        });

    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.05f, kEps);

    sched->Step();
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.05f, kEps);

    sched->Step();
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.1f, kEps);

    sched->Step();
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.01f, kEps);

    sched->Step();
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.01f, kEps);
}

// TC4: ConstantLR + StepLR 叠加（有穿插声明）
void TestConstantPlusStepDLC() {
    std::cout << "[TC4] TestConstantPlusStepDLC" << std::endl;
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto constant = CreateLRScheduler(opt, {
                                               .type = "constant",
                                               .constant_factor = 0.5f,
                                               .constant_total_iters = 2,
                                           });
    auto linear = CreateLRScheduler(opt, {
                                             .type = "linear",
                                             .linear_start_factor = 1e-8f,
                                             .linear_end_factor = 1.0f,
                                             .linear_total_iters = 3,
                                         });
    auto step_lr = CreateLRScheduler(opt, {
                                              .type = "step",
                                              .step_size = 3,
                                              .step_gamma = 0.1f,
                                          });
    auto Lambda = CreateLRScheduler(opt, {
                                             .type = "lambda",
                                             .lambda_fn = [](int64_t step) { return 1.0f - 0.1f * step; },
                                         });

    auto sched
        = LRScheduler::Create<ChainedScheduler>(opt, std::vector<std::shared_ptr<LRScheduler>>{constant, step_lr});

    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.1f, kEps);

    sched->Step();
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.1f, kEps);

    sched->Step();
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.2f, kEps);

    sched->Step();
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.02f, kEps);

    sched->Step();
    ASSERT_FLOAT_NEAR(sched->GetLR(), 0.02f, kEps);
}

// TC5: State/LoadState 往返
void TestStateRoundTrip() {
    std::cout << "[TC5] TestStateRoundTrip" << std::endl;
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto step_lr = std::make_shared<StepLR>(opt, 2, 0.5f);
    auto lambda_lr = std::make_shared<LambdaLR>(opt, [](int64_t step) { return 1.0f - 0.05f * step; });
    auto sched
        = LRScheduler::Create<ChainedScheduler>(opt, std::vector<std::shared_ptr<LRScheduler>>{step_lr, lambda_lr});

    for (int i = 0; i < 5; ++i) { sched->Step(); }
    StateDict saved = sched->State();

    auto opt2 = MakeDummyOptimizer(kBaseLR);
    auto step_lr2 = std::make_shared<StepLR>(opt2, 2, 0.5f);
    auto lambda_lr2 = std::make_shared<LambdaLR>(opt2, [](int64_t step) { return 1.0f - 0.05f * step; });
    auto sched2
        = LRScheduler::Create<ChainedScheduler>(opt2, std::vector<std::shared_ptr<LRScheduler>>{step_lr2, lambda_lr2});
    sched2->LoadState(saved);

    ASSERT_TRUE(sched2->LastStep() == sched->LastStep());
    ASSERT_FLOAT_NEAR(sched2->GetLR(), sched->GetLR(), kEps);
}

// TC6: resume 一致性
void TestResumeConsistency() {
    std::cout << "[TC6] TestResumeConsistency" << std::endl;
    constexpr int kN = 10, kK = 4;
    auto lambda_fn = [](int64_t step) { return 1.0f - 0.05f * step; };

    auto make_sched = [&](std::shared_ptr<Optimizer> opt) {
        auto step_lr = std::make_shared<StepLR>(opt, 2, 0.5f);
        auto lambda_lr = std::make_shared<LambdaLR>(opt, lambda_fn);
        return LRScheduler::Create<ChainedScheduler>(opt,
                                                     std::vector<std::shared_ptr<LRScheduler>>{step_lr, lambda_lr});
    };

    auto opt_ref = MakeDummyOptimizer(kBaseLR);
    auto sched_ref = make_sched(opt_ref);
    for (int i = 0; i < kN; ++i) { sched_ref->Step(); }

    auto opt_a = MakeDummyOptimizer(kBaseLR);
    auto sched_a = make_sched(opt_a);
    for (int i = 0; i < kK; ++i) { sched_a->Step(); }
    StateDict ckpt = sched_a->State();

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    auto sched_b = make_sched(opt_b);
    sched_b->LoadState(ckpt);
    for (int i = 0; i < kN - kK; ++i) { sched_b->Step(); }

    ASSERT_FLOAT_NEAR(sched_b->GetLR(), sched_ref->GetLR(), kEps);
    ASSERT_TRUE(sched_b->LastStep() == sched_ref->LastStep());
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    std::cout << "=== ChainedScheduler Tests ===" << std::endl;
    TestSingleScheduler();
    TestMultiplicativeChain();
    TestConstantPlusStep();
    TestConstantPlusStepDLC();
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