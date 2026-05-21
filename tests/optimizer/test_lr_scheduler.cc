#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/lr_scheduler.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;
using namespace infini_train::lr_schedulers;

namespace {

constexpr float kBaseLR = 0.1f;
constexpr float kEps = 1e-6f;

class LRSchedulerTest : public infini_train::test::InfiniTrainTest {};

class LinearDecayScheduler : public LRScheduler {
public:
    LinearDecayScheduler(std::shared_ptr<Optimizer> optimizer, int64_t total_steps, int64_t last_step = -1)
        : LRScheduler(std::move(optimizer), last_step), total_steps_(total_steps) {}

protected:
    float GetClosedFormLR() const override {
        if (last_step_ >= total_steps_) {
            return 0.0f;
        }
        return base_lr_ * (1.0f - static_cast<float>(last_step_) / static_cast<float>(total_steps_));
    }

private:
    int64_t total_steps_;
};

std::shared_ptr<Optimizer> MakeDummyOptimizer(float lr) {
    std::vector<std::shared_ptr<Tensor>> empty_params;
    return std::make_shared<optimizers::SGD>(empty_params, lr);
}

void ExpectStepSequence(const std::shared_ptr<LRScheduler> &scheduler, std::initializer_list<float> expected,
                        float eps = kEps) {
    for (float expected_lr : expected) {
        scheduler->Step();
        EXPECT_NEAR(scheduler->GetLR(), expected_lr, eps);
    }
}

std::shared_ptr<LRScheduler> MakeSequentialScheduler(std::shared_ptr<Optimizer> opt) {
    auto linear = LRScheduler::Create<LinearLR>(opt, /*start_factor=*/1e-8f, /*end_factor=*/1.0f,
                                                /*total_iters=*/3);
    auto step_lr = LRScheduler::Create<StepLR>(opt, /*step_size=*/3, /*gamma=*/0.5f);
    return LRScheduler::Create<SequentialLR>(opt,
                                             /*schedulers=*/std::vector<std::shared_ptr<LRScheduler>>{linear, step_lr},
                                             /*milestones=*/std::vector<int64_t>{3});
}

std::shared_ptr<LRScheduler> MakeChainedScheduler(std::shared_ptr<Optimizer> opt) {
    auto step_lr = LRScheduler::Create<StepLR>(opt, /*step_size=*/2, /*gamma=*/0.5f);
    auto lambda_lr = LRScheduler::Create<LambdaLR>(opt, /*lr_lambda=*/[](int64_t step) { return 1.0f - 0.05f * step; });
    return LRScheduler::Create<ChainedScheduler>(
        opt, /*schedulers=*/std::vector<std::shared_ptr<LRScheduler>>{step_lr, lambda_lr});
}

} // namespace

TEST_P(LRSchedulerTest, BaseSchedulerStateRoundTripAndResume) {
    constexpr int64_t kTotalSteps = 20;

    auto opt_ref = MakeDummyOptimizer(kBaseLR);
    auto sched_ref = LRScheduler::Create<LinearDecayScheduler>(opt_ref, /*total_steps=*/kTotalSteps);
    ExpectStepSequence(sched_ref, {0.095f, 0.09f, 0.085f, 0.08f, 0.075f, 0.07f, 0.065f});

    auto opt_a = MakeDummyOptimizer(kBaseLR);
    auto sched_a = LRScheduler::Create<LinearDecayScheduler>(opt_a, /*total_steps=*/kTotalSteps);
    ExpectStepSequence(sched_a, {0.095f, 0.09f, 0.085f});

    StateDict state = sched_a->State();
    EXPECT_EQ(state.count("last_step"), 1);
    EXPECT_EQ(state.count("recover_lr"), 1);
    EXPECT_EQ(state.count("base_lr"), 1);

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    auto sched_b = LRScheduler::Create<LinearDecayScheduler>(opt_b, /*total_steps=*/kTotalSteps);
    sched_b->LoadState(state);
    ExpectStepSequence(sched_b, {0.08f, 0.075f, 0.07f, 0.065f});

    EXPECT_EQ(sched_b->LastStep(), sched_ref->LastStep());
    EXPECT_NEAR(sched_b->GetLR(), sched_ref->GetLR(), kEps);
    EXPECT_NEAR(opt_b->learning_rate(), sched_ref->GetLR(), kEps);
}

TEST_P(LRSchedulerTest, ConstantLRMatchesExpectedSchedule) {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<ConstantLR>(opt, /*factor=*/0.5f, /*total_iters=*/3);

    EXPECT_EQ(sched->LastStep(), 0);
    EXPECT_NEAR(sched->GetLR(), 0.05f, kEps);
    EXPECT_NEAR(opt->learning_rate(), 0.05f, kEps);

    ExpectStepSequence(sched, {0.05f, 0.05f, 0.1f, 0.1f, 0.1f});
}

TEST_P(LRSchedulerTest, LinearLRMatchesExpectedScheduleAndClosedForm) {
    auto opt_a = MakeDummyOptimizer(kBaseLR);
    auto chainable = LRScheduler::Create<LinearLR>(opt_a, /*start_factor=*/0.2f, /*end_factor=*/1.0f,
                                                   /*total_iters=*/5);

    EXPECT_NEAR(chainable->GetLR(), 0.02f, kEps);
    ExpectStepSequence(chainable, {0.036f, 0.052f, 0.068f, 0.084f, 0.1f, 0.1f, 0.1f}, 1e-7f);

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    auto closed_form = LRScheduler::Create<LinearLR>(opt_b, /*start_factor=*/0.2f, /*end_factor=*/1.0f,
                                                     /*total_iters=*/5);
    auto opt_c = MakeDummyOptimizer(kBaseLR);
    auto chainable_again = LRScheduler::Create<LinearLR>(opt_c, /*start_factor=*/0.2f, /*end_factor=*/1.0f,
                                                         /*total_iters=*/5);

    for (int epoch = 1; epoch <= 10; ++epoch) {
        chainable_again->Step();
        closed_form->Step(epoch);
        EXPECT_NEAR(chainable_again->GetLR(), closed_form->GetLR(), kEps);
    }
}

TEST_P(LRSchedulerTest, StepLRMatchesExpectedScheduleAndClosedForm) {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<StepLR>(opt, /*step_size=*/3, /*gamma=*/0.1f);

    EXPECT_NEAR(sched->GetLR(), kBaseLR, kEps);
    ExpectStepSequence(sched, {0.1f, 0.1f, 0.01f, 0.01f, 0.01f, 0.001f, 0.001f}, 1e-7f);

    auto opt_a = MakeDummyOptimizer(kBaseLR);
    auto chainable = LRScheduler::Create<StepLR>(opt_a, /*step_size=*/3, /*gamma=*/0.1f);
    auto opt_b = MakeDummyOptimizer(kBaseLR);
    auto closed_form = LRScheduler::Create<StepLR>(opt_b, /*step_size=*/3, /*gamma=*/0.1f);

    for (int epoch = 1; epoch <= 12; ++epoch) {
        chainable->Step();
        closed_form->Step(epoch);
        EXPECT_NEAR(chainable->GetLR(), closed_form->GetLR(), 1e-7f);
    }
}

TEST_P(LRSchedulerTest, LambdaLRMatchesExpectedScheduleAndRestoresState) {
    auto lambda_fn = [](int64_t step) { return static_cast<float>(std::pow(0.95, step)); };

    auto opt_ref = MakeDummyOptimizer(kBaseLR);
    auto sched_ref = LRScheduler::Create<LambdaLR>(opt_ref, /*lr_lambda=*/lambda_fn);
    ExpectStepSequence(sched_ref, {0.095f, 0.09025f, 0.0857375f, 0.08145062f}, 1e-6f);

    auto opt_a = MakeDummyOptimizer(kBaseLR);
    auto sched_a = LRScheduler::Create<LambdaLR>(opt_a, /*lr_lambda=*/lambda_fn);
    ExpectStepSequence(sched_a, {0.095f, 0.09025f}, 1e-6f);
    StateDict state = sched_a->State();

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    auto sched_b = LRScheduler::Create<LambdaLR>(opt_b, /*lr_lambda=*/lambda_fn);
    sched_b->LoadState(state);
    ExpectStepSequence(sched_b, {0.0857375f, 0.08145062f}, 1e-6f);

    EXPECT_EQ(sched_b->LastStep(), sched_ref->LastStep());
    EXPECT_NEAR(sched_b->GetLR(), sched_ref->GetLR(), 1e-6f);
}

TEST_P(LRSchedulerTest, SequentialLRSwitchesAtMilestonesAndRestoresState) {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = MakeSequentialScheduler(opt);

    EXPECT_NEAR(sched->GetLR(), 0.0f, kEps);
    ExpectStepSequence(sched, {0.1f / 3.0f, 0.2f / 3.0f, 0.1f, 0.1f, 0.1f, 0.05f}, 1e-5f);

    auto opt_ref = MakeDummyOptimizer(kBaseLR);
    auto sched_ref = MakeSequentialScheduler(opt_ref);
    ExpectStepSequence(sched_ref, {0.1f / 3.0f, 0.2f / 3.0f, 0.1f, 0.1f, 0.1f, 0.05f, 0.05f, 0.05f}, 1e-5f);

    auto opt_a = MakeDummyOptimizer(kBaseLR);
    auto sched_a = MakeSequentialScheduler(opt_a);
    ExpectStepSequence(sched_a, {0.1f / 3.0f, 0.2f / 3.0f, 0.1f, 0.1f}, 1e-5f);
    StateDict state = sched_a->State();

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    auto sched_b = MakeSequentialScheduler(opt_b);
    sched_b->LoadState(state);
    ExpectStepSequence(sched_b, {0.1f, 0.05f, 0.05f, 0.05f}, 1e-5f);

    EXPECT_EQ(sched_b->LastStep(), sched_ref->LastStep());
    EXPECT_NEAR(sched_b->GetLR(), sched_ref->GetLR(), kEps);
}

TEST_P(LRSchedulerTest, ChainedSchedulerComposesChildrenAndRestoresState) {
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = MakeChainedScheduler(opt);

    EXPECT_NEAR(sched->GetLR(), 0.1f, kEps);
    ExpectStepSequence(sched, {0.095f, 0.09f, 0.085f, 0.08f}, kEps);

    auto opt_ref = MakeDummyOptimizer(kBaseLR);
    auto sched_ref = MakeChainedScheduler(opt_ref);
    ExpectStepSequence(sched_ref, {0.095f, 0.09f, 0.085f, 0.08f, 0.075f, 0.07f}, kEps);

    auto opt_a = MakeDummyOptimizer(kBaseLR);
    auto sched_a = MakeChainedScheduler(opt_a);
    ExpectStepSequence(sched_a, {0.095f, 0.09f, 0.085f}, kEps);
    StateDict state = sched_a->State();

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    auto sched_b = MakeChainedScheduler(opt_b);
    sched_b->LoadState(state);
    ExpectStepSequence(sched_b, {0.08f, 0.075f, 0.07f}, kEps);

    EXPECT_EQ(sched_b->LastStep(), sched_ref->LastStep());
    EXPECT_NEAR(sched_b->GetLR(), sched_ref->GetLR(), kEps);
}

TEST_P(LRSchedulerTest, TrainingSchedulerFactoryBuildsCommonDecayStyles) {
    {
        auto opt = MakeDummyOptimizer(0.1f);
        auto sched = CreateLRScheduler(opt, {
                                                .lr_decay_style = "constant",
                                                .lr = 0.1f,
                                                .min_lr = 0.0f,
                                                .lr_decay_iters = 10,
                                                .lr_warmup_iters = 0,
                                                .lr_warmup_init = 0.0f,
                                            });
        EXPECT_NEAR(sched->GetLR(), 0.1f, kEps);
        ExpectStepSequence(sched, {0.1f, 0.1f, 0.1f}, kEps);
    }

    {
        auto opt = MakeDummyOptimizer(1.0f);
        auto sched = CreateLRScheduler(opt, {
                                                .lr_decay_style = "linear",
                                                .lr = 1.0f,
                                                .min_lr = 0.1f,
                                                .lr_decay_iters = 6,
                                                .lr_warmup_iters = 2,
                                                .lr_warmup_init = 0.0f,
                                            });
        EXPECT_NEAR(sched->GetLR(), 0.0f, kEps);
        ExpectStepSequence(sched, {0.5f, 1.0f, 0.775f, 0.55f, 0.325f, 0.1f}, kEps);
    }

    {
        auto opt = MakeDummyOptimizer(1.0f);
        auto sched = CreateLRScheduler(opt, {
                                                .lr_decay_style = "cosine",
                                                .lr = 1.0f,
                                                .min_lr = 0.0f,
                                                .lr_decay_iters = 4,
                                                .lr_warmup_iters = 0,
                                                .lr_warmup_init = 0.0f,
                                            });
        ExpectStepSequence(sched, {0.853553f, 0.5f, 0.146447f, 0.0f}, 1e-5f);
    }

    {
        auto opt = MakeDummyOptimizer(1.0f);
        auto sched = CreateLRScheduler(opt, {
                                                .lr_decay_style = "inverse-square-root",
                                                .lr = 1.0f,
                                                .min_lr = 0.1f,
                                                .lr_decay_iters = 10,
                                                .lr_warmup_iters = 2,
                                                .lr_warmup_init = 0.0f,
                                            });
        ExpectStepSequence(sched, {0.5f, 1.0f, 0.8164966f, 0.7071068f, 0.6324555f}, 1e-5f);
    }
}

TEST_P(LRSchedulerTest, TrainingSchedulerFactoryReturnsNullForNoneStyle) {
    auto opt = MakeDummyOptimizer(0.1f);
    auto sched = CreateLRScheduler(opt, {
                                            .lr_decay_style = "none",
                                            .lr = 0.1f,
                                            .min_lr = 0.0f,
                                            .lr_decay_iters = 10,
                                            .lr_warmup_iters = 0,
                                            .lr_warmup_init = 0.0f,
                                        });

    EXPECT_EQ(sched, nullptr);
}

TEST_P(LRSchedulerTest, RejectsInvalidSchedulerConfigurations) {
    EXPECT_DEATH(
        {
            auto opt = MakeDummyOptimizer(0.1f);
            auto sched = LRScheduler::Create<StepLR>(opt, /*step_size=*/0, /*gamma=*/0.1f);
            (void)sched;
        },
        "");

    EXPECT_DEATH(
        {
            auto opt = MakeDummyOptimizer(0.1f);
            auto sched = LRScheduler::Create<LambdaLR>(opt, /*lr_lambda=*/LambdaLR::LambdaFunc{});
            (void)sched;
        },
        "");

    EXPECT_DEATH(
        {
            auto opt1 = MakeDummyOptimizer(0.1f);
            auto opt2 = MakeDummyOptimizer(0.1f);
            auto linear = LRScheduler::Create<LinearLR>(opt1, /*start_factor=*/0.5f, /*end_factor=*/1.0f,
                                                        /*total_iters=*/2);
            auto step_lr = LRScheduler::Create<StepLR>(opt2, /*step_size=*/2, /*gamma=*/0.5f);
            auto sched = LRScheduler::Create<SequentialLR>(
                opt1, /*schedulers=*/std::vector<std::shared_ptr<LRScheduler>>{linear, step_lr},
                /*milestones=*/std::vector<int64_t>{1});
            (void)sched;
        },
        "");

    EXPECT_DEATH(
        {
            auto opt = MakeDummyOptimizer(0.1f);
            auto step_lr = LRScheduler::Create<StepLR>(opt, /*step_size=*/2, /*gamma=*/0.5f);
            std::shared_ptr<LRScheduler> sched = LRScheduler::Create<ChainedScheduler>(
                opt,
                /*schedulers=*/std::vector<std::shared_ptr<LRScheduler>>{step_lr});
            sched->Step(0);
        },
        "");
}

INFINI_TRAIN_REGISTER_TEST(LRSchedulerTest);
