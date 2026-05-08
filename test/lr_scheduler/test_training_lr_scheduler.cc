#include "infini_train/include/lr_scheduler.h"
#include "test/lr_scheduler/test_helpers.h"

using namespace infini_train;

namespace {

void TestConstantLR() {
    auto opt = MakeDummyOptimizer(0.1f);
    auto sched = CreateLRScheduler(opt, {
                                            .lr_decay_style = "constant",
                                            .lr = 0.1f,
                                            .min_lr = 0.0f,
                                            .lr_decay_iters = 10,
                                            .lr_warmup_iters = 0,
                                            .lr_warmup_init = 0.0f,
                                        });

    ASSERT_FLOAT_EQ(sched->GetLR(), 0.1f);
    for (int i = 0; i < 5; ++i) { sched->Step(); }
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.1f);
}

void TestLinearWarmupAndDecay() {
    auto opt = MakeDummyOptimizer(1.0f);
    auto sched = CreateLRScheduler(opt, {
                                            .lr_decay_style = "linear",
                                            .lr = 1.0f,
                                            .min_lr = 0.1f,
                                            .lr_decay_iters = 6,
                                            .lr_warmup_iters = 2,
                                            .lr_warmup_init = 0.0f,
                                        });

    ASSERT_FLOAT_EQ(sched->GetLR(), 0.0f);
    sched->Step();
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.5f);
    sched->Step();
    ASSERT_FLOAT_EQ(sched->GetLR(), 1.0f);
    sched->Step();
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.775f);
    for (int i = 0; i < 3; ++i) { sched->Step(); }
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.1f);
}

void TestCosineDecay() {
    auto opt = MakeDummyOptimizer(1.0f);
    auto sched = CreateLRScheduler(opt, {
                                            .lr_decay_style = "cosine",
                                            .lr = 1.0f,
                                            .min_lr = 0.0f,
                                            .lr_decay_iters = 4,
                                            .lr_warmup_iters = 0,
                                            .lr_warmup_init = 0.0f,
                                        });

    ASSERT_FLOAT_EQ(sched->GetLR(), 1.0f);
    sched->Step();
    sched->Step();
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.5f);
    sched->Step();
    sched->Step();
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.0f);
}

void TestInverseSquareRootDecay() {
    auto opt = MakeDummyOptimizer(1.0f);
    auto sched = CreateLRScheduler(opt, {
                                            .lr_decay_style = "inverse-square-root",
                                            .lr = 1.0f,
                                            .min_lr = 0.1f,
                                            .lr_decay_iters = 10,
                                            .lr_warmup_iters = 2,
                                            .lr_warmup_init = 0.0f,
                                        });

    sched->Step();
    sched->Step();
    ASSERT_FLOAT_EQ(sched->GetLR(), 1.0f);
    for (int i = 0; i < 6; ++i) { sched->Step(); }
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.5f);
    for (int i = 0; i < 92; ++i) { sched->Step(); }
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.1f);
}

} // namespace

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    TestConstantLR();
    TestLinearWarmupAndDecay();
    TestCosineDecay();
    TestInverseSquareRootDecay();

    return g_fail_count > 0 ? 1 : 0;
}
