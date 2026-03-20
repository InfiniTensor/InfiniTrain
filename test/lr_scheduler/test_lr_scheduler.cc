#include <cmath>
#include <iostream>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/lr_scheduler.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

namespace {

constexpr float kBaseLR = 0.1f;
constexpr float kEps = 1e-7f;

class IdentityScheduler : public LRScheduler {
public:
    IdentityScheduler(std::shared_ptr<Optimizer> optimizer, int64_t last_step = -1)
        : LRScheduler(std::move(optimizer), last_step) {}
    ~IdentityScheduler() override = default;

protected:
    float GetClosedFormLR() const override { return base_lr_; }
};

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

bool FloatEq(float a, float b) { return std::fabs(a - b) < kEps; }

int g_fail_count = 0;

void Check(bool cond, const char *expr, int line) {
    if (!cond) {
        std::cerr << "FAIL [line " << line << "]: " << expr << std::endl;
        ++g_fail_count;
    }
}

#define ASSERT_TRUE(cond) Check((cond), #cond, __LINE__)
#define ASSERT_FLOAT_EQ(a, b) Check(FloatEq((a), (b)), #a " == " #b, __LINE__)

// T1: Init
void TestInitialState() {
    std::cout << "[T1] TestInitialState" << std::endl;
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<IdentityScheduler>(opt);

    ASSERT_FLOAT_EQ(sched->GetLR(), kBaseLR);
    ASSERT_TRUE(sched->LastStep() == 0);
    ASSERT_FLOAT_EQ(opt->GetLearningRate(), kBaseLR);
}

// T2: SingleStep
void TestSingleStep() {
    std::cout << "[T2] TestSingleStep" << std::endl;
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<IdentityScheduler>(opt);

    sched->Step();

    ASSERT_TRUE(sched->LastStep() == 1);
    ASSERT_FLOAT_EQ(sched->GetLR(), kBaseLR);
    ASSERT_FLOAT_EQ(opt->GetLearningRate(), kBaseLR);
}

// T3: ComputeLR
void TestLinearDecay() {
    std::cout << "[T3] TestLinearDecay" << std::endl;
    constexpr int64_t kTotalSteps = 10;
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<LinearDecayScheduler>(opt, kTotalSteps);
    ASSERT_FLOAT_EQ(sched->GetLR(), kBaseLR);
    ASSERT_FLOAT_EQ(opt->GetLearningRate(), kBaseLR);

    sched->Step(); // last_step = 1 -> 0.09
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.09f);
    ASSERT_FLOAT_EQ(opt->GetLearningRate(), 0.09f);

    for (int i = 0; i < 4; ++i) { sched->Step(); } // last_step = 5
    ASSERT_FLOAT_EQ(sched->GetLR(), 0.05f);
    ASSERT_FLOAT_EQ(opt->GetLearningRate(), 0.05f);
}

// T4: State → LoadState 
void TestStateRoundTrip() {
    std::cout << "[T4] TestStateRoundTrip" << std::endl;
    constexpr int64_t kTotalSteps = 20;
    auto opt = MakeDummyOptimizer(kBaseLR);
    auto sched = LRScheduler::Create<LinearDecayScheduler>(opt, kTotalSteps);

    for (int i = 0; i < 7; ++i) { sched->Step(); }

    StateDict saved = sched->State();

    ASSERT_TRUE(saved.count("last_step") == 1);
    ASSERT_TRUE(saved.count("recover_lr") == 1);
    ASSERT_TRUE(saved.count("base_lr") == 1);

    auto opt2 = MakeDummyOptimizer(kBaseLR);
    auto sched2 = LRScheduler::Create<LinearDecayScheduler>(opt2, kTotalSteps);
    sched2->LoadState(saved);

    ASSERT_TRUE(sched2->LastStep() == 7);
    ASSERT_FLOAT_EQ(sched2->GetLR(), sched->GetLR());
    ASSERT_FLOAT_EQ(opt2->GetLearningRate(), sched->GetLR());
}

// T5: resume Step
void TestResumeAndContinue() {
    std::cout << "[T5] TestResumeAndContinue" << std::endl;
    constexpr int64_t kTotalSteps = 20;

    auto opt_ref = MakeDummyOptimizer(kBaseLR);
    auto sched_ref = LRScheduler::Create<LinearDecayScheduler>(opt_ref, kTotalSteps);
    for (int i = 0; i < 10; ++i) { sched_ref->Step(); }
    float lr_at_10 = sched_ref->GetLR();

    auto opt_a = MakeDummyOptimizer(kBaseLR);
    auto sched_a = LRScheduler::Create<LinearDecayScheduler>(opt_a, kTotalSteps);
    for (int i = 0; i < 5; ++i) { sched_a->Step(); }
    StateDict checkpoint = sched_a->State();

    auto opt_b = MakeDummyOptimizer(kBaseLR);
    auto sched_b = LRScheduler::Create<LinearDecayScheduler>(opt_b, kTotalSteps);
    sched_b->LoadState(checkpoint);
    for (int i = 0; i < 5; ++i) { sched_b->Step(); }

    ASSERT_FLOAT_EQ(sched_b->GetLR(), lr_at_10);
    ASSERT_TRUE(sched_b->LastStep() == sched_ref->LastStep());
}

} // namespace

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    std::cout << "========================================" << std::endl;
    std::cout << "    LRScheduler Base Class Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    TestInitialState();
    TestSingleStep();
    TestLinearDecay();
    TestStateRoundTrip();
    TestResumeAndContinue();

    std::cout << "========================================" << std::endl;
    if (g_fail_count == 0) {
        std::cout << "    All Tests PASSED" << std::endl;
    } else {
        std::cout << "    " << g_fail_count << " test(s) FAILED" << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return g_fail_count > 0 ? 1 : 0;
}
