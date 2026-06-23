#include <filesystem>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/checkpoint/checkpoint.h"
#include "infini_train/include/lr_scheduler.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;
namespace nn = infini_train::nn;

namespace {
constexpr float kBaseLR = 0.1f;
constexpr float kEps = 1e-6f;

TrainingLRSchedulerConfig MakeSchedulerConfig() {
    return {
        .lr_decay_style = "linear",
        .lr = kBaseLR,
        .min_lr = 0.01f,
        .lr_decay_iters = 8,
        .lr_warmup_iters = 2,
        .lr_warmup_init = 0.0f,
    };
}

std::shared_ptr<nn::Linear> MakeModel(Device device) {
    auto model = std::make_shared<nn::Linear>(1, 2, true, device);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT32, device);
    weight->Fill(0.5f);
    *model->mutable_parameter("weight") = weight;
    return model;
}

void StepTimes(const std::shared_ptr<LRScheduler> &scheduler, int times) {
    for (int i = 0; i < times; ++i) { scheduler->Step(); }
}
} // namespace

class LRSchedulerCheckpointTest : public test::InfiniTrainTest {};

TEST_P(LRSchedulerCheckpointTest, SaveAndLoadLRSchedulerState) {
    auto dir = std::filesystem::temp_directory_path() / "test_lr_scheduler_ckpt";
    std::filesystem::remove_all(dir);

    auto model_ref = MakeModel(GetDevice());
    auto opt_ref = std::make_shared<optimizers::SGD>(model_ref->Parameters(), kBaseLR);
    auto sched_ref = CreateLRScheduler(opt_ref, MakeSchedulerConfig());
    StepTimes(sched_ref, 6);

    auto model1 = MakeModel(GetDevice());
    auto opt1 = std::make_shared<optimizers::SGD>(model1->Parameters(), kBaseLR);
    auto sched1 = CreateLRScheduler(opt1, MakeSchedulerConfig());
    StepTimes(sched1, 3);

    TrainerState saved{.global_step = 3, .consumed_batches = 12};
    Checkpoint::Save(dir, *model1, opt1.get(), saved, /*save_optimizer_state=*/false, sched1.get(),
                     /*save_lr_scheduler_state=*/true);
    EXPECT_TRUE(std::filesystem::exists(dir / "lr_scheduler.ckpt"));

    auto model2 = MakeModel(GetDevice());
    auto opt2 = std::make_shared<optimizers::SGD>(model2->Parameters(), kBaseLR);
    auto sched2 = CreateLRScheduler(opt2, MakeSchedulerConfig());

    TrainerState loaded;
    Checkpoint::Load(dir, *model2, opt2.get(), loaded, /*load_optimizer_state=*/false, sched2.get(),
                     /*load_lr_scheduler_state=*/true);

    EXPECT_EQ(loaded.global_step, 3);
    EXPECT_EQ(loaded.consumed_batches, 12);
    EXPECT_EQ(sched2->LastStep(), sched1->LastStep());
    EXPECT_NEAR(sched2->GetLR(), sched1->GetLR(), kEps);

    StepTimes(sched2, 3);
    EXPECT_EQ(sched2->LastStep(), sched_ref->LastStep());
    EXPECT_NEAR(sched2->GetLR(), sched_ref->GetLR(), kEps);
    EXPECT_NEAR(opt2->learning_rate(), opt_ref->learning_rate(), kEps);

    std::filesystem::remove_all(dir);
}

TEST_P(LRSchedulerCheckpointTest, HonorsLRSchedulerStateFlags) {
    auto dir = std::filesystem::temp_directory_path() / "test_lr_scheduler_ckpt_flags";
    std::filesystem::remove_all(dir);

    auto model1 = MakeModel(GetDevice());
    auto opt1 = std::make_shared<optimizers::SGD>(model1->Parameters(), kBaseLR);
    auto sched1 = CreateLRScheduler(opt1, MakeSchedulerConfig());
    StepTimes(sched1, 3);

    TrainerState saved{.global_step = 3};
    Checkpoint::Save(dir, *model1, opt1.get(), saved, /*save_optimizer_state=*/false, sched1.get(),
                     /*save_lr_scheduler_state=*/false);
    EXPECT_FALSE(std::filesystem::exists(dir / "lr_scheduler.ckpt"));

    Checkpoint::Save(dir, *model1, opt1.get(), saved, /*save_optimizer_state=*/false, sched1.get(),
                     /*save_lr_scheduler_state=*/true);
    ASSERT_TRUE(std::filesystem::exists(dir / "lr_scheduler.ckpt"));

    auto model2 = MakeModel(GetDevice());
    auto opt2 = std::make_shared<optimizers::SGD>(model2->Parameters(), kBaseLR);
    auto sched2 = CreateLRScheduler(opt2, MakeSchedulerConfig());
    const auto initial_step = sched2->LastStep();
    const auto initial_lr = sched2->GetLR();

    TrainerState loaded;
    Checkpoint::Load(dir, *model2, opt2.get(), loaded, /*load_optimizer_state=*/false, sched2.get(),
                     /*load_lr_scheduler_state=*/false);

    EXPECT_EQ(sched2->LastStep(), initial_step);
    EXPECT_NEAR(sched2->GetLR(), initial_lr, kEps);

    std::filesystem::remove_all(dir);
}

INFINI_TRAIN_REGISTER_TEST(LRSchedulerCheckpointTest);
