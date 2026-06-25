#include <filesystem>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/checkpoint/checkpoint.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;
namespace nn = infini_train::nn;

class CheckpointSerializationTest : public test::InfiniTrainTest {};

TEST_P(CheckpointSerializationTest, SaveAndLoadModelFP32) {
    auto dir = std::filesystem::temp_directory_path() / "test_ckpt_fp32";
    std::filesystem::remove_all(dir);

    auto model1 = std::make_shared<nn::Linear>(3, 2, true, GetDevice());
    auto p1 = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    p1->Fill(0.42f);
    *model1->mutable_parameter("weight") = p1;
    auto p2 = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice());
    p2->Fill(-1.5f);
    *model1->mutable_parameter("bias") = p2;

    auto opt1 = std::make_shared<optimizers::Adam>(model1->Parameters(), 0.01);
    TrainerState saved{.global_step = 42, .consumed_batches = 100};
    Checkpoint::Save(dir, *model1, opt1.get(), saved, /*save_optimizer_state=*/true);

    auto model2 = std::make_shared<nn::Linear>(3, 2, true, GetDevice());
    auto q1 = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    q1->Fill(0.0f);
    *model2->mutable_parameter("weight") = q1;
    auto q2 = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice());
    q2->Fill(0.0f);
    *model2->mutable_parameter("bias") = q2;
    auto opt2 = std::make_shared<optimizers::Adam>(model2->Parameters(), 0.01);

    TrainerState loaded;
    Checkpoint::Load(dir, *model2, opt2.get(), loaded, /*load_optimizer_state=*/true);

    EXPECT_EQ(loaded.global_step, 42);
    EXPECT_EQ(loaded.consumed_batches, 100);

    auto w1_cpu = model2->parameter("weight")->To(Device());
    const float *data = static_cast<const float *>(w1_cpu.DataPtr());
    for (int i = 0; i < 6; ++i) { EXPECT_NEAR(data[i], 0.42f, 1e-6); }

    std::filesystem::remove_all(dir);
}

INFINI_TRAIN_REGISTER_TEST(CheckpointSerializationTest);
