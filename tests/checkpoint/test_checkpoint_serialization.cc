#include <filesystem>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/checkpoint.h"
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

    auto opt1 = std::make_shared<optimizers::SGD>(model1->Parameters(), 0.01);
    TrainerState saved{.global_step = 42, .consumed_batches = 100};
    Checkpoint::Save(dir, *model1, opt1.get(), saved, false);

    auto model2 = std::make_shared<nn::Linear>(3, 2, true, GetDevice());
    auto q1 = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    q1->Fill(0.0f);
    *model2->mutable_parameter("weight") = q1;
    auto q2 = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice());
    q2->Fill(0.0f);
    *model2->mutable_parameter("bias") = q2;
    auto opt2 = std::make_shared<optimizers::SGD>(model2->Parameters(), 0.01);

    TrainerState loaded;
    Checkpoint::Load(dir, *model2, opt2.get(), loaded, true);

    EXPECT_EQ(loaded.global_step, 42);
    EXPECT_EQ(loaded.consumed_batches, 100);

    auto w1_cpu = model2->parameter("weight")->To(Device());
    const float *data = static_cast<const float *>(w1_cpu.DataPtr());
    for (int i = 0; i < 6; ++i) { EXPECT_NEAR(data[i], 0.42f, 1e-6); }

    std::filesystem::remove_all(dir);
}

TEST_P(CheckpointSerializationTest, InferFormat) {
    auto dir = std::filesystem::temp_directory_path() / "test_ckpt_fmt";
    std::filesystem::remove_all(dir);

    auto model = std::make_shared<nn::Linear>(1, 2, true, GetDevice());
    auto p = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice());
    p->Fill(1.0f);
    *model->mutable_parameter("weight") = p;
    auto opt = std::make_shared<optimizers::SGD>(model->Parameters(), 0.01);
    TrainerState state;
    Checkpoint::Save(dir, *model, opt.get(), state, false);

    auto model2 = std::make_shared<nn::Linear>(1, 2, true, GetDevice());
    auto p2 = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice());
    p2->Fill(0.0f);
    *model2->mutable_parameter("weight") = p2;
    TrainerState loaded;
    Checkpoint::Load(dir, *model2, nullptr, loaded, true);

    EXPECT_NEAR(static_cast<const float *>(model2->parameter("weight")->To(Device()).DataPtr())[0], 1.0f, 1e-6);

    std::filesystem::remove_all(dir);
}

INFINI_TRAIN_REGISTER_TEST(CheckpointSerializationTest);
