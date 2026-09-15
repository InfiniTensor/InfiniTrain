#include <filesystem>
#include <fstream>
#include <string>

#include "gtest/gtest.h"

#include "infini_train/include/checkpoint/checkpoint.h"
#include "infini_train/include/checkpoint/checkpoint_manager.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;
namespace nn = infini_train::nn;

class TrainerStateTest : public test::InfiniTrainTest {};

TEST_P(TrainerStateTest, DefaultValues) {
    TrainerState state;
    EXPECT_EQ(state.global_step, 0);
    EXPECT_EQ(state.consumed_train_samples, 0);
    EXPECT_EQ(state.n_layer, 0);
    EXPECT_EQ(state.n_head, 0);
    EXPECT_EQ(state.n_kv_head, 0);
    EXPECT_EQ(state.n_embd, 0);
    EXPECT_EQ(state.vocab_size, 0);
    EXPECT_EQ(state.ddp_size, 1);
    EXPECT_EQ(state.tp_size, 1);
    EXPECT_EQ(state.sp_size, 1);
    EXPECT_EQ(state.pp_size, 1);
    EXPECT_EQ(state.vpp_size, 1);
}

TEST_P(TrainerStateTest, TrainerStateFileCreated) {
    auto dir = std::filesystem::temp_directory_path() / "test_trainer_json";
    std::filesystem::remove_all(dir);

    TrainerState saved{.global_step = 30, .consumed_train_samples = 1200};

    auto model = std::make_shared<nn::Linear>(1, 2, true, GetDevice());
    auto p = std::make_shared<Tensor>(std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice());
    p->Fill(1.0f);
    *model->mutable_parameter("weight") = p;
    auto opt = std::make_shared<optimizers::SGD>(model->Parameters(), 0.01);

    Checkpoint::Save(dir, *model, opt.get(), saved, nullptr);

    EXPECT_TRUE(std::filesystem::exists(dir / "trainer_state.json"));

    std::ifstream ifs(dir / "trainer_state.json");
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("\"global_step\""), std::string::npos);
    EXPECT_NE(content.find("\"consumed_train_samples\""), std::string::npos);

    std::filesystem::remove_all(dir);
}

TEST_P(TrainerStateTest, RoundTrip) {
    auto dir = std::filesystem::temp_directory_path() / "test_trainer_rt";
    std::filesystem::remove_all(dir);

    TrainerState saved{
        .global_step = 99,
        .consumed_train_samples = 5000,
        .n_layer = 24,
        .n_head = 16,
        .n_kv_head = 8,
        .n_embd = 1024,
        .vocab_size = 128256,
        .ddp_size = 2,
        .tp_size = 1,
        .sp_size = 1,
        .pp_size = 2,
        .vpp_size = 4,
    };

    auto model1 = std::make_shared<nn::Linear>(1, 3, true, GetDevice());
    auto p1 = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice());
    p1->Fill(0.5f);
    *model1->mutable_parameter("weight") = p1;
    auto opt1 = std::make_shared<optimizers::SGD>(model1->Parameters(), 0.01);

    Checkpoint::Save(dir, *model1, nullptr, saved, nullptr);

    auto model2 = std::make_shared<nn::Linear>(1, 3, true, GetDevice());
    auto p2 = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice());
    p2->Fill(0.0f);
    *model2->mutable_parameter("weight") = p2;
    auto opt2 = std::make_shared<optimizers::SGD>(model2->Parameters(), 0.01);

    TrainerState loaded;
    Checkpoint::Load(dir, *model2, nullptr, loaded, nullptr);

    EXPECT_EQ(loaded.global_step, 99);
    EXPECT_EQ(loaded.consumed_train_samples, 5000);
    EXPECT_EQ(loaded.n_layer, 24);
    EXPECT_EQ(loaded.n_head, 16);
    EXPECT_EQ(loaded.n_kv_head, 8);
    EXPECT_EQ(loaded.n_embd, 1024);
    EXPECT_EQ(loaded.vocab_size, 128256);
    EXPECT_EQ(loaded.ddp_size, 2);
    EXPECT_EQ(loaded.pp_size, 2);
    EXPECT_EQ(loaded.vpp_size, 4);

    std::filesystem::remove_all(dir);
}

TEST_P(TrainerStateTest, DataLoaderSkipUsesCurrentBatchConfiguration) {
    EXPECT_EQ(DataLoaderBatchesToSkip(/*consumed_train_samples=*/400, /*local_batch_size=*/4,
                                      /*ddp_world_size=*/2),
              50);
    EXPECT_EQ(DataLoaderBatchesToSkip(/*consumed_train_samples=*/400, /*local_batch_size=*/8,
                                      /*ddp_world_size=*/2),
              25);
}

TEST_P(TrainerStateTest, DataLoaderSkipRejectsUnalignedConfiguration) {
    EXPECT_DEATH(DataLoaderBatchesToSkip(/*consumed_train_samples=*/400, /*local_batch_size=*/6,
                                         /*ddp_world_size=*/4),
                 "does not align");
}

INFINI_TRAIN_REGISTER_TEST(TrainerStateTest);
