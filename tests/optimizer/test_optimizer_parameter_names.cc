#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel_config.h"
#include "infini_train/include/nn/parallel/ddp/distributed_optimizer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class OptimizerParameterNamesTest : public test::InfiniTrainTest {};

TEST_P(OptimizerParameterNamesTest, AdamStateDictUsesStableParameterNames) {
    auto first = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice());
    auto second = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice());
    const NamedParameterList named_parameters{{"transformer.h.0.weight", first}, {"transformer.h.0.bias", second}};
    auto adam = optimizers::Adam::CreateNamed(0.001)(named_parameters);

    const auto state = adam->StateDict();
    EXPECT_TRUE(state.contains("adam.m.transformer.h.0.weight"));
    EXPECT_TRUE(state.contains("adam.v.transformer.h.0.weight"));
    EXPECT_TRUE(state.contains("adam.m.transformer.h.0.bias"));
    EXPECT_TRUE(state.contains("adam.v.transformer.h.0.bias"));
    EXPECT_TRUE(state.contains("adam.t"));

    auto restored = optimizers::Adam::CreateNamed(0.001)(named_parameters);
    restored->LoadStateDict(state);
    EXPECT_EQ(restored->StateDict().size(), state.size());
}

TEST_P(OptimizerParameterNamesTest, ConstructorMatchesNamesToOptimizerParameterOrder) {
    auto first = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice());
    auto second = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice());
    const NamedParameterList named_parameters{{"second", second}, {"first", first}};

    auto adam = optimizers::Adam::CreateNamed(0.001)(named_parameters);
    const auto state = adam->StateDict();

    EXPECT_TRUE(state.contains("adam.m.second"));
    EXPECT_TRUE(state.contains("adam.v.second"));
    EXPECT_TRUE(state.contains("adam.m.first"));
    EXPECT_TRUE(state.contains("adam.v.first"));
}

TEST_P(OptimizerParameterNamesTest, DistributedOptimizerPropagatesNamesToShardOptimizer) {
    ONLY_CUDA();
    REQUIRE_MIN_DEVICES(2);
    if (nn::parallel::global::GetDataParallelSize() != 2) {
        GTEST_SKIP() << "requires PROC_WORLD_SIZE=2";
    }

    const nn::parallel::Rank rank(/*process_rank=*/0, /*thread_rank=*/0, /*process_size=*/1, /*thread_size=*/2);
    auto *pg_factory = nn::parallel::ProcessGroupFactory::Instance(Device::DeviceType::kCUDA);
    pg_factory->GetOrCreate(nn::parallel::GetDataParallelProcessGroupName(rank.GlobalRank()),
                            nn::parallel::GetDataParallelGroupRanks(rank.GlobalRank()));

    auto model = std::make_shared<nn::Linear>(4, 4, /*bias=*/false, GetDevice());
    const auto named_parameters = model->NamedParameters();

    nn::parallel::DistributedDataParallelConfig ddp_config;
    ddp_config.zero_stage = 1;
    ddp_config.overlap_grad_reduce = false;
    ddp_config.overlap_param_gather = false;
    auto ddp_model = std::make_shared<nn::parallel::DistributedDataParallel>(model, rank, ddp_config);

    nn::parallel::DistributedOptimizer optimizer(optimizers::Adam::CreateNamed(0.001), named_parameters,
                                                 std::vector<std::shared_ptr<nn::Module>>{ddp_model},
                                                 /*ddp_world_size=*/2, /*ddp_rank=*/0);
    const auto state = optimizer.StateDict();

    EXPECT_TRUE(state.contains("adam.m.weight"));
    EXPECT_TRUE(state.contains("adam.v.weight"));
    EXPECT_FALSE(state.contains("adam.m.0"));
    EXPECT_FALSE(state.contains("adam.v.0"));
}

TEST_P(OptimizerParameterNamesTest, PreservesNumericKeysWhenNamesAreNotSet) {
    auto parameter = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice());
    auto adam = std::make_shared<optimizers::Adam>(std::vector<std::shared_ptr<Tensor>>{parameter}, 0.001);

    const auto state = adam->StateDict();
    EXPECT_TRUE(state.contains("adam.m.0"));
    EXPECT_TRUE(state.contains("adam.v.0"));
}

INFINI_TRAIN_REGISTER_TEST(OptimizerParameterNamesTest);
