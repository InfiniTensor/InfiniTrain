#include <cstdlib>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/utils.h"

namespace {

using namespace infini_train::nn::parallel;

TEST(GlobalEnvEtpTest, UsesIndependentDenseAndExpertTensorParallelSizes) {
    EXPECT_EQ(global::GetWorldSize(), 48);
    EXPECT_EQ(global::GetTensorParallelSize(), 3);
    EXPECT_EQ(global::GetExpertTensorParallelSize(), 2);
    EXPECT_EQ(global::GetExpertParallelSize(), 2);
    EXPECT_EQ(global::GetPipelineParallelSize(), 2);
    EXPECT_EQ(global::GetDataParallelSize(), 8);
    EXPECT_EQ(global::GetExpertDataParallelSize(), 6);

    const auto &dense = global::GetDenseRankGenerator();
    const auto &expert = global::GetExpertRankGenerator();
    EXPECT_EQ(dense.AxisSize(global::TP), 3);
    EXPECT_EQ(expert.AxisSize(global::TP), 2);
    EXPECT_EQ(dense.AxisSize(global::DP), 8);
    EXPECT_EQ(expert.AxisSize(global::DP), 6);
    EXPECT_EQ(dense.WorldSize(), global::GetWorldSize());
    EXPECT_EQ(expert.WorldSize(), global::GetWorldSize());
}

TEST(GlobalEnvEtpTest, ExpertGroupsUseTheExpertTensorParallelAxis) {
    constexpr int kRank = 11;

    EXPECT_EQ(GetTensorParallelGroupRanks(kRank), (std::vector<int>{9, 10, 11}));
    EXPECT_EQ(GetTensorParallelProcessGroupName(kRank), "TP3");

    EXPECT_EQ(GetExpertTensorParallelGroupRanks(kRank), (std::vector<int>{10, 11}));
    EXPECT_EQ(GetExpertTensorParallelProcessGroupName(kRank), "ETP5");

    EXPECT_EQ(GetExpertParallelGroupRanks(kRank), (std::vector<int>{9, 11}));
    EXPECT_EQ(GetExpertParallelProcessGroupName(kRank), "EP5");

    EXPECT_EQ(GetExpertTensorAndExpertParallelGroupRanks(kRank), (std::vector<int>{8, 9, 10, 11}));
    EXPECT_EQ(GetExpertTensorAndExpertParallelProcessGroupName(kRank), "ETP_EP2");

    EXPECT_EQ(GetExpertDataParallelGroupRanks(kRank), (std::vector<int>{3, 7, 11, 15, 19, 23}));
    EXPECT_EQ(GetExpertDataParallelProcessGroupName(kRank), "EDP3");
}

TEST(GlobalEnvEtpTest, DenseAndExpertViewsPreservePipelineGroups) {
    const auto &dense = global::GetDenseRankGenerator();
    const auto &expert = global::GetExpertRankGenerator();
    EXPECT_EQ(dense.GetRanks(global::PP), expert.GetRanks(global::PP));

    constexpr int kRank = 35;
    EXPECT_EQ(GetPipelineParallelGroupRanks(kRank), (std::vector<int>{11, 35}));
    EXPECT_EQ(GetExpertTensorParallelGroupRanks(kRank), (std::vector<int>{34, 35}));
    EXPECT_EQ(GetExpertDataParallelGroupRanks(kRank), (std::vector<int>{27, 31, 35, 39, 43, 47}));
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    setenv("PROC_WORLD_SIZE", "48", 1);
    infini_train::nn::parallel::global::GlobalEnv::Instance().Init(
        /*nthread_per_process=*/1,
        /*tensor_parallel_size=*/3,
        /*sequence_parallel_enabled=*/false,
        /*pipeline_parallel_size=*/2,
        /*virtual_pipeline_parallel_size=*/1,
        /*expert_parallel_size=*/2,
        /*expert_tensor_parallel_size=*/2);
    return RUN_ALL_TESTS();
}
