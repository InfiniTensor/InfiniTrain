#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/utils.h"

namespace {

using namespace infini_train::nn::parallel;

using GroupNameFn = std::string (*)(int);
using GroupRanksFn = std::vector<int> (*)(int);

void ExpectNamesIdentifyExactRankSets(GroupNameFn get_name, GroupRanksFn get_ranks) {
    std::map<std::string, std::vector<int>> name_to_ranks;
    std::map<std::vector<int>, std::string> ranks_to_name;

    for (int rank = 0; rank < global::GetWorldSize(); ++rank) {
        const auto name = get_name(rank);
        const auto ranks = get_ranks(rank);
        EXPECT_NE(std::find(ranks.begin(), ranks.end(), rank), ranks.end());

        const auto [name_it, inserted_name] = name_to_ranks.emplace(name, ranks);
        if (!inserted_name) {
            EXPECT_EQ(name_it->second, ranks);
        }

        const auto [ranks_it, inserted_ranks] = ranks_to_name.emplace(ranks, name);
        if (!inserted_ranks) {
            EXPECT_EQ(ranks_it->second, name);
        }
    }
}

TEST(GlobalEnvTest, DistinguishesDenseAndExpertDataParallelSizes) {
    EXPECT_EQ(global::GetWorldSize(), 16);
    EXPECT_EQ(global::GetTensorParallelSize(), 2);
    EXPECT_EQ(global::GetExpertTensorParallelSize(), 2);
    EXPECT_EQ(global::GetPipelineParallelSize(), 2);
    EXPECT_EQ(global::GetExpertParallelSize(), 2);
    EXPECT_EQ(global::GetDataParallelSize(), 4);
    EXPECT_EQ(global::GetExpertDataParallelSize(), 2);
}

TEST(GlobalEnvTest, OwnsDenseAndExpertRankViewsOverTheSameWorld) {
    const auto &dense = global::GetDenseRankGenerator();
    const auto &expert = global::GetExpertRankGenerator();

    EXPECT_EQ(dense.WorldSize(), global::GetWorldSize());
    EXPECT_EQ(expert.WorldSize(), global::GetWorldSize());
    EXPECT_EQ(dense.AxisSize(global::DP), 4);
    EXPECT_EQ(dense.AxisSize(global::TP), 2);
    EXPECT_EQ(dense.AxisSize(global::EP), 1);
    EXPECT_EQ(expert.AxisSize(global::DP), 2);
    EXPECT_EQ(expert.AxisSize(global::TP), 2);
    EXPECT_EQ(expert.AxisSize(global::EP), 2);

    EXPECT_EQ(dense.GetRanks(global::PP), expert.GetRanks(global::PP));
}

TEST(GlobalEnvTest, PublicGroupHelpersUseMainstreamMoeDomains) {
    constexpr int kRank = 7;

    EXPECT_EQ(GetDataParallelGroupRanks(kRank), (std::vector<int>{1, 3, 5, 7}));
    EXPECT_EQ(GetDataParallelProcessGroupName(kRank), "DP1");

    EXPECT_EQ(GetTensorParallelGroupRanks(kRank), (std::vector<int>{6, 7}));
    EXPECT_EQ(GetTensorParallelProcessGroupName(kRank), "TP3");

    EXPECT_EQ(GetExpertTensorParallelGroupRanks(kRank), (std::vector<int>{6, 7}));
    EXPECT_EQ(GetExpertTensorParallelProcessGroupName(kRank), "ETP3");

    EXPECT_EQ(GetPipelineParallelGroupRanks(kRank), (std::vector<int>{7, 15}));
    EXPECT_EQ(GetPipelineParallelProcessGroupName(kRank), "PP7");

    EXPECT_EQ(GetExpertDataParallelGroupRanks(kRank), (std::vector<int>{3, 7}));
    EXPECT_EQ(GetExpertDataParallelProcessGroupName(kRank), "EDP3");

    EXPECT_EQ(GetExpertTensorAndExpertParallelGroupRanks(kRank), (std::vector<int>{4, 5, 6, 7}));
    EXPECT_EQ(GetExpertTensorAndExpertParallelProcessGroupName(kRank), "ETP_EP1");

    EXPECT_EQ(GetExpertParallelGroupRanks(kRank), (std::vector<int>{5, 7}));
    EXPECT_EQ(GetExpertParallelProcessGroupName(kRank), "EP3");

    constexpr int kSecondPipelineStageRank = 15;
    EXPECT_EQ(GetDataParallelGroupRanks(kSecondPipelineStageRank), (std::vector<int>{9, 11, 13, 15}));
    EXPECT_EQ(GetDataParallelProcessGroupName(kSecondPipelineStageRank), "DP3");
    EXPECT_EQ(GetExpertDataParallelGroupRanks(kSecondPipelineStageRank), (std::vector<int>{11, 15}));
    EXPECT_EQ(GetExpertDataParallelProcessGroupName(kSecondPipelineStageRank), "EDP7");
    EXPECT_EQ(GetExpertTensorAndExpertParallelGroupRanks(kSecondPipelineStageRank), (std::vector<int>{12, 13, 14, 15}));
    EXPECT_EQ(GetExpertTensorAndExpertParallelProcessGroupName(kSecondPipelineStageRank), "ETP_EP3");
}

TEST(GlobalEnvTest, GroupNamesAreStableForEveryMember) {
    ExpectNamesIdentifyExactRankSets(GetDataParallelProcessGroupName, GetDataParallelGroupRanks);
    ExpectNamesIdentifyExactRankSets(GetExpertDataParallelProcessGroupName, GetExpertDataParallelGroupRanks);
    ExpectNamesIdentifyExactRankSets(GetTensorParallelProcessGroupName, GetTensorParallelGroupRanks);
    ExpectNamesIdentifyExactRankSets(GetExpertTensorParallelProcessGroupName, GetExpertTensorParallelGroupRanks);
    ExpectNamesIdentifyExactRankSets(GetPipelineParallelProcessGroupName, GetPipelineParallelGroupRanks);
    ExpectNamesIdentifyExactRankSets(GetExpertParallelProcessGroupName, GetExpertParallelGroupRanks);
    ExpectNamesIdentifyExactRankSets(GetExpertTensorAndExpertParallelProcessGroupName,
                                     GetExpertTensorAndExpertParallelGroupRanks);
}

TEST(GlobalEnvTest, DefaultOverviewReportsOnlyTheDenseView) {
    const std::string overview = global::ProcessGroupOverview();

    EXPECT_NE(overview.find("config: {DP=4, TP=2, PP=2}"), std::string::npos);
    EXPECT_NE(overview.find("[Dense Rank View] shape={TP=2, DP=4, PP=2}"), std::string::npos);
    EXPECT_EQ(overview.find("[Expert Rank View]"), std::string::npos);
    EXPECT_EQ(overview.find("[EDP]"), std::string::npos);
    EXPECT_EQ(overview.find("[ETP]"), std::string::npos);
    EXPECT_EQ(overview.find("[EP]"), std::string::npos);
    EXPECT_EQ(overview.find("[ETP_EP]"), std::string::npos);
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    setenv("PROC_WORLD_SIZE", "16", 1);
    infini_train::nn::parallel::global::GlobalEnv::Instance().Init(
        /*nthread_per_process=*/1,
        /*tensor_parallel_size=*/2,
        /*sequence_parallel_enabled=*/false,
        /*pipeline_parallel_size=*/2,
        /*virtual_pipeline_parallel_size=*/1,
        /*expert_parallel_size=*/2,
        /*expert_tensor_parallel_size=*/std::nullopt);
    return RUN_ALL_TESTS();
}
