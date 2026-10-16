#include <cstdlib>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/utils.h"

namespace {

using namespace infini_train::nn::parallel;

TEST(RankGeneratorTest, GeneratesDenseAndExpertGroups) {
    const global::RankGenerator dense(/*tp=*/2, /*ep=*/1, /*dp=*/4, /*pp=*/2);
    const global::RankGenerator expert(/*etp=*/1, /*ep=*/2, /*edp=*/4, /*pp=*/2);

    EXPECT_EQ(dense.GroupRanks(global::DP, 7), (std::vector<int>{1, 3, 5, 7}));
    EXPECT_EQ(dense.GroupRanks(global::TP, 7), (std::vector<int>{6, 7}));
    EXPECT_EQ(expert.GroupRanks(global::DP, 7), (std::vector<int>{1, 3, 5, 7}));
    EXPECT_EQ(expert.GroupRanks(global::TP, 7), (std::vector<int>{7}));
    EXPECT_EQ(expert.GroupRanks(global::EP, 7), (std::vector<int>{6, 7}));
    EXPECT_EQ(expert.GroupRanks({global::TP, global::EP}, 7), (std::vector<int>{6, 7}));

    EXPECT_EQ(dense.GroupId(global::DP, 7), 2);
    EXPECT_EQ(dense.GetRanks(global::DP).at(2), (std::vector<int>{1, 3, 5, 7}));
    EXPECT_EQ(expert.GroupId({global::TP, global::EP}, 7), 6);
    EXPECT_EQ(expert.GetRanks({global::TP, global::EP}).at(6), (std::vector<int>{6, 7}));
    EXPECT_EQ(dense.GetRanks(global::PP), expert.GetRanks(global::PP));
}

TEST(GlobalEnvTest, ExposesDenseAndExpertViewsWithoutRenamingDenseGroups) {
    // These IDs match the pre-RankGenerator dense Layout implementation.
    for (int rank = 0; rank < global::GetWorldSize(); ++rank) {
        const int tp = rank % 2;
        const int dp = (rank / 2) % 4;
        const int pp = rank / 8;
        EXPECT_EQ(GetDataParallelProcessGroupName(rank), "DP" + std::to_string(pp + tp * 2));
        EXPECT_EQ(GetTensorParallelProcessGroupName(rank), "TP" + std::to_string(pp + dp * 2));
        EXPECT_EQ(GetPipelineParallelProcessGroupName(rank), "PP" + std::to_string(dp + tp * 4));
    }
}

} // namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    setenv("WORLD_SIZE", "16", 1);
    setenv("LOCAL_WORLD_SIZE", "1", 1);
    setenv("RANK", "0", 1);
    setenv("LOCAL_RANK", "0", 1);
    infini_train::nn::parallel::global::GlobalEnv::Instance().Init(
        /*nthread_per_process=*/1,
        /*tensor_parallel_size=*/2,
        /*sequence_parallel_enabled=*/false,
        /*pipeline_parallel_size=*/2,
        /*virtual_pipeline_parallel_size=*/1,
        /*expert_parallel_size=*/2,
        /*expert_tensor_parallel_size=*/1);
    return RUN_ALL_TESTS();
}
