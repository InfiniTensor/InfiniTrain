#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/utils.h"

namespace {

using namespace infini_train::nn::parallel;

TEST(RankGeneratorTest, GeneratesDenseAndExpertGroups) {
    const global::RankGenerator dense(/*tp=*/2, /*ep=*/1, /*dp=*/4, /*pp=*/2);
    const global::RankGenerator expert(/*etp=*/2, /*ep=*/2, /*edp=*/2, /*pp=*/2);

    EXPECT_EQ(dense.GroupRanks(global::DP, 7), (std::vector<int>{1, 3, 5, 7}));
    EXPECT_EQ(dense.GroupRanks(global::TP, 7), (std::vector<int>{6, 7}));
    EXPECT_EQ(expert.GroupRanks(global::DP, 7), (std::vector<int>{3, 7}));
    EXPECT_EQ(expert.GroupRanks(global::EP, 7), (std::vector<int>{5, 7}));
    EXPECT_EQ(expert.GroupRanks({global::TP, global::EP}, 7), (std::vector<int>{4, 5, 6, 7}));
    EXPECT_EQ(dense.GetRanks(global::PP), expert.GetRanks(global::PP));
}

TEST(RankGeneratorTest, GroupIdsIndexGeneratedGroups) {
    const global::RankGenerator generator(/*tp=*/2, /*ep=*/3, /*dp=*/2, /*pp=*/2);
    const auto verify = [&generator](std::initializer_list<global::Axis> axes) {
        const auto groups = generator.GetRanks(axes);
        std::vector<int> occurrences(generator.WorldSize(), 0);
        for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
            for (const int rank : groups[group_id]) {
                ++occurrences.at(rank);
                EXPECT_EQ(generator.GroupId(axes, rank), static_cast<int>(group_id));
                EXPECT_EQ(generator.GroupRanks(axes, rank), groups[group_id]);
            }
        }
        for (const int count : occurrences) { EXPECT_EQ(count, 1); }
    };

    verify({global::DP});
    verify({global::TP});
    verify({global::PP});
    verify({global::EP});
    verify({global::TP, global::EP});
}

TEST(GlobalEnvTest, ExposesDenseAndExpertViewsWithoutRenamingDenseGroups) {
    EXPECT_EQ(global::GetWorldSize(), 16);
    EXPECT_EQ(global::GetTensorParallelSize(), 2);
    EXPECT_EQ(global::GetExpertTensorParallelSize(), 2);
    EXPECT_EQ(global::GetDataParallelSize(), 4);
    EXPECT_EQ(global::GetExpertDataParallelSize(), 2);
    EXPECT_EQ(global::GetPipelineParallelSize(), 2);
    EXPECT_EQ(global::GetExpertParallelSize(), 2);

    constexpr int kRank = 7;
    EXPECT_EQ(GetDataParallelGroupRanks(kRank), (std::vector<int>{1, 3, 5, 7}));
    EXPECT_EQ(GetTensorParallelGroupRanks(kRank), (std::vector<int>{6, 7}));
    EXPECT_EQ(GetPipelineParallelGroupRanks(kRank), (std::vector<int>{7, 15}));

    // These IDs match the pre-RankGenerator dense Layout implementation.
    for (int rank = 0; rank < global::GetWorldSize(); ++rank) {
        const int tp = rank % 2;
        const int dp = (rank / 2) % 4;
        const int pp = rank / 8;
        EXPECT_EQ(GetDataParallelProcessGroupName(rank), "DP" + std::to_string(pp + tp * 2));
        EXPECT_EQ(GetTensorParallelProcessGroupName(rank), "TP" + std::to_string(pp + dp * 2));
        EXPECT_EQ(GetPipelineParallelProcessGroupName(rank), "PP" + std::to_string(dp + tp * 4));
    }

    EXPECT_EQ(GetExpertDataParallelGroupRanks(kRank), (std::vector<int>{3, 7}));
    EXPECT_EQ(GetExpertParallelGroupRanks(kRank), (std::vector<int>{5, 7}));
    EXPECT_EQ(GetExpertTensorAndExpertParallelGroupRanks(kRank), (std::vector<int>{4, 5, 6, 7}));
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
        /*expert_parallel_size=*/2);
    return RUN_ALL_TESTS();
}
