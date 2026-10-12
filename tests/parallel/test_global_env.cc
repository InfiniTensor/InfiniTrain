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

  EXPECT_EQ(dense.GroupRanks(global::kDp, 7), (std::vector<int>{1, 3, 5, 7}));
  EXPECT_EQ(dense.GroupRanks(global::kTp, 7), (std::vector<int>{6, 7}));
  EXPECT_EQ(expert.GroupRanks(global::kDp, 7), (std::vector<int>{1, 3, 5, 7}));
  EXPECT_EQ(expert.GroupRanks(global::kTp, 7), (std::vector<int>{7}));
  EXPECT_EQ(expert.GroupRanks(global::kEp, 7), (std::vector<int>{6, 7}));
  EXPECT_EQ(expert.GroupRanks({global::kTp, global::kEp}, 7),
            (std::vector<int>{6, 7}));

  EXPECT_EQ(dense.GroupId(global::kDp, 7), 2);
  EXPECT_EQ(dense.GetRanks(global::kDp).at(2), (std::vector<int>{1, 3, 5, 7}));
  EXPECT_EQ(expert.GroupId({global::kTp, global::kEp}, 7), 6);
  EXPECT_EQ(expert.GetRanks({global::kTp, global::kEp}).at(6),
            (std::vector<int>{6, 7}));
  EXPECT_EQ(dense.GetRanks(global::kPp), expert.GetRanks(global::kPp));
}

TEST(RankGeneratorTest, ContextParallelGroupsPreserveExpertPipelineView) {
  const global::RankGenerator dense(/*tp=*/2, /*ep=*/1, /*dp=*/2, /*pp=*/2,
                                    /*cp=*/2);
  const global::RankGenerator expert(/*etp=*/1, /*ep=*/4, /*edp=*/2, /*pp=*/2);
  EXPECT_EQ(dense.WorldSize(), 16);
  EXPECT_EQ(dense.GroupRanks(global::kCp, 7), (std::vector<int>{5, 7}));
  EXPECT_EQ(dense.GroupRanks(global::kDp, 7), (std::vector<int>{3, 7}));
  EXPECT_EQ(dense.GroupRanks(global::kTp, 7), (std::vector<int>{6, 7}));
  EXPECT_EQ(dense.GroupRanks({global::kDp, global::kCp}, 7),
            (std::vector<int>{1, 3, 5, 7}));
  EXPECT_EQ(dense.GroupRanks(global::kPp, 7), (std::vector<int>{7, 15}));
  EXPECT_EQ(expert.AxisSize(global::kCp), 1);
  for (int rank = 0; rank < dense.WorldSize(); ++rank) {
    EXPECT_EQ(dense.GroupRanks(global::kPp, rank),
              expert.GroupRanks(global::kPp, rank));
    int dp, tp, pp, ep, cp;
    dense.CoordOf(rank, dp, tp, pp, ep, cp);
    EXPECT_EQ(dense.RankOf(dp, tp, pp, ep, cp), rank);
    const int group_id = dense.GroupId({global::kDp, global::kCp}, rank);
    EXPECT_EQ(dense.GetRanks({global::kDp, global::kCp}).at(group_id),
              dense.GroupRanks({global::kDp, global::kCp}, rank));
  }
}

TEST(GlobalEnvTest, ExposesDenseAndExpertViewsWithoutRenamingDenseGroups) {
  // These IDs match the pre-RankGenerator dense Layout implementation.
  for (int rank = 0; rank < global::GetWorldSize(); ++rank) {
    const int tp = rank % 4;
    const int dp = (rank / 4) % 2;
    const int pp = rank / 8;
    EXPECT_EQ(GetDataParallelProcessGroupName(rank),
              "DP" + std::to_string(pp + tp * 2));
    EXPECT_EQ(GetTensorParallelProcessGroupName(rank),
              "TP" + std::to_string(pp + dp * 2));
    EXPECT_EQ(GetPipelineParallelProcessGroupName(rank),
              "PP" + std::to_string(dp + tp * 2));
  }
}

}  // namespace
