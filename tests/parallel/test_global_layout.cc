#include "infini_train/include/nn/parallel/global.h"

#include <array>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace {
using namespace infini_train::nn::parallel::global;

TEST(RankGeneratorTest, DefaultGeneratorUsesViewLocalDpAxis) {
    const RankGenerator rank_generator(/*tp=*/1, /*ep=*/1, /*dp=*/1, /*pp=*/1);

    EXPECT_EQ(rank_generator.AxisSize(DP), 1);
    EXPECT_EQ(rank_generator.AxisSize(TP), 1);
    EXPECT_EQ(rank_generator.AxisSize(PP), 1);
    EXPECT_EQ(rank_generator.AxisSize(EP), 1);
    EXPECT_EQ(rank_generator.order(), (std::array<Axis, AXIS_COUNT>{TP, EP, DP, PP}));
    EXPECT_EQ(rank_generator.WorldSize(), 1);
}

TEST(RankGeneratorTest, ExpertGroupsMatchMegatronOrthogonalOrdering) {
    const RankGenerator expert(/*tp=*/2, /*ep=*/2, /*dp=*/2, /*pp=*/2);

    EXPECT_EQ(expert.GetRanks(DP),
              (std::vector<std::vector<int>>{{0, 4}, {1, 5}, {2, 6}, {3, 7}, {8, 12}, {9, 13}, {10, 14}, {11, 15}}));
    EXPECT_EQ(expert.GetRanks(EP),
              (std::vector<std::vector<int>>{{0, 2}, {1, 3}, {4, 6}, {5, 7}, {8, 10}, {9, 11}, {12, 14}, {13, 15}}));
    EXPECT_EQ(expert.GetRanks({TP, EP}),
              (std::vector<std::vector<int>>{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}}));
}

TEST(RankGeneratorTest, AsymmetricGroupsMatchMegatronOrthogonalOrdering) {
    const RankGenerator rank_generator(/*tp=*/2, /*ep=*/3, /*dp=*/2, /*pp=*/1);

    EXPECT_EQ(rank_generator.GetRanks(DP),
              (std::vector<std::vector<int>>{{0, 6}, {1, 7}, {2, 8}, {3, 9}, {4, 10}, {5, 11}}));
    EXPECT_EQ(rank_generator.GetRanks(EP),
              (std::vector<std::vector<int>>{{0, 2, 4}, {1, 3, 5}, {6, 8, 10}, {7, 9, 11}}));
    EXPECT_EQ(rank_generator.GetRanks({TP, EP}),
              (std::vector<std::vector<int>>{{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9, 10, 11}}));
    EXPECT_EQ(rank_generator.GroupId(DP, 11), 5);
    EXPECT_EQ(rank_generator.GroupId(EP, 11), 3);
    EXPECT_EQ(rank_generator.GroupId({TP, EP}, 11), 1);
}

TEST(RankGeneratorTest, DenseAndExpertViewsShareTheSamePhysicalRanks) {
    const RankGenerator dense(/*tp=*/2, /*ep=*/1, /*dp=*/4, /*pp=*/2);
    const RankGenerator expert(/*tp=*/2, /*ep=*/2, /*dp=*/2, /*pp=*/2);

    ASSERT_EQ(dense.WorldSize(), 16);
    ASSERT_EQ(expert.WorldSize(), dense.WorldSize());

    constexpr int kRank = 7;
    EXPECT_EQ(dense.GroupRanks(DP, kRank), (std::vector<int>{1, 3, 5, 7}));
    EXPECT_EQ(expert.GroupRanks(DP, kRank), (std::vector<int>{3, 7}));
    EXPECT_EQ(expert.GroupRanks(EP, kRank), (std::vector<int>{5, 7}));
    EXPECT_EQ(expert.GroupRanks({TP, EP}, kRank), (std::vector<int>{4, 5, 6, 7}));

    // Dense DP and expert DPxEP cover the same physical ranks, but are
    // intentionally generated from different logical views.
    EXPECT_EQ(dense.GetRanks(DP), expert.GetRanks({DP, EP}));
    EXPECT_EQ(dense.GetRanks(TP), expert.GetRanks(TP));
    EXPECT_EQ(dense.GetRanks(PP), expert.GetRanks(PP));
}

TEST(RankGeneratorTest, DenseAndExpertViewsUseIndependentTensorParallelSizes) {
    const RankGenerator dense(/*tp=*/3, /*ep=*/1, /*dp=*/8, /*pp=*/2);
    const RankGenerator expert(/*etp=*/2, /*ep=*/2, /*edp=*/6, /*pp=*/2);

    ASSERT_EQ(dense.WorldSize(), 48);
    ASSERT_EQ(expert.WorldSize(), dense.WorldSize());
    EXPECT_EQ(dense.AxisSize(TP), 3);
    EXPECT_EQ(expert.AxisSize(TP), 2);
    EXPECT_EQ(dense.AxisSize(DP), 8);
    EXPECT_EQ(expert.AxisSize(DP), 6);
    EXPECT_EQ(dense.GetRanks(PP), expert.GetRanks(PP));

    constexpr int kRank = 11;
    EXPECT_EQ(dense.GroupRanks(TP, kRank), (std::vector<int>{9, 10, 11}));
    EXPECT_EQ(expert.GroupRanks(TP, kRank), (std::vector<int>{10, 11}));
    EXPECT_EQ(expert.GroupRanks(EP, kRank), (std::vector<int>{9, 11}));
    EXPECT_EQ(expert.GroupRanks({TP, EP}, kRank), (std::vector<int>{8, 9, 10, 11}));
    EXPECT_EQ(expert.GroupRanks(DP, kRank), (std::vector<int>{3, 7, 11, 15, 19, 23}));
}

TEST(RankGeneratorTest, DenseAndExpertViewsAreIdenticalWhenEpIsOne) {
    const RankGenerator dense(/*tp=*/2, /*ep=*/1, /*dp=*/4, /*pp=*/2);
    const RankGenerator expert(/*tp=*/2, /*ep=*/1, /*dp=*/4, /*pp=*/2);

    EXPECT_EQ(dense.GetRanks(DP), expert.GetRanks(DP));
    EXPECT_EQ(dense.GetRanks(TP), expert.GetRanks(TP));
    EXPECT_EQ(dense.GetRanks(PP), expert.GetRanks(PP));
    EXPECT_EQ(dense.GetRanks(EP), expert.GetRanks(EP));
}

TEST(RankGeneratorTest, EdpOneDoesNotCollapseTheDenseDpView) {
    const RankGenerator dense(/*tp=*/2, /*ep=*/1, /*dp=*/2, /*pp=*/2);
    const RankGenerator expert(/*tp=*/2, /*ep=*/2, /*dp=*/1, /*pp=*/2);

    constexpr int kRank = 7;
    EXPECT_EQ(dense.GroupRanks(DP, kRank), (std::vector<int>{5, 7}));
    EXPECT_EQ(expert.GroupRanks(DP, kRank), (std::vector<int>{7}));
    EXPECT_EQ(expert.GroupRanks(EP, kRank), (std::vector<int>{5, 7}));
    EXPECT_EQ(dense.GroupRanks(DP, kRank), expert.GroupRanks({DP, EP}, kRank));
}

TEST(RankGeneratorTest, GroupIdsAreIndicesIntoGeneratedRankGroups) {
    const RankGenerator expert(/*tp=*/2, /*ep=*/2, /*dp=*/2, /*pp=*/2);

    const auto edp_groups = expert.GetRanks(DP);
    for (size_t group_id = 0; group_id < edp_groups.size(); ++group_id) {
        for (const int rank : edp_groups[group_id]) {
            EXPECT_EQ(expert.GroupId(DP, rank), group_id);
            EXPECT_EQ(expert.GroupRanks(DP, rank), edp_groups[group_id]);
        }
    }

    const auto tensor_expert_groups = expert.GetRanks({TP, EP});
    for (size_t group_id = 0; group_id < tensor_expert_groups.size(); ++group_id) {
        for (const int rank : tensor_expert_groups[group_id]) {
            EXPECT_EQ(expert.GroupId({TP, EP}, rank), group_id);
            EXPECT_EQ(expert.GroupRanks({TP, EP}, rank), tensor_expert_groups[group_id]);
        }
    }
}

TEST(RankGeneratorTest, AsymmetricShapeCoversEveryRankForAllAxisSubsets) {
    const RankGenerator rank_generator(/*tp=*/2, /*ep=*/3, /*dp=*/4, /*pp=*/5);

    const auto verify_groups = [&rank_generator](std::initializer_list<Axis> varying_axes) {
        std::array<bool, AXIS_COUNT> varying_axis_mask{};
        for (const Axis axis : varying_axes) { varying_axis_mask[axis] = true; }

        int expected_group_size = 1;
        for (int axis = 0; axis < AXIS_COUNT; ++axis) {
            if (varying_axis_mask[axis]) {
                expected_group_size *= rank_generator.AxisSize(static_cast<Axis>(axis));
            }
        }

        const auto groups = rank_generator.GetRanks(varying_axes);
        ASSERT_EQ(groups.size(), rank_generator.WorldSize() / expected_group_size);
        std::vector<int> rank_occurrences(rank_generator.WorldSize(), 0);
        for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
            ASSERT_EQ(groups[group_id].size(), expected_group_size);
            for (const int rank : groups[group_id]) {
                ASSERT_GE(rank, 0);
                ASSERT_LT(rank, rank_generator.WorldSize());
                ++rank_occurrences[rank];
                EXPECT_EQ(rank_generator.GroupId(varying_axes, rank), static_cast<int>(group_id));
                EXPECT_EQ(rank_generator.GroupRanks(varying_axes, rank), groups[group_id]);
            }
        }
        for (const int occurrence_count : rank_occurrences) { EXPECT_EQ(occurrence_count, 1); }
    };

    verify_groups({DP});
    verify_groups({TP});
    verify_groups({PP});
    verify_groups({EP});
    verify_groups({DP, TP});
    verify_groups({DP, PP});
    verify_groups({DP, EP});
    verify_groups({TP, PP});
    verify_groups({TP, EP});
    verify_groups({PP, EP});
    verify_groups({DP, TP, PP});
    verify_groups({DP, TP, EP});
    verify_groups({DP, PP, EP});
    verify_groups({TP, PP, EP});
    verify_groups({DP, TP, PP, EP});
}

TEST(RankGeneratorTest, CompositeAxisOrderDoesNotAffectGroups) {
    const RankGenerator expert(/*tp=*/2, /*ep=*/2, /*dp=*/2, /*pp=*/1);

    EXPECT_EQ(expert.GetRanks({DP, EP}), expert.GetRanks({EP, DP}));
    EXPECT_EQ(expert.GetRanks({DP, EP, DP}), expert.GetRanks({DP, EP}));
    EXPECT_EQ(expert.GroupId({DP, EP}, 7), expert.GroupId({EP, DP}, 7));
}

TEST(RankGeneratorTest, CompositeGroupsSupportNonDefaultRankOrder) {
    const RankGenerator rank_generator(
        /*tp=*/2, /*ep=*/2, /*dp=*/2, /*pp=*/1, std::array<Axis, AXIS_COUNT>{EP, TP, DP, PP});

    EXPECT_EQ(rank_generator.GroupRanks({DP, EP}, 7), (std::vector<int>{2, 3, 6, 7}));
    EXPECT_EQ(rank_generator.GroupRanks({TP, EP}, 7), (std::vector<int>{4, 5, 6, 7}));
}

TEST(RankGeneratorTest, ProcessGroupOverviewReportsOnlyTheDenseViewByDefault) {
    const RankGenerator dense(/*tp=*/2, /*ep=*/1, /*dp=*/2, /*pp=*/1);

    const std::string overview = ProcessGroupOverview(dense);
    EXPECT_NE(overview.find("config: {DP=2, TP=2, PP=1}"), std::string::npos);
    EXPECT_NE(overview.find("[Dense Rank View] shape={TP=2, DP=2, PP=1}"), std::string::npos);
    EXPECT_NE(overview.find("order={ TP -> DP -> PP }"), std::string::npos);
    EXPECT_NE(overview.find("[DP] size=2, num_groups=2"), std::string::npos);
    EXPECT_NE(overview.find("[TP] size=2, num_groups=2"), std::string::npos);
    EXPECT_EQ(overview.find("[Expert Rank View]"), std::string::npos);
    EXPECT_EQ(overview.find("[EDP]"), std::string::npos);
    EXPECT_EQ(overview.find("[ETP]"), std::string::npos);
    EXPECT_EQ(overview.find("[EP]"), std::string::npos);
    EXPECT_EQ(overview.find("[ETP_EP]"), std::string::npos);
}

TEST(RankGeneratorTest, ProcessGroupOverviewSeparatesDenseAndExpertViews) {
    const RankGenerator dense(/*tp=*/2, /*ep=*/1, /*dp=*/4, /*pp=*/2);
    const RankGenerator expert(/*etp=*/1, /*ep=*/2, /*edp=*/4, /*pp=*/2);

    const std::string overview = ProcessGroupOverview(dense, expert);
    EXPECT_NE(overview.find("config: {DP=4, EDP=4, TP=2, ETP=1, PP=2, EP=2}"), std::string::npos);
    EXPECT_NE(overview.find("[Dense Rank View] shape={TP=2, DP=4, PP=2}"), std::string::npos);
    EXPECT_NE(overview.find("order={ TP -> DP -> PP }"), std::string::npos);
    EXPECT_NE(overview.find("[Expert Rank View] shape={ETP=1, EP=2, EDP=4, PP=2}"), std::string::npos);
    EXPECT_NE(overview.find("order={ ETP -> EP -> EDP -> PP }"), std::string::npos);
    EXPECT_NE(overview.find("[DP] size=4, num_groups=4"), std::string::npos);
    EXPECT_NE(overview.find("[EDP] size=4, num_groups=4"), std::string::npos);
    EXPECT_NE(overview.find("[ETP] size=1, unenabled"), std::string::npos);
    EXPECT_NE(overview.find("[EP] size=2, num_groups=8"), std::string::npos);
    EXPECT_NE(overview.find("[ETP_EP] size=2, num_groups=8"), std::string::npos);
}

TEST(RankGeneratorTest, ProcessGroupOverviewReportsTrivialGroupsAsUnenabled) {
    const RankGenerator dense(/*tp=*/1, /*ep=*/1, /*dp=*/1, /*pp=*/1);
    const RankGenerator expert(/*tp=*/1, /*ep=*/1, /*dp=*/1, /*pp=*/1);

    const std::string overview = ProcessGroupOverview(dense, expert);
    EXPECT_NE(overview.find("config: {DP=1, EDP=1, TP=1, ETP=1, PP=1, EP=1}"), std::string::npos);
    EXPECT_NE(overview.find("[DP] size=1, unenabled"), std::string::npos);
    EXPECT_NE(overview.find("[EDP] size=1, unenabled"), std::string::npos);
    EXPECT_NE(overview.find("[ETP] size=1, unenabled"), std::string::npos);
    EXPECT_NE(overview.find("[EP] size=1, unenabled"), std::string::npos);
    EXPECT_NE(overview.find("[ETP_EP] size=1, unenabled"), std::string::npos);
}

} // namespace
