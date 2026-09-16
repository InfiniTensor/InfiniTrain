#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/parallel/pp/pipeline_layout.h"

namespace infini_train::nn::parallel {
namespace {

TEST(PipelineLayoutSuggestTest, UniformCostsProduceBalancedCounts) {
    const std::vector<int> expected{6, 6};
    EXPECT_EQ(SuggestBalancedPartition(12, 2, {}), expected);
}

TEST(PipelineLayoutSuggestTest, UniformCostsWithRemainder) {
    const std::vector<int> expected{3, 3, 4};
    EXPECT_EQ(SuggestBalancedPartition(10, 3, {}), expected);
}

TEST(PipelineLayoutSuggestTest, CostImbalanceShiftsLayersToLightStage) {
    // Four "light" layers (cost 1) followed by eight "heavy" layers (cost 2). Balancing
    // total cost (20 / 2 = 10 per stage) yields {7, 5} instead of the uniform {6, 6}.
    std::vector<double> costs(12, 2.0);
    for (int i = 0; i < 4; ++i) { costs[i] = 1.0; }
    const std::vector<int> expected{7, 5};
    EXPECT_EQ(SuggestBalancedPartition(12, 2, costs), expected);
}

TEST(PipelineLayoutSuggestTest, PartitionSumsToTotalLayers) {
    const std::vector<double> costs{3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    auto partition = SuggestBalancedPartition(8, 3, costs);
    ASSERT_EQ(partition.size(), 3u);
    int sum = 0;
    for (int count : partition) {
        EXPECT_GT(count, 0);
        sum += count;
    }
    EXPECT_EQ(sum, 8);
}

TEST(PipelineLayoutSuggestTest, SingleStageTakesAllLayers) {
    EXPECT_EQ(SuggestBalancedPartition(12, 1, {}), std::vector<int>{12});
}

TEST(PipelineLayoutSuggestTest, RejectsNegativeCost) {
    EXPECT_DEATH(SuggestBalancedPartition(4, 2, {-1.0, 1.0, 1.0, 1.0}), "non-negative");
}

TEST(PipelineLayoutSuggestTest, RejectsWrongCostCount) {
    EXPECT_DEATH(SuggestBalancedPartition(4, 2, {1.0, 2.0}), "entries");
}

TEST(PipelineLayoutSuggestTest, RejectsFewerLayersThanStages) {
    EXPECT_DEATH(SuggestBalancedPartition(2, 3, {}), "fewer layers than stages");
}

TEST(PipelineLayerCostsTest, ParsesValidCosts) {
    const std::vector<double> expected{1.0, 2.0, 1.5};
    EXPECT_EQ(ParsePipelineLayerCosts("1.0,2.0,1.5"), expected);
}

TEST(PipelineLayerCostsTest, EmptyStringGivesNoCosts) { EXPECT_TRUE(ParsePipelineLayerCosts("").empty()); }

TEST(PipelineLayerCostsTest, RejectsNegativeCost) { EXPECT_DEATH(ParsePipelineLayerCosts("-1,2"), "non-negative"); }

TEST(PipelineLayerCostsTest, RejectsNonNumber) { EXPECT_DEATH(ParsePipelineLayerCosts("1,abc"), "not a number"); }

TEST(PipelineLayerCostsTest, RejectsEmptyEntry) { EXPECT_DEATH(ParsePipelineLayerCosts("1,,2"), "empty entry"); }

TEST(PipelineLayerCostsTest, RejectsInfinity) { EXPECT_DEATH(ParsePipelineLayerCosts("inf"), "finite"); }

TEST(ComputePerLayerParamCountsTest, MatchesAnalyticGELULayerNorm) {
    nn::TransformerConfig config{
        .block_size = 64,
        .vocab_size = 128,
        .n_layer = 4,
        .n_head = 2,
        .n_kv_head = 2,
        .n_embd = 16,
        .position_embedding_type = nn::PositionEmbeddingType::kLearnedAbsolute,
        .activation_type = nn::MLPType::kGELU,
        .ffn_type = nn::FFNType::kDense,
        .norm_type = nn::NormType::kLayerNorm,
        .add_bias_linear = true,
        .ffn_expansion_ratio = 4.0f,
        .ffn_dim_multiplier = std::nullopt,
        .multiple_of = 1,
    };
    const std::vector<double> expected(4, 3280.0);
    EXPECT_EQ(nn::ComputePerLayerParamCounts(config), expected);
}

TEST(ComputePerLayerParamCountsTest, SwigluRMSNormYieldsPositiveUniformCounts) {
    nn::TransformerConfig config{
        .block_size = 64,
        .vocab_size = 128,
        .n_layer = 3,
        .n_head = 2,
        .n_kv_head = 2,
        .n_embd = 16,
        .position_embedding_type = nn::PositionEmbeddingType::kRoPE,
        .activation_type = nn::MLPType::kSwiGLU,
        .ffn_type = nn::FFNType::kDense,
        .norm_type = nn::NormType::kRMSNorm,
        .add_bias_linear = false,
        .ffn_expansion_ratio = 4.0f,
        .ffn_dim_multiplier = std::nullopt,
        .multiple_of = 1,
    };
    auto counts = nn::ComputePerLayerParamCounts(config);
    ASSERT_EQ(counts.size(), 3u);
    for (double c : counts) { EXPECT_GT(c, 0.0); }
    EXPECT_EQ(counts[0], counts[1]);
    EXPECT_EQ(counts[1], counts[2]);
}

TEST(ComputePerLayerParamCountsTest, RejectsMoE) {
    nn::TransformerConfig config{.n_layer = 4, .n_embd = 16, .ffn_type = nn::FFNType::kMoE};
    EXPECT_DEATH(nn::ComputePerLayerParamCounts(config), "MoE");
}

TEST(PipelineLoadAnalysisTest, UniformCostsArePerfectlyBalanced) {
    // 12 layers / 3 stages with unit costs: uniform {4,4,4} -> every stage load == 4.
    auto stats = ComputePipelineLoadAnalysis(12, 3, {4, 4, 4}, {}, /*num_micro_batches=*/8);
    ASSERT_EQ(stats.stage_loads.size(), 3u);
    for (double load : stats.stage_loads) { EXPECT_DOUBLE_EQ(load, 4.0); }
    EXPECT_DOUBLE_EQ(stats.bottleneck, 4.0);
    EXPECT_DOUBLE_EQ(stats.average, 4.0);
    EXPECT_DOUBLE_EQ(stats.imbalance_bubble, 0.0);
    EXPECT_DOUBLE_EQ(stats.efficiency, 1.0);
    EXPECT_DOUBLE_EQ(stats.structural_bubble, 2.0 / (2.0 + 8.0));
}

TEST(PipelineLoadAnalysisTest, ImbalancedCostsMakeUniformLayoutSkewed) {
    // 4 light layers (cost 1) + 8 heavy layers (cost 2). Uniform {6,6} assigns
    // stage 0: 4*1 + 2*2 = 8, stage 1: 6*2 = 12.
    std::vector<double> costs(12, 2.0);
    for (int i = 0; i < 4; ++i) { costs[i] = 1.0; }
    auto stats = ComputePipelineLoadAnalysis(12, 2, {6, 6}, costs, 8);
    EXPECT_DOUBLE_EQ(stats.stage_loads[0], 8.0);
    EXPECT_DOUBLE_EQ(stats.stage_loads[1], 12.0);
    EXPECT_DOUBLE_EQ(stats.bottleneck, 12.0);
    EXPECT_DOUBLE_EQ(stats.average, 10.0);
    EXPECT_DOUBLE_EQ(stats.efficiency, 10.0 / 12.0);
    EXPECT_DOUBLE_EQ(stats.imbalance_bubble, 1.0 - 10.0 / 12.0);
}

TEST(PipelineLoadAnalysisTest, BalancedPartitionRemovesImbalanceBubble) {
    // Same costs, but the cost-balanced partition {7,5} yields load 10 / 10.
    std::vector<double> costs(12, 2.0);
    for (int i = 0; i < 4; ++i) { costs[i] = 1.0; }
    auto stats = ComputePipelineLoadAnalysis(12, 2, {7, 5}, costs, 8);
    EXPECT_DOUBLE_EQ(stats.stage_loads[0], 10.0);
    EXPECT_DOUBLE_EQ(stats.stage_loads[1], 10.0);
    EXPECT_DOUBLE_EQ(stats.imbalance_bubble, 0.0);
    EXPECT_DOUBLE_EQ(stats.efficiency, 1.0);
}

TEST(PipelineLoadAnalysisTest, EmptyPartitionDefaultsToUniform) {
    auto stats = ComputePipelineLoadAnalysis(12, 3, {}, {}, 1);
    ASSERT_EQ(stats.stage_loads.size(), 3u);
    for (double load : stats.stage_loads) { EXPECT_DOUBLE_EQ(load, 4.0); }
}

TEST(PipelineLoadAnalysisTest, StructuralBubbleFollowsGpipeFormula) {
    // Two stages, one micro-batch: (S-1)/(S-1+n) = 1/2.
    auto stats = ComputePipelineLoadAnalysis(4, 2, {2, 2}, {}, 1);
    EXPECT_DOUBLE_EQ(stats.structural_bubble, 0.5);
}

} // namespace
} // namespace infini_train::nn::parallel
