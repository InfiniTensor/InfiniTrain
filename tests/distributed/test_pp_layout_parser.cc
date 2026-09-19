#include "gtest/gtest.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "example/common/parser.h"

namespace infini_train::examples {
namespace {
using nn::parallel::LayerIndex;
using nn::parallel::PipelineChunkSpec;

TEST(PipelineLayoutCLITest, ResolvesDefaultAndInterleavedRequests) {
    auto request = ParsePipelineLayoutRequest("", "", 2, 2);
    EXPECT_TRUE(std::holds_alternative<std::monostate>(request));
    EXPECT_FALSE(IsExplicitChunkRequest(request));
    const auto uniform = ResolvePipelineLayout(13, 2, 2, request);
    ASSERT_NE(uniform, nullptr);
    EXPECT_FALSE(uniform->IsCustom());
    EXPECT_EQ(uniform->GetNumChunks(), 4);
    EXPECT_EQ(uniform->GetChunk(0).layer_range.end, 4);

    request = ParsePipelineLayoutRequest("02,4,3,3", "", 2, 2);
    EXPECT_EQ(std::get<std::vector<LayerIndex>>(request), (std::vector<LayerIndex>{2, 4, 3, 3}));
    EXPECT_FALSE(IsExplicitChunkRequest(request));
    const auto custom = ResolvePipelineLayout(12, 2, 2, request);
    ASSERT_NE(custom, nullptr);
    EXPECT_TRUE(custom->IsCustom());
    EXPECT_EQ(custom->GetChunk(2).stage_id, 0);
    EXPECT_EQ(custom->GetChunk(2).local_chunk_idx, 1);
    EXPECT_EQ(custom->GetChunk(2).layer_range.begin, 6);
    EXPECT_EQ(custom->GetChunk(2).layer_range.end, 9);
    EXPECT_THROW((void)ResolvePipelineLayout(13, 2, 2, request), std::invalid_argument);
}

TEST(PipelineLayoutCLITest, PreservesExplicitOwnerOrderAndValidatesActualModelSize) {
    const auto request = ParsePipelineLayoutRequest("", "1:2,0:3,1:7", 2, 2);
    ASSERT_TRUE(IsExplicitChunkRequest(request));
    const auto &specs = std::get<std::vector<PipelineChunkSpec>>(request);
    ASSERT_EQ(specs.size(), 3);
    EXPECT_EQ(specs[0].stage_id, 1);
    EXPECT_EQ(specs[0].layer_count, 2);
    EXPECT_EQ(specs[1].stage_id, 0);
    EXPECT_EQ(specs[2].layer_count, 7);
    const auto layout = ResolvePipelineLayout(12, 2, 2, request);
    ASSERT_NE(layout, nullptr);
    EXPECT_EQ(layout->GetNumChunks(), 3); // Not PP * vPP.
    EXPECT_EQ(layout->GetMaxLocalChunks(), 2);
    EXPECT_EQ(layout->GetInputStage(), 1);
    EXPECT_EQ(layout->GetOutputStage(), 1);
    EXPECT_EQ(layout->GetChunk(2).layer_range.begin, 5);
    EXPECT_THROW((void)ResolvePipelineLayout(13, 2, 2, request), std::invalid_argument);
    EXPECT_THROW((void)ResolvePipelineLayout(12, 2, 1, request), std::invalid_argument);
}

TEST(PipelineLayoutCLITest, FallsBackOnlyForUnderfilledDefaults) {
    const auto request = ParsePipelineLayoutRequest("", "", 2, 2);
    EXPECT_EQ(ResolvePipelineLayout(0, 2, 2, request), nullptr);
    EXPECT_EQ(ResolvePipelineLayout(3, 2, 2, request), nullptr);
    EXPECT_NE(ResolvePipelineLayout(4, 2, 2, request), nullptr);
    EXPECT_THROW((void)ResolvePipelineLayout(-1, 2, 2, request), std::invalid_argument);
    EXPECT_THROW((void)ResolvePipelineLayout(0, 2, 2, PipelineLayoutRequest{std::vector<LayerIndex>{}}),
                 std::invalid_argument);
    EXPECT_THROW((void)ResolvePipelineLayout(0, 2, 2, PipelineLayoutRequest{std::vector<PipelineChunkSpec>{}}),
                 std::invalid_argument);
    const PipelineLayoutRequest explicit_request = std::vector<PipelineChunkSpec>{{0, 2}, {1, 2}};
    EXPECT_THROW((void)ResolvePipelineLayout(4, 0, 1, explicit_request), std::invalid_argument);
    EXPECT_THROW((void)ResolvePipelineLayout(4, 2, 0, explicit_request), std::invalid_argument);
    EXPECT_THROW((void)ResolvePipelineLayout(4, 2, std::numeric_limits<int>::max(), explicit_request),
                 std::invalid_argument);
}

TEST(PipelineLayoutCLITest, RejectsMalformedAndConflictingCLI) {
    // One representative per syntax/semantic error class, rather than a Cartesian product.
    for (const auto *value : {"1,", ",1", "1,,2", "1, 2", "+1,2", "-1,2", "0,2", "1.5,2", "9223372036854775808,1"}) {
        SCOPED_TRACE(value);
        EXPECT_THROW((void)ParsePipelineLayerPartition(value), std::invalid_argument);
    }
    for (const auto *value : {"0:6,1:6,", "0:6:1,1:6", "0:,1:12", "0:6, 1:6", "0:0,1:12", "0:6,2:6", "0:6,0:6",
                              "-1:6,1:6", "2147483648:6,1:6", "0:9223372036854775808,1:1"}) {
        SCOPED_TRACE(value);
        EXPECT_THROW((void)ParsePipelineLayoutRequest("", value, 2, 1), std::invalid_argument);
    }
    EXPECT_THROW((void)ParsePipelineLayoutRequest("6,6", "0:6,1:6", 2, 1), std::invalid_argument);
    EXPECT_THROW((void)ParsePipelineLayoutRequest("6,6", "", 2, 2), std::invalid_argument);
    EXPECT_THROW((void)ParsePipelineLayoutRequest("", "0:3,1:3,0:6", 2, 1), std::invalid_argument);
    EXPECT_THROW((void)ParsePipelineLayoutRequest("", "", 0, 1), std::invalid_argument);
    EXPECT_THROW((void)ParsePipelineLayoutRequest("", "", 2, 0), std::invalid_argument);
    EXPECT_THROW((void)ParsePipelineLayoutRequest("", "", 2, std::numeric_limits<int>::max()), std::invalid_argument);
    try {
        (void)ParsePipelineLayerPartition("6,x");
        FAIL() << "Malformed count must be rejected";
    } catch (const std::invalid_argument &error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("pipeline_layer_partition"), std::string::npos);
        EXPECT_NE(message.find("6,x"), std::string::npos);
        EXPECT_NE(message.find("item 1"), std::string::npos);
    }
}

TEST(PipelineLayoutCLITest, FormatsActualChunkIdentityAndEndpoints) {
    const auto layout = ResolvePipelineLayout(12, 2, 2, ParsePipelineLayoutRequest("", "1:2,0:3,1:7", 2, 2));
    const auto formatted = FormatPipelineLayout(*layout);
    EXPECT_NE(formatted.find("custom pipeline layout:"), std::string::npos);
    EXPECT_NE(formatted.find("Stage 0: chunk 1 (local 0) layers [2,5)"), std::string::npos);
    EXPECT_NE(
        formatted.find(
            "Stage 1: chunk 0 (local 0) layers [0,2) chunk 2 (local 1) layers [5,12) embedding final_norm lm_head"),
        std::string::npos);
}
} // namespace
} // namespace infini_train::examples
