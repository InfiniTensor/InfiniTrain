#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/pipeline_layout.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

using namespace infini_train;

namespace {

nn::TransformerConfig MakeConfig() {
    nn::TransformerConfig config;
    config.n_layer = 12;
    config.n_head = 4;
    config.n_kv_head = 4;
    config.n_embd = 32;
    config.vocab_size = 64;
    config.original_vocab_size = 64;
    config.position_embedding_type = nn::PositionEmbeddingType::kLearnedAbsolute;
    config.activation_type = nn::MLPType::kGELU;
    config.norm_type = nn::NormType::kLayerNorm;
    config.add_bias_linear = true;
    config.tie_weights = false;
    return config;
}

std::shared_ptr<nn::parallel::PipelineParallel> BuildPipeline(const nn::parallel::PipelineLayout &layout,
                                                              int stage_id) {
    nn::parallel::global::GlobalEnv::Instance().set_pipeline_layout(layout);
    nn::parallel::pp_rank = stage_id;

    auto model = std::make_shared<nn::TransformerModel>(MakeConfig());
    const std::vector<std::vector<int64_t>> recv_shape = {{2, 4, 32}};
    const int local_chunk_count = static_cast<int>(layout.stage(stage_id).global_chunk_ids.size());

    return std::make_shared<nn::parallel::PipelineParallel>(model, layout.num_stages(), /*num_micro_batches=*/1,
                                                            recv_shape, stage_id, Device(Device::DeviceType::kCPU, 0),
                                                            local_chunk_count);
}

std::shared_ptr<nn::Sequential> AsSequential(const std::shared_ptr<nn::parallel::PipelineParallel> &pipeline,
                                             size_t index) {
    auto *chunks = pipeline->mutable_chunks();
    if (chunks == nullptr || index >= chunks->size()) {
        return nullptr;
    }
    return std::dynamic_pointer_cast<nn::Sequential>(chunks->at(index));
}

} // namespace

TEST(PipelineParallelChunkingTest, Stage0IncludesEmbeddingOnly) {
    const auto layout = nn::parallel::PipelineLayout::BuildContiguous({2, 4, 3, 3});
    auto pipeline = BuildPipeline(layout, 0);

    ASSERT_EQ(pipeline->mutable_chunks()->size(), 1U);
    auto chunk = AsSequential(pipeline, 0);
    ASSERT_NE(chunk, nullptr);
    EXPECT_EQ(chunk->module("0").type(), nn::TransformerFirstStage::kType);
    EXPECT_EQ(chunk->module("1").type(), nn::TransformerChunk::kType);
}

TEST(PipelineParallelChunkingTest, IntermediateStageIncludesOnlyTransformerChunk) {
    const auto layout = nn::parallel::PipelineLayout::BuildContiguous({2, 4, 3, 3});
    auto pipeline = BuildPipeline(layout, 1);

    ASSERT_EQ(pipeline->mutable_chunks()->size(), 1U);
    auto chunk = AsSequential(pipeline, 0);
    ASSERT_NE(chunk, nullptr);
    EXPECT_EQ(chunk->module("0").type(), nn::TransformerChunk::kType);
}

TEST(PipelineParallelChunkingTest, LastStageIncludesTransformerLastStage) {
    const auto layout = nn::parallel::PipelineLayout::BuildContiguous({2, 4, 3, 3});
    auto pipeline = BuildPipeline(layout, 3);

    ASSERT_EQ(pipeline->mutable_chunks()->size(), 1U);
    auto chunk = AsSequential(pipeline, 0);
    ASSERT_NE(chunk, nullptr);
    EXPECT_EQ(chunk->module("0").type(), nn::TransformerChunk::kType);
    EXPECT_EQ(chunk->module("1").type(), nn::TransformerLastStage::kType);
}

TEST(PipelineParallelChunkingTest, VirtualPipelinePreservesLocalChunkOrder) {
    const auto layout = nn::parallel::PipelineLayout::BuildDefault(12, 2, 2);
    auto pipeline = BuildPipeline(layout, 0);

    ASSERT_EQ(pipeline->mutable_chunks()->size(), 2U);
    auto first = AsSequential(pipeline, 0);
    auto second = AsSequential(pipeline, 1);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->module("0").type(), nn::TransformerFirstStage::kType);
    EXPECT_EQ(first->module("1").type(), nn::TransformerChunk::kType);
    EXPECT_EQ(second->module("0").type(), nn::TransformerChunk::kType);
}

TEST(PipelineParallelChunkingTest, RejectsStageCountMismatchWithCatchableError) {
    const auto layout = nn::parallel::PipelineLayout::BuildContiguous({2, 4, 3, 3});
    nn::parallel::global::GlobalEnv::Instance().set_pipeline_layout(layout);
    nn::parallel::pp_rank = 0;
    auto model = std::make_shared<nn::TransformerModel>(MakeConfig());

    EXPECT_THROW(std::make_shared<nn::parallel::PipelineParallel>(
                     model, /*num_stages=*/2, /*num_micro_batches=*/1, std::vector<std::vector<int64_t>>{{2, 4, 32}},
                     /*rank=*/0, Device(Device::DeviceType::kCPU, 0), /*vpp=*/1),
                 nn::parallel::PipelineLayoutError);
}

class PipelineParallelChunkingEnvironment : public ::testing::Environment {
public:
    void TearDown() override {
        nn::parallel::pp_rank = 0;
        nn::parallel::global::GlobalEnv::Instance().set_pipeline_layout(
            nn::parallel::PipelineLayout::BuildDefault(1, 1, 1));
    }
};

::testing::Environment *const pipeline_parallel_chunking_environment
    = ::testing::AddGlobalTestEnvironment(new PipelineParallelChunkingEnvironment());
