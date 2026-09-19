#include "gtest/gtest.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_OMP
#include <omp.h>
#endif

#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
namespace {

TransformerConfig SmallConfig(bool llama = false) {
    TransformerConfig config;
    config.n_layer = 6;
    config.n_embd = 8;
    config.n_head = 2;
    config.n_kv_head = 2;
    config.vocab_size = config.original_vocab_size = 16;
    config.block_size = 8;
    config.multiple_of = 8;
    if (llama) {
        config.position_embedding_type = PositionEmbeddingType::kRoPE;
        config.norm_type = NormType::kRMSNorm;
        config.activation_type = MLPType::kSwiGLU;
        config.add_bias_linear = false;
        config.tie_weights = false;
    }
    return config;
}

class PipelineParallelTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef USE_OMP
        threads_ = omp_get_max_threads();
        omp_set_num_threads(1);
#endif
    }
    void TearDown() override {
#ifdef USE_OMP
        omp_set_num_threads(threads_);
#endif
    }
    const std::vector<std::vector<int64_t>> shapes_{{2, 4, 8}};
#ifdef USE_OMP
    int threads_;
#endif
};

// Verify the real model's layers and the wrapper's pointer identity/order. No GPU
// communication is needed to catch wrong stage initialization or copied modules.
TEST_F(PipelineParallelTest, BuildsAndWrapsActualModelChunks) {
    struct Case {
        const char *name;
        PipelineLayout layout;
        bool explicit_mode;
        std::vector<std::vector<int>> chunk_sizes;
        int input_stage;
        int output_stage;
    };
    const std::vector<LayerIndex> two_stage_counts{1, 2, 1, 2};
    const std::vector<LayerIndex> three_stage_counts{1, 2, 3};
    const std::vector<LayerIndex> three_stage_vpp_counts{1, 2, 1, 1, 1, 3};
    const std::vector<PipelineChunkSpec> two_stage_specs{{1, 1}, {0, 2}, {1, 1}, {1, 2}};
    const std::vector<PipelineChunkSpec> three_stage_specs{{1, 1}, {0, 1}, {2, 1}, {1, 1}, {2, 2}};
    // Literal expectations keep the model/wrapper assertions independent of layout queries.
    const std::vector<Case> cases{
        {"PP2 uniform", PipelineLayout::BuildUniformLayout(6, 2), false, {{3}, {3}}, 0, 1},
        {"PP2 custom vPP2", PipelineLayout::BuildCustomLayout(6, 2, two_stage_counts, 2),
         false, {{1, 1}, {2, 2}}, 0, 1},
        {"PP2 explicit", PipelineLayout::BuildChunkLayout(6, 2, two_stage_specs),
         true, {{2}, {1, 1, 2}}, 1, 1},
        {"PP3 uniform", PipelineLayout::BuildUniformLayout(8, 3), false, {{3}, {3}, {2}}, 0, 2},
        {"PP3 custom", PipelineLayout::BuildCustomLayout(6, 3, three_stage_counts),
         false, {{1}, {2}, {3}}, 0, 2},
        {"PP3 uniform vPP2", PipelineLayout::BuildUniformLayout(9, 3, 2),
         false, {{2, 1}, {2, 1}, {2, 1}}, 0, 2},
        {"PP3 custom vPP2", PipelineLayout::BuildCustomLayout(9, 3, three_stage_vpp_counts, 2),
         false, {{1, 1}, {2, 1}, {1, 3}}, 0, 2},
        {"PP3 explicit", PipelineLayout::BuildChunkLayout(6, 3, three_stage_specs),
         true, {{1}, {1, 1}, {1, 2}}, 1, 2},
    };
    for (bool llama : {false, true}) {
        for (const auto &test : cases) {
            const auto layout = std::make_shared<const PipelineLayout>(test.layout);
            const int stages = static_cast<int>(test.chunk_sizes.size());
            auto config = SmallConfig(llama);
            config.n_layer = static_cast<int>(layout->GetNumLayers());
            for (int stage_id = 0; stage_id < stages; ++stage_id) {
                SCOPED_TRACE(::testing::Message() << "llama=" << llama << " case=" << test.name << " stage=" << stage_id);
                auto model = std::make_shared<TransformerModel>(config, layout, stage_id);
                EXPECT_EQ(model->GetStageId(), stage_id);
                EXPECT_EQ(model->GetPipelineLayout(), layout);
                PipelineParallel wrapper(model, stages, 2, shapes_, stage_id, Device(), layout, test.explicit_mode);
                EXPECT_EQ(wrapper.GetPipelineLayout(), layout); // Regression for moved-from pointer check.
                const auto &stage = layout->GetStage(stage_id);
                const auto &chunks = *wrapper.mutable_chunks();
                ASSERT_EQ(chunks.size(), test.chunk_sizes[stage_id].size());
                EXPECT_EQ(stage.has_embedding, stage_id == test.input_stage);
                EXPECT_EQ(stage.has_final_norm, stage_id == test.output_stage);
                EXPECT_EQ(stage.has_lm_head, stage_id == test.output_stage);
                if (stage_id != test.input_stage) {
                    EXPECT_THROW((void)model->mutable_module(Module::kPPFirstStageName), std::out_of_range);
                }
                if (stage_id != test.output_stage) {
                    EXPECT_THROW((void)model->mutable_module(Module::kPPLastStageName), std::out_of_range);
                }
                EXPECT_EQ(model->GetNumChunks(), static_cast<int>(chunks.size()));
                auto all_layers = std::dynamic_pointer_cast<ModuleList>(
                    model->mutable_module(TransformerModel::kTransformerModelName)
                        ->mutable_module(TransformerChunk::kHLayerName));
                ASSERT_NE(all_layers, nullptr);
                std::size_t offset = 0;
                for (std::size_t local = 0; local < chunks.size(); ++local) {
                    auto original = model->mutable_module(Module::kPPChunkNamePrefix + std::to_string(local));
                    auto layers = std::dynamic_pointer_cast<ModuleList>(
                        original->mutable_module(TransformerChunk::kHLayerName));
                    ASSERT_NE(layers, nullptr);
                    ASSERT_EQ(std::distance(layers->begin(), layers->end()), test.chunk_sizes[stage_id][local]);
                    for (const auto &layer : *layers) { EXPECT_EQ(layer, (*all_layers)[offset++]); }
                    std::vector<std::shared_ptr<Module>> expected;
                    if (local == 0 && stage.has_embedding) {
                        expected.push_back(model->mutable_module(Module::kPPFirstStageName));
                    }
                    expected.push_back(original);
                    if (local + 1 == chunks.size() && stage.has_lm_head) {
                        expected.push_back(model->mutable_module(Module::kPPLastStageName));
                    }
                    EXPECT_EQ(chunks[local]->type(), Sequential::kType);
                    for (std::size_t part = 0; part < expected.size(); ++part) {
                        EXPECT_EQ(chunks[local]->mutable_module(std::to_string(part)), expected[part]);
                    }
                    EXPECT_THROW((void)chunks[local]->mutable_module(std::to_string(expected.size())),
                                 std::out_of_range);
                }
                EXPECT_EQ(std::distance(all_layers->begin(), all_layers->end()), offset);
            }
        }
    }
}

TEST_F(PipelineParallelTest, SingleStageRetainsLegacyModelAndWrapper) {
    auto config = SmallConfig();
    auto legacy_model = std::make_shared<TransformerModel>(config);
    auto layout = std::make_shared<const PipelineLayout>(PipelineLayout::BuildUniformLayout(6, 1));
    auto model = std::make_shared<TransformerModel>(config, layout, 0);
    EXPECT_EQ(legacy_model->GetStageId(), 0);
    const auto old_state = legacy_model->StateDict();
    const auto new_state = model->StateDict();
    ASSERT_FALSE(old_state.empty());
    ASSERT_EQ(old_state.size(), new_state.size());
    for (const auto &[name, parameter] : old_state) {
        ASSERT_TRUE(new_state.contains(name)) << name;
        EXPECT_EQ(parameter->Dims(), new_state.at(name)->Dims());
    }
    PipelineParallel legacy(legacy_model, 1, 2, shapes_, 0, Device(), 1);
    PipelineParallel current(model, 1, 2, shapes_, 0, Device(), layout);
    ASSERT_EQ(legacy.mutable_chunks()->size(), 1);
    ASSERT_EQ(current.mutable_chunks()->size(), 1);
    for (int part = 0; part < 3; ++part) {
        EXPECT_EQ(legacy.mutable_chunks()->at(0)->module(std::to_string(part)).type(),
                  current.mutable_chunks()->at(0)->module(std::to_string(part)).type());
    }
}

TEST_F(PipelineParallelTest, RejectsInvalidModelAndWrapperInputs) {
    auto layout = std::make_shared<const PipelineLayout>(PipelineLayout::BuildUniformLayout(6, 2));
    const auto config = SmallConfig();
    EXPECT_THROW((void)TransformerModel(config, nullptr, 0), std::invalid_argument);
    EXPECT_THROW((void)TransformerModel(config, layout, 2), std::out_of_range);
    auto wrong_config = config;
    wrong_config.n_layer = 7;
    EXPECT_THROW((void)TransformerModel(wrong_config, layout, 0), std::invalid_argument);
    auto model = std::make_shared<TransformerModel>(config, layout, 0);
    EXPECT_THROW((void)PipelineParallel(model, 2, 2, shapes_, 0, Device(), nullptr), std::invalid_argument);
    EXPECT_THROW((void)PipelineParallel(nullptr, 2, 2, shapes_, 0, Device(), layout), std::invalid_argument);
    EXPECT_THROW((void)PipelineParallel(model, 3, 2, shapes_, 0, Device(), layout), std::invalid_argument);
    EXPECT_THROW((void)PipelineParallel(model, 2, 2, shapes_, 2, Device(), layout), std::out_of_range);
    EXPECT_THROW((void)PipelineParallel(model, 2, 0, shapes_, 0, Device(), layout), std::invalid_argument);
    const std::vector<LayerIndex> other_counts{1, 5};
    auto other_layout = std::make_shared<const PipelineLayout>(PipelineLayout::BuildCustomLayout(6, 2, other_counts));
    EXPECT_THROW((void)PipelineParallel(model, 2, 2, shapes_, 0, Device(), other_layout), std::invalid_argument);
    EXPECT_THROW((void)PipelineParallel(model, 2, 2, shapes_, 1, Device(), layout), std::invalid_argument);
}

TEST_F(PipelineParallelTest, LegacyStageInfoPreservesInterleavedRanges) {
    const auto stage0 = PipelineParallel::GetStageInfo(13, 2, 0, 2);
    const auto stage1 = PipelineParallel::GetStageInfo(13, 2, 1, 2);
    EXPECT_EQ(stage0.layer_ranges_per_chunk, (std::vector<std::pair<int, int>>{{0, 4}, {7, 10}}));
    EXPECT_EQ(stage1.layer_ranges_per_chunk, (std::vector<std::pair<int, int>>{{4, 7}, {10, 13}}));
    EXPECT_TRUE(stage0.is_first_stage);
    EXPECT_FALSE(stage0.is_last_stage);
    EXPECT_FALSE(stage1.is_first_stage);
    EXPECT_TRUE(stage1.is_last_stage);
    EXPECT_TRUE(PipelineParallel::GetStageInfo(0, 2, 0, 2).layer_ranges_per_chunk.empty());
}

// Invalid requests must fail before any point-to-point communication.
TEST_F(PipelineParallelTest, ScheduleRejectsInconsistentModeAndInvalidInputs) {
    const std::vector<PipelineChunkSpec> specs{{1, 2}, {0, 2}, {1, 2}};
    auto layout = std::make_shared<const PipelineLayout>(PipelineLayout::BuildChunkLayout(6, 2, specs));
    std::vector<std::shared_ptr<Module>> chunks{std::make_shared<Module>()};
    auto stage = std::make_shared<PipelineStage>(0, 2, shapes_, Device(), std::move(chunks));
    EXPECT_THROW((void)PipelineSchedule(stage, 2, 2, layout, false), std::invalid_argument);
    EXPECT_THROW((void)PipelineSchedule(stage, 2, 2, nullptr, true), std::invalid_argument);
    PipelineSchedule schedule(stage, 2, 2, layout, true);
    EXPECT_THROW((void)schedule.StepMicroBatches({}, {}, nullptr, DataType::kFLOAT32), std::invalid_argument);
}

} // namespace
} // namespace infini_train::nn::parallel
