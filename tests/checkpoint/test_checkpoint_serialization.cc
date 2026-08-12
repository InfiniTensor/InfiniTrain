#include <filesystem>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/checkpoint/checkpoint.h"
#include "infini_train/include/checkpoint/load_planner.h"
#include "infini_train/include/checkpoint/load_strategy.h"
#include "infini_train/include/checkpoint/save_planner.h"
#include "infini_train/include/checkpoint/shard_spec.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;
namespace nn = infini_train::nn;

namespace {

class NamedParameterModule final : public nn::Module {
public:
    void AddParameter(const std::string &name, const std::shared_ptr<Tensor> &parameter) {
        parameters_[name] = parameter;
    }

    void AddModule(const std::string &name, const std::shared_ptr<nn::Module> &module) { modules_[name] = module; }
};

} // namespace

TEST(ModuleNamedParametersTest, SupportsTorchStyleArgumentsAndSharedParameterDeduplication) {
    auto root = std::make_shared<NamedParameterModule>();
    auto child = std::make_shared<NamedParameterModule>();
    auto shared = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, Device());
    auto child_weight = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, Device());
    root->AddParameter("root_weight", shared);
    child->AddParameter("alias", shared);
    child->AddParameter("weight", child_weight);
    root->AddModule("child", child);

    const auto local = root->NamedParameters("model", false);
    ASSERT_EQ(local.size(), 1);
    EXPECT_EQ(local[0].first, "model.root_weight");
    EXPECT_EQ(local[0].second, shared);

    const auto deduplicated = root->NamedParameters("model");
    ASSERT_EQ(deduplicated.size(), 2);
    EXPECT_EQ(deduplicated[0].first, "model.root_weight");
    EXPECT_EQ(deduplicated[1].first, "model.child.weight");

    const auto aliases = root->NamedParameters("model", true, false);
    ASSERT_EQ(aliases.size(), 3);
    EXPECT_EQ(aliases[0].first, "model.root_weight");
    EXPECT_EQ(aliases[1].first, "model.child.alias");
    EXPECT_EQ(aliases[1].second, shared);
    EXPECT_EQ(aliases[2].first, "model.child.weight");
}

class CheckpointSerializationTest : public test::InfiniTrainTest {};

TEST(ShardedStateDictTest, RejectsDuplicateKeysWhenMerging) {
    checkpoint::ShardedStateDict destination;
    destination.tensors["weight"] = {.key = "weight"};
    checkpoint::ShardedStateDict source;
    source.tensors["weight"] = {.key = "weight"};

    EXPECT_DEATH(destination.Merge(std::move(source)), "Duplicate sharded state-dict key: weight");
}

TEST_P(CheckpointSerializationTest, SaveAndLoadModelFP32) {
    auto dir = std::filesystem::temp_directory_path() / "test_ckpt_fp32";
    std::filesystem::remove_all(dir);

    auto model1 = std::make_shared<nn::Linear>(3, 2, true, GetDevice());
    auto p1 = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    p1->Fill(0.42f);
    *model1->mutable_parameter("weight") = p1;
    auto p2 = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice());
    p2->Fill(-1.5f);
    *model1->mutable_parameter("bias") = p2;

    auto opt1 = std::make_shared<optimizers::Adam>(model1->Parameters(), 0.01);
    TrainerState saved{.global_step = 42, .consumed_train_samples = 100};
    Checkpoint::Save(dir, *model1, opt1.get(), saved, nullptr);

    auto model2 = std::make_shared<nn::Linear>(3, 2, true, GetDevice());
    auto q1 = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    q1->Fill(0.0f);
    *model2->mutable_parameter("weight") = q1;
    auto q2 = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice());
    q2->Fill(0.0f);
    *model2->mutable_parameter("bias") = q2;
    auto opt2 = std::make_shared<optimizers::Adam>(model2->Parameters(), 0.01);

    TrainerState loaded;
    Checkpoint::Load(dir, *model2, opt2.get(), loaded, nullptr);

    EXPECT_EQ(loaded.global_step, 42);
    EXPECT_EQ(loaded.consumed_train_samples, 100);

    test::ExpectTensorNear(model2->parameter("weight"), 0.42f, 1e-6f);

    std::filesystem::remove_all(dir);
}

TEST_P(CheckpointSerializationTest, DirectMetadataOffsetSupportsColumnSlices) {
    auto dir = std::filesystem::temp_directory_path() / "test_ckpt_region";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    auto matrix = std::make_shared<Tensor>(std::vector<int64_t>{4, 4}, DataType::kFLOAT32, Device());
    auto *values = static_cast<float *>(matrix->DataPtr());
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) { values[row * 4 + column] = row * 10.0f + column; }
    }
    auto path = dir / "model.ckpt";
    Checkpoint::SaveStateDictFile(path, {{"matrix", matrix}});
    constexpr uint64_t data_offset = sizeof(uint32_t) * 3 + sizeof(uint32_t) + sizeof("matrix") - 1 + sizeof(int8_t)
                                   + sizeof(uint32_t) + sizeof(int64_t) * 2 + sizeof(uint64_t);
    checkpoint::LoadPlan plan;
    plan.tensors["matrix"] = {.key = "matrix",
                              .dtype = DataType::kFLOAT32,
                              .global_shape = {4, 4},
                              .target_shape = {4, 2},
                              .shard_dim = 1,
                              .reads = {{.key = "matrix",
                                         .filename = "model.ckpt",
                                         .dtype = DataType::kFLOAT32,
                                         .global_shape = {4, 4},
                                         .byte_size = sizeof(float) * 16,
                                         .data_offset = data_offset,
                                         .shard_dim = 1,
                                         .source_offset = 1,
                                         .target_offset = 0,
                                         .length = 2,
                                         .source_shape = {4, 4}}}};
    checkpoint::IndexedRegionLoadStrategy strategy;
    const auto planned = strategy.Execute(dir, plan);
    const auto *planned_data = static_cast<const float *>(planned.at("matrix")->DataPtr());
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 2; ++column) {
            EXPECT_FLOAT_EQ(planned_data[row * 2 + column], row * 10.0f + column + 1);
        }
    }

    std::filesystem::remove_all(dir);
}

TEST(CheckpointLoadPlannerTest, PadsVocabularyTailWhenTargetTpUsesPaddedVocab) {
    const auto dir = std::filesystem::temp_directory_path() / "test_vocab_padding_reshard";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    auto source = std::make_shared<Tensor>(std::vector<int64_t>{5, 2}, DataType::kFLOAT32, Device());
    auto *source_data = static_cast<float *>(source->DataPtr());
    for (int i = 0; i < 10; ++i) { source_data[i] = static_cast<float>(i); }
    Checkpoint::SaveStateDictFile(dir / "model.ckpt", {{"lm_head.weight", source}});
    constexpr uint64_t data_offset = sizeof(uint32_t) * 3 + sizeof(uint32_t) + sizeof("lm_head.weight") - 1
                                   + sizeof(int8_t) + sizeof(uint32_t) + sizeof(int64_t) * 2 + sizeof(uint64_t);

    Checkpoint::CheckpointMetadata metadata;
    metadata.tensors.push_back({.key = "lm_head.weight",
                                .dtype_str = "float32",
                                .global_shape = {5, 2},
                                .local_shape = {5, 2},
                                .global_offset = {0, 0},
                                .axis_fragmentations = {1, 1},
                                .file = "model.ckpt",
                                .offset = data_offset,
                                .byte_size = sizeof(float) * 10});

    checkpoint::ShardedStateDict target;
    target.tensors["lm_head.weight"] = {.key = "lm_head.weight",
                                        .dtype = DataType::kFLOAT32,
                                        .global_shape = {8, 2},
                                        .local_shape = {4, 2},
                                        .global_offset = {4, 0},
                                        .axis_fragmentations = {2, 1}};

    const auto plan = checkpoint::LoadPlanner::PlanReshard(metadata, target);
    ASSERT_EQ(plan.tensors.at("lm_head.weight").trailing_zero_fill, 3);
    checkpoint::IndexedRegionLoadStrategy strategy;
    const auto loaded = strategy.Execute(dir, plan);
    const auto *values = static_cast<const float *>(loaded.at("lm_head.weight")->DataPtr());
    EXPECT_FLOAT_EQ(values[0], 8.0f);
    EXPECT_FLOAT_EQ(values[1], 9.0f);
    for (int i = 2; i < 8; ++i) { EXPECT_FLOAT_EQ(values[i], 0.0f); }

    std::filesystem::remove_all(dir);
}

TEST_P(CheckpointSerializationTest, GlobalMetadataRoundTrip) {
    auto dir = std::filesystem::temp_directory_path() / "test_global_metadata";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    Checkpoint::CheckpointMetadata metadata;
    metadata.version = 3;
    metadata.iteration = 17;
    metadata.has_metadata = true;
    metadata.parallel_config = {.tp_size = 2, .pp_size = 2, .dp_size = 1, .sp_size = 1};
    metadata.tensors.push_back({.key = "layer.0.weight",
                                .dtype_str = "float32",
                                .global_shape = {8, 4},
                                .local_shape = {4, 4},
                                .global_offset = {0, 0},
                                .axis_fragmentations = {2, 1},
                                .segments = {{.global_offset = 0, .local_offset = 0, .length = 4}},
                                .file = "rank_000000/model.ckpt",
                                .byte_size = 64,
                                .stored_on_ranks = {0},
                                .pp_rank = 0});
    Checkpoint::SaveMetadataFile(dir / "metadata.json", metadata);

    auto loaded = Checkpoint::LoadMetadata(dir);
    ASSERT_TRUE(loaded.has_metadata);
    EXPECT_EQ(loaded.iteration, 17);
    EXPECT_EQ(loaded.parallel_config.tp_size, 2);
    EXPECT_EQ(loaded.parallel_config.pp_size, 2);
    ASSERT_EQ(loaded.tensors.size(), 1);
    EXPECT_EQ(loaded.tensors[0].file, "rank_000000/model.ckpt");
    EXPECT_EQ(loaded.tensors[0].global_offset, std::vector<int64_t>({0, 0}));
    EXPECT_EQ(loaded.tensors[0].axis_fragmentations, std::vector<int>({2, 1}));
    ASSERT_EQ(loaded.tensors[0].segments.size(), 1);
    EXPECT_EQ(loaded.tensors[0].segments[0],
              (checkpoint::ShardSegment{.global_offset = 0, .local_offset = 0, .length = 4}));
    std::filesystem::remove_all(dir);
}

INFINI_TRAIN_REGISTER_TEST(CheckpointSerializationTest);

namespace {
Checkpoint::CheckpointMetadata::TensorEntry MakeSavedShard(const std::string &key, int count, int index,
                                                           int64_t global_size, const std::string &file) {
    return {.key = key,
            .dtype_str = "float32",
            .global_shape = {global_size, 4},
            .local_shape = {global_size / count, 4},
            .global_offset = {global_size / count * index, 0},
            .axis_fragmentations = {count, 1},
            .file = file};
}

checkpoint::ShardedStateDict MakeTarget(const std::string &key, int count, int index, int64_t global_size) {
    checkpoint::ShardedStateDict target;
    target.tensors[key] = {.key = key,
                           .dtype = DataType::kFLOAT32,
                           .global_shape = {global_size, 4},
                           .local_shape = {global_size / count, 4},
                           .global_offset = {global_size / count * index, 0},
                           .axis_fragmentations = {count, 1}};
    return target;
}
} // namespace

TEST(CheckpointOptimizerShardingTest, AdamMomentsReuseModelShardMetadata) {
    checkpoint::ShardedStateDict model;
    model.tensors["c_attn.weight"] = {
        .key = "c_attn.weight",
        .dtype = DataType::kFLOAT32,
        .global_shape = {24, 4},
        .local_shape = {6, 4},
        .global_offset = {0, 0},
        .axis_fragmentations = {4, 1},
        .segments = {
            {.global_offset = 0, .local_offset = 0, .length = 4},
            {.global_offset = 16, .local_offset = 4, .length = 1},
            {.global_offset = 20, .local_offset = 5, .length = 1},
        },
    };

    auto moment = std::make_shared<Tensor>(std::vector<int64_t>{6, 4}, DataType::kFLOAT32, Device());
    auto step = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kINT64, Device());
    std::unordered_map<std::string, std::shared_ptr<Tensor>> optimizer_state = {
        {"adam.m.c_attn.weight", moment},
        {"adam.v.c_attn.weight", moment},
        {"adam.t", step},
    };

    const auto optimizer = checkpoint::BuildOptimizerShardedStateDict(model, optimizer_state);
    ASSERT_EQ(optimizer.tensors.size(), 3);
    const auto &m = optimizer.tensors.at("adam.m.c_attn.weight");
    EXPECT_EQ(m.global_shape, model.tensors.at("c_attn.weight").global_shape);
    EXPECT_EQ(m.local_shape, model.tensors.at("c_attn.weight").local_shape);
    EXPECT_EQ(m.segments, model.tensors.at("c_attn.weight").segments);
    EXPECT_EQ(m.local_key, "adam.m.c_attn.weight");
    const auto &t = optimizer.tensors.at("adam.t");
    EXPECT_TRUE(t.global_shape.empty());
    EXPECT_TRUE(t.local_shape.empty());
}

TEST(CheckpointLoadPlannerTest, RejectsUnknownCheckpointDtype) {
    auto metadata = Checkpoint::CheckpointMetadata{};
    metadata.tensors = {MakeSavedShard("weight", 1, 0, 16, "rank_0/model.ckpt")};
    metadata.tensors.front().dtype_str = "unknown_dtype";

    EXPECT_DEATH(checkpoint::LoadPlanner::PlanReshard(metadata, MakeTarget("weight", 1, 0, 16)),
                 "Unsupported checkpoint tensor dtype: unknown_dtype");
}

TEST(CheckpointLoadPlannerTest, TensorParallelTwoToFourReadsOnlyOverlap) {
    Checkpoint::CheckpointMetadata metadata;
    metadata.tensors = {MakeSavedShard("weight", 2, 0, 16, "rank_0/model.ckpt"),
                        MakeSavedShard("weight", 2, 1, 16, "rank_1/model.ckpt")};

    auto plan = checkpoint::LoadPlanner::PlanReshard(metadata, MakeTarget("weight", 4, 1, 16));
    const auto &reads = plan.tensors.at("weight").reads;
    ASSERT_EQ(reads.size(), 1);
    EXPECT_EQ(reads[0].filename, "rank_0/model.ckpt");
    EXPECT_EQ(reads[0].source_offset, 4);
    EXPECT_EQ(reads[0].target_offset, 0);
    EXPECT_EQ(reads[0].length, 4);
}

TEST(CheckpointLoadPlannerTest, TensorParallelFourToTwoReadsTwoOverlaps) {
    Checkpoint::CheckpointMetadata metadata;
    for (int index = 0; index < 4; ++index) {
        metadata.tensors.push_back(
            MakeSavedShard("weight", 4, index, 16, "rank_" + std::to_string(index) + "/model.ckpt"));
    }

    auto plan = checkpoint::LoadPlanner::PlanReshard(metadata, MakeTarget("weight", 2, 1, 16));
    const auto &reads = plan.tensors.at("weight").reads;
    ASSERT_EQ(reads.size(), 2);
    EXPECT_EQ(reads[0].filename, "rank_2/model.ckpt");
    EXPECT_EQ(reads[0].target_offset, 0);
    EXPECT_EQ(reads[1].filename, "rank_3/model.ckpt");
    EXPECT_EQ(reads[1].target_offset, 4);
}

TEST(CheckpointLoadPlannerTest, UsesExplicitGlobalOffsetsForUnevenShards) {
    Checkpoint::CheckpointMetadata metadata;
    metadata.tensors = {{.key = "weight",
                         .dtype_str = "float32",
                         .global_shape = {8, 4},
                         .local_shape = {3, 4},
                         .global_offset = {0, 0},
                         .axis_fragmentations = {2, 1},
                         .file = "rank_0/model.ckpt"},
                        {.key = "weight",
                         .dtype_str = "float32",
                         .global_shape = {8, 4},
                         .local_shape = {5, 4},
                         .global_offset = {3, 0},
                         .axis_fragmentations = {2, 1},
                         .file = "rank_1/model.ckpt"}};
    checkpoint::ShardedStateDict target;
    target.tensors["weight"] = {.key = "weight",
                                .dtype = DataType::kFLOAT32,
                                .global_shape = {8, 4},
                                .local_shape = {4, 4},
                                .global_offset = {2, 0},
                                .axis_fragmentations = {2, 1}};

    auto plan = checkpoint::LoadPlanner::PlanReshard(metadata, target);
    const auto &reads = plan.tensors.at("weight").reads;
    ASSERT_EQ(reads.size(), 2);
    EXPECT_EQ(reads[0].source_offset, 2);
    EXPECT_EQ(reads[0].target_offset, 0);
    EXPECT_EQ(reads[0].length, 1);
    EXPECT_EQ(reads[1].source_offset, 0);
    EXPECT_EQ(reads[1].target_offset, 1);
    EXPECT_EQ(reads[1].length, 3);
}

TEST(CheckpointLoadPlannerTest, QkvSegmentsUseDimZeroWhenTpIsOne) {
    Checkpoint::CheckpointMetadata metadata;
    auto saved = MakeSavedShard("c_attn.weight", 1, 0, 12, "old_pp/model.ckpt");
    saved.local_shape = {12, 4};
    saved.axis_fragmentations = {1, 1};
    saved.segments = {
        {.global_offset = 0, .local_offset = 0, .length = 8},
        {.global_offset = 8, .local_offset = 8, .length = 2},
        {.global_offset = 10, .local_offset = 10, .length = 2},
    };
    metadata.tensors.push_back(std::move(saved));

    checkpoint::ShardedStateDict target;
    target.tensors["c_attn.weight"] = {
        .key = "c_attn.weight",
        .dtype = DataType::kFLOAT32,
        .global_shape = {12, 4},
        .local_shape = {12, 4},
        .global_offset = {0, 0},
        .axis_fragmentations = {1, 1},
        .segments = {
            {.global_offset = 0, .local_offset = 0, .length = 8},
            {.global_offset = 8, .local_offset = 8, .length = 2},
            {.global_offset = 10, .local_offset = 10, .length = 2},
        },
    };

    const auto plan = checkpoint::LoadPlanner::PlanReshard(metadata, target);
    const auto &tensor_plan = plan.tensors.at("c_attn.weight");
    EXPECT_EQ(tensor_plan.shard_dim, 0);
    ASSERT_EQ(tensor_plan.reads.size(), 3);
    EXPECT_EQ(tensor_plan.reads[0].target_offset, 0);
    EXPECT_EQ(tensor_plan.reads[1].target_offset, 8);
    EXPECT_EQ(tensor_plan.reads[2].target_offset, 10);
}

TEST(CheckpointLoadPlannerTest, QkvSegmentsPreserveTargetLocalLayoutAcrossTpChange) {
    Checkpoint::CheckpointMetadata metadata;
    for (int rank = 0; rank < 4; ++rank) {
        auto shard = MakeSavedShard("c_attn.weight", 4, rank, 24, "rank_" + std::to_string(rank) + "/model.ckpt");
        shard.local_shape = {6, 4};
        shard.global_offset = {0, 0};
        shard.segments = {
            {.global_offset = rank * 4, .local_offset = 0, .length = 4},
            {.global_offset = 16 + rank, .local_offset = 4, .length = 1},
            {.global_offset = 20 + rank, .local_offset = 5, .length = 1},
        };
        metadata.tensors.push_back(std::move(shard));
    }

    checkpoint::ShardedStateDict target;
    target.tensors["c_attn.weight"] = {
        .key = "c_attn.weight",
        .dtype = DataType::kFLOAT32,
        .global_shape = {24, 4},
        .local_shape = {12, 4},
        .global_offset = {0, 0},
        .axis_fragmentations = {2, 1},
        .segments = {
            {.global_offset = 0, .local_offset = 0, .length = 8},
            {.global_offset = 16, .local_offset = 8, .length = 2},
            {.global_offset = 20, .local_offset = 10, .length = 2},
        },
    };

    const auto plan = checkpoint::LoadPlanner::PlanReshard(metadata, target);
    const auto &reads = plan.tensors.at("c_attn.weight").reads;
    ASSERT_EQ(reads.size(), 6);
    const std::vector<std::string> files = {"rank_0/model.ckpt", "rank_1/model.ckpt", "rank_0/model.ckpt",
                                            "rank_1/model.ckpt", "rank_0/model.ckpt", "rank_1/model.ckpt"};
    const std::vector<int64_t> source_offsets = {0, 0, 4, 4, 5, 5};
    const std::vector<int64_t> target_offsets = {0, 4, 8, 9, 10, 11};
    for (size_t i = 0; i < reads.size(); ++i) {
        EXPECT_EQ(reads[i].filename, files[i]);
        EXPECT_EQ(reads[i].source_offset, source_offsets[i]);
        EXPECT_EQ(reads[i].target_offset, target_offsets[i]);
    }
}

TEST(CheckpointLoadPlannerTest, PipelineReshardPlansOnlyTargetStageKeys) {
    Checkpoint::CheckpointMetadata metadata;
    metadata.tensors = {MakeSavedShard("layer.0.weight", 1, 0, 8, "old_pp0/model.ckpt"),
                        MakeSavedShard("layer.1.weight", 1, 0, 8, "old_pp1/model.ckpt")};

    auto plan = checkpoint::LoadPlanner::PlanReshard(metadata, MakeTarget("layer.1.weight", 1, 0, 8));
    ASSERT_EQ(plan.tensors.size(), 1);
    ASSERT_EQ(plan.tensors.at("layer.1.weight").reads.size(), 1);
    EXPECT_EQ(plan.tensors.at("layer.1.weight").reads[0].filename, "old_pp1/model.ckpt");
}
