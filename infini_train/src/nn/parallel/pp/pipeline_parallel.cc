// pipeline_parallel.cc
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/parallel/pp/pipeline_layout.h"
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

namespace infini_train::nn::parallel {
namespace {
constexpr char kModuleName[] = "module";

std::vector<std::shared_ptr<Module>> BuildPipelineChunksFromLayout(const std::shared_ptr<Module> &module,
                                                                   const PipelineStageLayout &stage) {
    if (!module) {
        throw std::invalid_argument("pipeline module must not be null");
    }

    if (stage.has_final_norm != stage.has_lm_head) {
        throw std::invalid_argument("final norm and LM head must belong to the same pipeline stage");
    }

    std::vector<std::shared_ptr<Module>> chunks;
    chunks.reserve(stage.chunks.size());
    for (std::size_t chunk_idx = 0; chunk_idx < stage.chunks.size(); ++chunk_idx) {
        const PipelineChunkLayout &chunk_layout = stage.chunks[chunk_idx];
        if (chunk_layout.stage_id != stage.stage_id || chunk_layout.global_chunk_id < 0) {
            throw std::invalid_argument("pipeline chunk identity is inconsistent with its stage");
        }
        if (chunk_layout.local_chunk_idx != static_cast<int>(chunk_idx)) {
            throw std::invalid_argument("pipeline local chunk indices must be contiguous and start at zero");
        }

        std::vector<std::shared_ptr<Module>> chunk_parts;
        if (chunk_idx == 0 && stage.has_embedding) {
            chunk_parts.push_back(module->mutable_module(Module::kPPFirstStageName));
        }
        chunk_parts.push_back(module->mutable_module(std::string(Module::kPPChunkNamePrefix)
                                                     + std::to_string(chunk_layout.local_chunk_idx)));
        if (chunk_idx == stage.chunks.size() - 1 && stage.has_final_norm) {
            chunk_parts.push_back(module->mutable_module(Module::kPPLastStageName));
        }
        chunks.push_back(std::make_shared<Sequential>(std::move(chunk_parts)));
    }
    return chunks;
}
} // namespace

thread_local int pp_rank = 0;

void PipelineParallel::BuildPipelineStage(const std::vector<std::vector<int64_t>> &recv_shape, Device device,
                                          std::vector<std::shared_ptr<Module>> &&chunks) {
    pipeline_stage_ = std::make_shared<PipelineStage>(rank_, num_stages_, recv_shape, device, std::move(chunks));
}

void PipelineParallel::SetupSchedule(int num_micro_batches) {
    schedule_ = std::make_shared<PipelineSchedule>(pipeline_stage_, num_stages_, num_micro_batches, pipeline_layout_,
                                                   explicit_chunk_mode_);
}

float PipelineParallel::TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                                  const std::vector<std::shared_ptr<Tensor>> &target,
                                  const std::shared_ptr<Optimizer> &optimizer, const std::shared_ptr<Module> &loss_fn,
                                  DataType dtype) {
    std::shared_ptr<Tensor> stage_input;
    if (rank_ == (pipeline_layout_ ? pipeline_layout_->GetInputStage() : 0)) {
        stage_input = input.at(0);
    }

    return schedule_->Step(stage_input, target.at(0), optimizer, loss_fn, dtype);
}

StageInfo PipelineParallel::GetStageInfo(int total_layers, int pp_size, int rank, int chunks_per_stage) {
    const auto layout = PipelineLayout::BuildUniformLayout(total_layers, pp_size, chunks_per_stage);
    const auto &stage = layout.GetStage(rank);
    std::vector<std::pair<int, int>> local_ranges;
    for (const auto &chunk : stage.chunks) {
        local_ranges.emplace_back(static_cast<int>(chunk.layer_range.begin), static_cast<int>(chunk.layer_range.end));
    }
    return {rank == 0, rank == pp_size - 1, std::move(local_ranges)};
}

PipelineParallel::PipelineParallel(const std::shared_ptr<Module> module, int num_stages, int num_micro_batches,
                                   const std::vector<std::vector<int64_t>> &recv_shape, int pp_rank, Device device,
                                   int chunk_size)
    : num_stages_(num_stages), rank_(pp_rank) {
    modules_[kModuleName] = std::move(module);

    int stage_id = pp_rank;
    int stage_size = num_stages;

    std::vector<std::shared_ptr<Module>> chunks;
    for (int chunk_id = 0; chunk_id < chunk_size; ++chunk_id) {
        std::vector<std::shared_ptr<Module>> chunk_parts;
        if (chunk_id == 0 && stage_id == 0) {
            chunk_parts.push_back(module->mutable_module(kPPFirstStageName));
        }
        chunk_parts.push_back(module->mutable_module(kPPChunkNamePrefix + std::to_string(chunk_id)));
        if (chunk_id == chunk_size - 1 && stage_id == stage_size - 1) {
            chunk_parts.push_back(module->mutable_module(kPPLastStageName));
        }
        chunks.push_back(std::make_shared<Sequential>(std::move(chunk_parts)));
    }

    BuildPipelineStage(recv_shape, device, std::move(chunks));

    SetupSchedule(num_micro_batches);
}
PipelineParallel::PipelineParallel(const std::shared_ptr<Module> module, int num_stages, int num_micro_batches,
                                   const std::vector<std::vector<int64_t>> &recv_shape, int pp_rank, Device device,
                                   std::shared_ptr<const PipelineLayout> pipeline_layout, bool explicit_chunk_mode)
    : num_stages_(num_stages), rank_(pp_rank), pipeline_layout_(std::move(pipeline_layout)),
      explicit_chunk_mode_(explicit_chunk_mode) {
    if (pipeline_layout_ == nullptr) {
        throw std::invalid_argument("pipeline layout must not be null");
    }

    if (pipeline_layout_->GetNumStages() != num_stages_) {
        throw std::invalid_argument("pipeline layout stage count does not match PipelineParallel::num_stages");
    }

    const PipelineStageLayout &stage = pipeline_layout_->GetStage(rank_);
    if (const auto transformer = std::dynamic_pointer_cast<TransformerModel>(module)) {
        if (transformer->GetPipelineLayout() != pipeline_layout_ || transformer->GetStageId() != rank_) {
            throw std::invalid_argument("pipeline wrapper must share the model's layout and stage");
        }
    }
    modules_[kModuleName] = module;
    auto chunks = BuildPipelineChunksFromLayout(module, stage);
    BuildPipelineStage(recv_shape, device, std::move(chunks));
    SetupSchedule(num_micro_batches);
}
std::vector<std::shared_ptr<Module>> *PipelineParallel::mutable_chunks() { return pipeline_stage_->mutable_chunks(); }
} // namespace infini_train::nn::parallel
