// pipeline_parallel.cc
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
namespace {
constexpr char kModuleName[] = "module";
} // namespace

thread_local int pp_rank = 0;

void PipelineParallel::BuildPipelineStage(const std::vector<std::vector<int64_t>> &recv_shape, Device device,
                                          std::vector<std::shared_ptr<Module>> &&chunks) {
    pipeline_stage_ = std::make_shared<PipelineStage>(rank_, num_stages_, recv_shape, device, std::move(chunks));
}

void PipelineParallel::SetupSchedule(int num_micro_batches) {
    schedule_ = std::make_shared<PipelineSchedule>(pipeline_stage_, num_stages_, num_micro_batches);
}

float PipelineParallel::TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                                  const std::vector<std::shared_ptr<Tensor>> &target,
                                  const std::shared_ptr<Optimizer> &optimizer, const std::shared_ptr<Module> &loss_fn,
                                  DataType dtype) {
    std::shared_ptr<Tensor> stage_input;
    std::shared_ptr<Tensor> stage_target = target[0];
    if (rank_ == 0) {
        stage_input = input[0];
    }

    return schedule_->Step(stage_input, stage_target, optimizer, loss_fn, dtype);
}

PipelineParallel::PipelineParallel(const std::shared_ptr<Module> module, int num_stages, int num_micro_batches,
                                   const std::vector<std::vector<int64_t>> &recv_shape, int pp_rank, Device device,
                                   const StageInfo &stage_info)
    : num_stages_(num_stages), rank_(pp_rank) {
    modules_[kModuleName] = std::move(module);

    const int chunk_size = static_cast<int>(stage_info.layer_ranges_per_chunk.size());

    std::vector<std::shared_ptr<Module>> chunks;
    for (int chunk_id = 0; chunk_id < chunk_size; ++chunk_id) {
        std::vector<std::shared_ptr<Module>> chunk_parts;
        if (chunk_id == 0 && stage_info.is_first_stage) {
            chunk_parts.push_back(module->mutable_module(kPPFirstStageName));
        }
        chunk_parts.push_back(module->mutable_module(kPPChunkNamePrefix + std::to_string(chunk_id)));
        if (chunk_id == chunk_size - 1 && stage_info.is_last_stage) {
            chunk_parts.push_back(module->mutable_module(kPPLastStageName));
        }
        chunks.push_back(std::make_shared<Sequential>(std::move(chunk_parts)));
    }

    BuildPipelineStage(recv_shape, device, std::move(chunks));

    SetupSchedule(num_micro_batches);
}

std::vector<std::shared_ptr<Module>> *PipelineParallel::mutable_chunks() { return pipeline_stage_->mutable_chunks(); }

void PipelineParallel::ReportPipelineStats() {
    const int num_stages = num_stages_;
    if (num_stages <= 1) {
        return; // Nothing to compare with a single pipeline stage.
    }

    // Flush pending event timing and read this stage's accumulated compute time.
    const double fwd = schedule_->ForwardSeconds();
    const double bwd = schedule_->BackwardSeconds();
    const int64_t fwd_count = schedule_->ForwardTaskCount();
    const int64_t bwd_count = schedule_->BackwardTaskCount();

    Device device = pipeline_stage_->device();
    auto *pp_group = ProcessGroupFactory::Instance(device.type())
                         ->Get(GetPipelineParallelProcessGroupName(device.Rank().GlobalRank()));

    // Gather [fwd_seconds, bwd_seconds] from every pipeline rank along dim 0.
    const float host[2] = {static_cast<float>(fwd), static_cast<float>(bwd)};
    auto input = std::make_shared<Tensor>(host, std::vector<int64_t>{2}, DataType::kFLOAT32, device);
    auto gathered = std::make_shared<Tensor>(std::vector<int64_t>{2 * num_stages}, DataType::kFLOAT32, device);
    pp_group->AllGather(gathered, input, /*async_op=*/false);

    const auto gathered_cpu = gathered->To(Device());
    const float *data = static_cast<const float *>(gathered_cpu.DataPtr());

    std::vector<double> stage_fwd(num_stages), stage_bwd(num_stages), stage_total(num_stages);
    for (int s = 0; s < num_stages; ++s) {
        stage_fwd[s] = data[2 * s];
        stage_bwd[s] = data[2 * s + 1];
        stage_total[s] = stage_fwd[s] + stage_bwd[s];
    }

    // The gather is collective; only the first pipeline rank prints the summary.
    if (rank_ != 0) {
        return;
    }

    const double bottleneck = *std::max_element(stage_total.begin(), stage_total.end());
    const double average = std::accumulate(stage_total.begin(), stage_total.end(), 0.0) / num_stages;
    const double efficiency = bottleneck > 0.0 ? average / bottleneck : 0.0;
    const double imbalance_bubble = 1.0 - efficiency;

    // Use LOG(ERROR) so the summary reaches stderr even when glog's stderrthreshold
    // filters out INFO; this matches the per-step progress lines above.
    LOG(ERROR) << std::format("=== Pipeline Timing Summary ({} stages) ===", num_stages);
    LOG(ERROR) << std::format("{:<6} {:>14} {:>14} {:>14}", "Stage", "Fwd(ms)", "Bwd(ms)", "Total(ms)");
    for (int s = 0; s < num_stages; ++s) {
        LOG(ERROR) << std::format("{:<6} {:>14.3f} {:>14.3f} {:>14.3f}", s, stage_fwd[s] * 1e3, stage_bwd[s] * 1e3,
                                  stage_total[s] * 1e3);
    }
    LOG(ERROR) << std::format("Compute tasks per stage: {} forward + {} backward", fwd_count, bwd_count);
    LOG(ERROR) << std::format("Bottleneck stage: {:.3f} ms | average: {:.3f} ms", bottleneck * 1e3, average * 1e3);
    LOG(ERROR) << std::format("Load-imbalance bubble: {:.1f}% | pipeline efficiency: {:.1f}%",
                              imbalance_bubble * 100.0, efficiency * 100.0);
}
} // namespace infini_train::nn::parallel
