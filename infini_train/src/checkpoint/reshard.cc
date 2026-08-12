#include "infini_train/include/checkpoint/reshard.h"

#include <filesystem>
#include <format>

#include "glog/logging.h"

#include "infini_train/include/checkpoint/load_planner.h"
#include "infini_train/include/checkpoint/load_strategy.h"
#include "infini_train/include/checkpoint/save_planner.h"
#include "infini_train/include/lr_scheduler.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/optimizer.h"

namespace infini_train::checkpoint {

void LoadDistributedCheckpoint(const std::filesystem::path &checkpoint_dir, nn::Module &model, Optimizer *optimizer,
                               TrainerState &state, LRScheduler *lr_scheduler,
                               const Checkpoint::CheckpointMetadata &metadata) {
    CHECK(metadata.has_metadata);
    CHECK_EQ(metadata.version, 3) << "Unsupported distributed checkpoint version: " << metadata.version;
    auto model_sharded_state = model.ShardedStateDict();
    auto plan = LoadPlanner::PlanReshard(metadata, model_sharded_state);
    IndexedRegionLoadStrategy strategy;
    auto result = strategy.Execute(checkpoint_dir, plan);
    model.LoadStateDict(result);

    state = Checkpoint::LoadTrainerStateFile(checkpoint_dir / "trainer_state.json");
    const int current_tp = nn::parallel::global::GetTensorParallelSize();
    const int current_pp = nn::parallel::global::GetPipelineParallelSize();
    const bool topology_changed
        = current_tp != metadata.parallel_config.tp_size || current_pp != metadata.parallel_config.pp_size;
    state.tp_size = current_tp;
    state.pp_size = current_pp;
    state.ddp_size = nn::parallel::global::GetDataParallelSize();
    state.sp_size = nn::parallel::global::GetSequenceParallelEnabled() ? current_tp : 1;

    if (optimizer != nullptr) {
        if (topology_changed) {
            const auto initialized_optimizer_state = optimizer->StateDict();
            auto optimizer_sharded_state
                = BuildOptimizerShardedStateDict(model_sharded_state, initialized_optimizer_state);
            auto optimizer_plan = LoadPlanner::PlanReshard(metadata, optimizer_sharded_state);
            auto loaded_optimizer_state = strategy.Execute(checkpoint_dir, optimizer_plan);
            optimizer->LoadStateDict(loaded_optimizer_state);
            LOG(INFO) << "[CKPT] Resharded " << loaded_optimizer_state.size()
                      << " optimizer tensors across TP/PP topology change";
        } else {
            int dp_rank = 0, tp_rank = 0, pp_rank = 0;
            nn::parallel::global::GetCoordOf(nn::parallel::global::thread_global_rank, dp_rank, tp_rank, pp_rank);
            const int writer_rank = nn::parallel::global::GetRankOf(0, tp_rank, pp_rank);
            const auto optimizer_path = checkpoint_dir / std::format("rank_{:06d}/optimizer.ckpt", writer_rank);
            CHECK(std::filesystem::exists(optimizer_path))
                << "Optimizer checkpoint not found for current_rank=" << nn::parallel::global::thread_global_rank
                << ", coords=(dp=" << dp_rank << ", tp=" << tp_rank << ", pp=" << pp_rank
                << "), writer_rank=" << writer_rank << ": " << optimizer_path;
            LOG(INFO) << "[CKPT] Loading optimizer for current_rank=" << nn::parallel::global::thread_global_rank
                      << " from writer_rank=" << writer_rank << ": " << optimizer_path;
            optimizer->LoadStateDict(Checkpoint::LoadStateDictFile(optimizer_path));
        }
    }
    if (lr_scheduler != nullptr && std::filesystem::exists(checkpoint_dir / "lr_scheduler.ckpt")) {
        lr_scheduler->LoadStateDict(Checkpoint::LoadLRSchedulerStateFile(checkpoint_dir / "lr_scheduler.ckpt"));
    }
    LOG(INFO) << "[CKPT] Restored " << result.size()
              << " tensors with overlap reads from TP=" << metadata.parallel_config.tp_size
              << ", PP=" << metadata.parallel_config.pp_size << " to TP=" << current_tp << ", PP=" << current_pp;
}

} // namespace infini_train::checkpoint
