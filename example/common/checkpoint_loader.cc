#include "example/common/checkpoint_loader.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;
namespace nn = infini_train::nn;

ResumeFromCheckpointResult ResumeFromCheckpoint(const ResumeFromCheckpointArgs &args) {
    ResumeFromCheckpointResult result;
    if (args.resume_root.empty()) {
        LOG(INFO) << "No checkpoint specified for resume. Starting training from scratch.";
        return result;
    }

    int ddp_world_size = nn::parallel::global::GetDataParallelSize();
    int tp_world_size = nn::parallel::global::GetTensorParallelSize();
    int sp_world_size = nn::parallel::global::GetSequenceParallelEnabled() ? tp_world_size : 1;
    int pp_world_size = nn::parallel::global::GetPipelineParallelSize();

    std::filesystem::path resume_dir = args.resume_root;
    if (args.rank.IsParallel()) {
        const auto rank_dir = resume_dir / std::format("rank_{:06d}", args.rank.GlobalRank());
        if (std::filesystem::exists(rank_dir)) {
            resume_dir = rank_dir;
        }
    }

    Checkpoint::Load(resume_dir, *args.model, args.optimizer.get(), args.state);

    result.global_step = static_cast<int>(args.state.global_step);

    CHECK_EQ(args.state.ddp_size, ddp_world_size) << "DDP size mismatch: checkpoint has DDP=" << args.state.ddp_size
                                                  << ", but current run has DDP=" << ddp_world_size;
    CHECK_EQ(args.state.tp_size, tp_world_size)
        << "TP size mismatch: checkpoint has TP=" << args.state.tp_size << ", but current run has TP=" << tp_world_size;
    CHECK_EQ(args.state.sp_size, sp_world_size)
        << "SP size mismatch: checkpoint has SP=" << args.state.sp_size << ", but current run has SP=" << sp_world_size;
    CHECK_EQ(args.state.pp_size, pp_world_size)
        << "PP size mismatch: checkpoint has PP=" << args.state.pp_size << ", but current run has PP=" << pp_world_size;

    result.consumed_batches = static_cast<size_t>(std::max<int64_t>(args.state.consumed_batches, 0));
    if (args.rank.IsMainRank()) {
        LOG(INFO) << std::format("Resume training from step {}, last_lr {:.3e}, consumed_batches  {}",
                                 args.state.global_step, args.state.last_lr, args.state.consumed_batches);
    }

    return result;
}

void SaveCheckpoint(const SaveCheckpointArgs &args) {
    const auto ckpt_start = std::chrono::high_resolution_clock::now();

    TrainerState state;
    state.global_step = args.global_step;
    state.consumed_batches = static_cast<int64_t>(args.consumed_batches);
    state.last_lr = args.last_lr;
    state.ddp_size = args.ddp_size;
    state.tp_size = args.tp_size;
    state.sp_size = args.sp_size;
    state.pp_size = args.pp_size;

    Checkpoint::Save(args.save_dir, args.model, &args.optimizer, state);

    const auto ckpt_end = std::chrono::high_resolution_clock::now();
    const double ckpt_ms = std::chrono::duration<double, std::milli>(ckpt_end - ckpt_start).count();

    if (!args.rank.IsMainRank()) {
        return;
    }

    LOG(INFO) << std::format("Checkpoint saved at: {} ({:.2f} ms)", args.save_dir.string(), ckpt_ms);

    if (!args.prune_step_checkpoints) {
        return;
    }

    std::vector<std::filesystem::path> ckpts;
    if (std::filesystem::exists(args.checkpoint_root_dir)) {
        for (const auto &entry : std::filesystem::directory_iterator(args.checkpoint_root_dir)) {
            if (entry.is_directory() && entry.path().filename().string().starts_with("checkpoint_step_")) {
                ckpts.push_back(entry.path());
            }
        }
        std::sort(ckpts.begin(), ckpts.end());
        while (ckpts.size() > args.max_checkpoint_keep) {
            std::filesystem::remove_all(ckpts.front());
            ckpts.erase(ckpts.begin());
        }
    }
}
