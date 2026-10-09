#include "infini_train/include/checkpoint/checkpoint_manager.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/checkpoint/checkpoint.h"
#include "infini_train/include/checkpoint/constants.h"
#include "infini_train/include/lr_scheduler.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/parallel/ddp/distributed_optimizer.h"

using namespace infini_train;
namespace nn = infini_train::nn;

namespace {

std::filesystem::path ResolveCheckpointDirectory(const std::filesystem::path &root) {
    const auto latest_path = root / checkpoint::kLatestIterationFilename;
    if (!std::filesystem::exists(latest_path)) {
        return root;
    }
    std::ifstream latest(latest_path);
    int64_t iteration = 0;
    latest >> iteration;
    const auto directory = root / std::format("iter_{:07d}", iteration);
    CHECK(std::filesystem::exists(directory)) << "Latest checkpoint directory does not exist: " << directory;
    return directory;
}

} // namespace

ResumeFromCheckpointResult ResumeFromCheckpoint(const ResumeFromCheckpointArgs &args) {
    ResumeFromCheckpointResult result;
    if (args.resume_root.empty()) {
        LOG(INFO) << "No checkpoint specified for resume. Starting training from scratch.";
        return result;
    }
    CHECK(dynamic_cast<nn::parallel::DistributedOptimizer *>(args.optimizer.get()) == nullptr)
        << "Checkpoint restore does not support DistributedOptimizer/ZeRO optimizer state; use zero_stage=0";

    const auto checkpoint_dir = ResolveCheckpointDirectory(args.resume_root);
    CHECK(std::filesystem::exists(checkpoint_dir / checkpoint::kMetadataFilename))
        << "Checkpoint metadata.json not found: " << checkpoint_dir;
    Checkpoint::Load(checkpoint_dir, *args.model, args.optimizer.get(), args.state, args.lr_scheduler.get());

    CHECK_EQ(args.state.n_layer, args.model_config.n_layer)
        << "n_layer mismatch: ckpt=" << args.state.n_layer << ", config=" << args.model_config.n_layer;
    CHECK_EQ(args.state.n_head, args.model_config.n_head)
        << "n_head mismatch: ckpt=" << args.state.n_head << ", config=" << args.model_config.n_head;
    CHECK_EQ(args.state.n_kv_head, args.model_config.n_kv_head)
        << "n_kv_head mismatch: ckpt=" << args.state.n_kv_head << ", config=" << args.model_config.n_kv_head;
    CHECK_EQ(args.state.n_embd, args.model_config.n_embd)
        << "n_embd mismatch: ckpt=" << args.state.n_embd << ", config=" << args.model_config.n_embd;
    if (args.state.original_vocab_size > 0) {
        CHECK_EQ(args.state.original_vocab_size, args.model_config.original_vocab_size)
            << "original_vocab_size mismatch: ckpt=" << args.state.original_vocab_size
            << ", config=" << args.model_config.original_vocab_size;
        CHECK_GE(args.state.padded_vocab_size, args.state.original_vocab_size)
            << "Checkpoint padded vocabulary cannot represent its logical vocabulary";
    } else {
        // Legacy trainer_state.json only stored the padded vocab size.
        CHECK_GE(args.state.padded_vocab_size, args.model_config.original_vocab_size)
            << "Legacy checkpoint vocabulary cannot represent the configured logical vocabulary";
    }
    result.global_step = static_cast<int>(args.state.global_step);
    result.consumed_train_samples = static_cast<size_t>(std::max<int64_t>(args.state.consumed_train_samples, 0));
    if (args.rank.IsMainRank()) {
        LOG(INFO) << std::format("Resume training from step {}, consumed_train_samples {}", args.state.global_step,
                                 args.state.consumed_train_samples);
    }
    return result;
}

void SaveCheckpoint(const SaveCheckpointArgs &args) {
    CHECK(dynamic_cast<const nn::parallel::DistributedOptimizer *>(args.optimizer) == nullptr)
        << "Checkpoint save does not support DistributedOptimizer/ZeRO optimizer state; use zero_stage=0";
    CHECK_GT(args.original_vocab_size, 0);
    CHECK_GE(args.padded_vocab_size, args.original_vocab_size);
    const auto checkpoint_start = std::chrono::high_resolution_clock::now();
    // Snapshot training progress and the topology that produced this checkpoint.
    TrainerState state{.global_step = args.global_step,
                       .consumed_train_samples = static_cast<int64_t>(args.consumed_train_samples),
                       .n_layer = args.n_layer,
                       .n_head = args.n_head,
                       .n_kv_head = args.n_kv_head,
                       .n_embd = args.n_embd,
                       .original_vocab_size = args.original_vocab_size,
                       .padded_vocab_size = args.padded_vocab_size,
                       .ddp_size = args.ddp_size,
                       .tp_size = args.tp_size,
                       .sp_size = args.sp_size,
                       .pp_size = args.pp_size,
                       .vpp_size = args.vpp_size};
    const auto iteration_dir = args.checkpoint_root_dir.empty()
                                 ? args.save_dir
                                 : args.checkpoint_root_dir / std::format("iter_{:07d}", args.global_step);
    Checkpoint::Save(iteration_dir, args.model, args.optimizer, state, args.lr_scheduler);

    if (args.rank.IsMainRank() && !args.checkpoint_root_dir.empty()) {
        const auto latest = args.checkpoint_root_dir / checkpoint::kLatestIterationFilename;
        const auto temporary_latest = args.checkpoint_root_dir / checkpoint::kTemporaryLatestIterationFilename;
        {
            std::ofstream output(temporary_latest);
            CHECK(output.is_open());
            output << args.global_step;
        }
        if (std::filesystem::exists(latest)) {
            std::filesystem::remove(latest);
        }
        std::filesystem::rename(temporary_latest, latest);
    }

    if (args.rank.IsMainRank() && args.max_checkpoint_keep > 0 && std::filesystem::exists(args.checkpoint_root_dir)) {
        std::vector<std::filesystem::path> checkpoints;
        for (const auto &entry : std::filesystem::directory_iterator(args.checkpoint_root_dir)) {
            if (entry.is_directory() && entry.path().filename().string().starts_with("iter_")) {
                checkpoints.push_back(entry.path());
            }
        }
        // FIXME(jym): Pruning relies on lexicographic sorting of checkpoint directory names.
        // This is only correct while iteration directories use zero-padded names (e.g. iter_0000042).
        // If the naming convention changes to unpadded names, parse the iteration and sort numerically instead.
        std::sort(checkpoints.begin(), checkpoints.end());
        while (checkpoints.size() > args.max_checkpoint_keep) {
            std::filesystem::remove_all(checkpoints.front());
            checkpoints.erase(checkpoints.begin());
        }
    }

    const auto checkpoint_end = std::chrono::high_resolution_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(checkpoint_end - checkpoint_start).count();
    LOG(INFO) << std::format("Checkpoint saved at: {} ({:.2f} ms)", iteration_dir.string(), elapsed_ms);
}

size_t DataLoaderBatchesToSkip(size_t consumed_train_samples, size_t local_batch_size, size_t ddp_world_size) {
    CHECK_GT(local_batch_size, 0);
    CHECK_GT(ddp_world_size, 0);
    CHECK_LE(local_batch_size, std::numeric_limits<size_t>::max() / ddp_world_size)
        << "Data loader batch size overflows size_t";
    const size_t global_loader_batch_size = local_batch_size * ddp_world_size;
    CHECK_EQ(consumed_train_samples % global_loader_batch_size, 0)
        << "consumed_train_samples=" << consumed_train_samples
        << " does not align with current local_batch_size=" << local_batch_size
        << " and ddp_world_size=" << ddp_world_size;
    return consumed_train_samples / global_loader_batch_size;
}
