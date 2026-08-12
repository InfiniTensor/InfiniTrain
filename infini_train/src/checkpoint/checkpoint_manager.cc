#include "infini_train/include/checkpoint/checkpoint_manager.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <thread>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/checkpoint/checkpoint.h"
#include "infini_train/include/checkpoint/reshard.h"
#include "infini_train/include/checkpoint/save_planner.h"
#include "infini_train/include/lr_scheduler.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/parallel/ddp/distributed_optimizer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/work.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;
namespace nn = infini_train::nn;

namespace {

std::filesystem::path ResolveCheckpointDirectory(const std::filesystem::path &root) {
    const auto latest_path = root / "latest_checkpointed_iteration.txt";
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

void SynchronizeCheckpointRanks(const nn::Module &model) {
    const auto parameters = model.Parameters();
    CHECK(!parameters.empty()) << "Cannot synchronize checkpoint save for a model without parameters";
    auto token = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, parameters.front()->GetDevice());
    token->Fill(1.0f);
    nn::parallel::function::AllReduce(token, nn::parallel::function::ReduceOpType::kSum, nullptr, true)->Synchronize();
}

void WaitForWriterManifests(const std::filesystem::path &staging_root, int tp_size, int pp_size,
                            int64_t expected_iteration) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(10);
    for (;;) {
        bool ready = true;
        for (int pp = 0; pp < pp_size && ready; ++pp) {
            for (int tp = 0; tp < tp_size; ++tp) {
                const int rank = nn::parallel::global::GetRankOf(0, tp, pp);
                const auto manifest = staging_root / std::format("rank_{:06d}", rank) / "metadata.json";
                if (!std::filesystem::exists(manifest)) {
                    ready = false;
                    break;
                }
                const auto rank_metadata = Checkpoint::LoadMetadata(manifest.parent_path());
                if (!rank_metadata.has_metadata || rank_metadata.iteration != expected_iteration) {
                    ready = false;
                    break;
                }
            }
        }
        if (ready) {
            return;
        }
        CHECK(std::chrono::steady_clock::now() < deadline)
            << "Timed out waiting for checkpoint manifests in " << staging_root;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void WaitForGlobalMetadata(const std::filesystem::path &metadata_path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(10);
    while (!std::filesystem::exists(metadata_path)) {
        CHECK(std::chrono::steady_clock::now() < deadline)
            << "Timed out waiting for global checkpoint metadata: " << metadata_path;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
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

    auto checkpoint_dir = ResolveCheckpointDirectory(args.resume_root);
    CHECK(std::filesystem::exists(checkpoint_dir / "metadata.json"))
        << "Checkpoint metadata.json not found: " << checkpoint_dir;
    auto metadata = Checkpoint::LoadMetadata(checkpoint_dir);

    CHECK(metadata.has_metadata);
    CHECK_EQ(metadata.version, 3) << "Unsupported distributed checkpoint version: " << metadata.version;
    checkpoint::LoadDistributedCheckpoint(checkpoint_dir, *args.model, args.optimizer.get(), args.state,
                                          args.lr_scheduler.get(), metadata);

    CHECK_EQ(args.state.n_layer, args.model_config.n_layer);
    CHECK_EQ(args.state.n_head, args.model_config.n_head);
    CHECK_EQ(args.state.n_kv_head, args.model_config.n_kv_head);
    CHECK_EQ(args.state.n_embd, args.model_config.n_embd);
    CHECK_GE(args.state.vocab_size, args.model_config.original_vocab_size)
        << "Checkpoint vocabulary cannot represent the configured logical vocabulary";
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
    const auto checkpoint_start = std::chrono::high_resolution_clock::now();
    TrainerState state{.global_step = args.global_step,
                       .consumed_train_samples = static_cast<int64_t>(args.consumed_train_samples),
                       .n_layer = args.n_layer,
                       .n_head = args.n_head,
                       .n_kv_head = args.n_kv_head,
                       .n_embd = args.n_embd,
                       .vocab_size = args.vocab_size,
                       .ddp_size = args.ddp_size,
                       .tp_size = args.tp_size,
                       .sp_size = args.sp_size,
                       .pp_size = args.pp_size,
                       .vpp_size = args.vpp_size};
    const auto iteration_dir = args.checkpoint_root_dir.empty()
                                 ? args.save_dir
                                 : args.checkpoint_root_dir / std::format("iter_{:07d}", args.global_step);
    std::filesystem::create_directories(iteration_dir);

    const auto staging_root = iteration_dir / ".metadata_tmp";
    if (args.rank.IsMainRank()) {
        std::filesystem::remove_all(staging_root);
    }
    SynchronizeCheckpointRanks(args.model);

    int dp_rank = 0, tp_rank = 0, pp_rank = 0;
    nn::parallel::global::GetCoordOf(args.rank.GlobalRank(), dp_rank, tp_rank, pp_rank);
    if (dp_rank != 0) {
        return;
    }

    const auto rank_dir = iteration_dir / std::format("rank_{:06d}", args.rank.GlobalRank());
    std::filesystem::create_directories(rank_dir);
    auto sharded_state = args.model.ShardedStateDict();
    std::unordered_map<std::string, std::shared_ptr<Tensor>> optimizer_state;
    if (args.optimizer != nullptr) {
        optimizer_state = args.optimizer->StateDict();
        auto optimizer_sharded_state = checkpoint::BuildOptimizerShardedStateDict(sharded_state, optimizer_state);
        sharded_state.Merge(std::move(optimizer_sharded_state));
    }
    auto write_items = checkpoint::SavePlanner::Plan(sharded_state, args.rank.GlobalRank());
    Checkpoint::SaveSharded(rank_dir, sharded_state, write_items, args.model.StateDict(), optimizer_state, state,
                            args.rank.GlobalRank());

    const auto staging_rank_dir = staging_root / std::format("rank_{:06d}", args.rank.GlobalRank());
    std::filesystem::create_directories(staging_rank_dir);
    const auto local_manifest = staging_rank_dir / "metadata.json";
    if (std::filesystem::exists(local_manifest)) {
        std::filesystem::remove(local_manifest);
    }
    std::filesystem::rename(rank_dir / "metadata.json", local_manifest);

    if (args.rank.IsMainRank()) {
        Checkpoint::SaveTrainerStateFile(iteration_dir / "trainer_state.json", state);
        if (args.lr_scheduler != nullptr) {
            Checkpoint::SaveLRSchedulerStateFile(iteration_dir / "lr_scheduler.ckpt", args.lr_scheduler->StateDict());
        }
        WaitForWriterManifests(staging_root, args.tp_size, args.pp_size, args.global_step);
        auto global_metadata = Checkpoint::LoadMetadata(staging_root);
        CHECK(global_metadata.has_metadata);
        const auto temporary_metadata = iteration_dir / "metadata.json.tmp";
        const auto final_metadata = iteration_dir / "metadata.json";
        if (std::filesystem::exists(temporary_metadata)) {
            std::filesystem::remove(temporary_metadata);
        }
        Checkpoint::SaveMetadataFile(temporary_metadata, global_metadata);
        if (std::filesystem::exists(final_metadata)) {
            std::filesystem::remove(final_metadata);
        }
        std::filesystem::rename(temporary_metadata, final_metadata);
        std::filesystem::remove_all(staging_root);
    } else {
        WaitForGlobalMetadata(iteration_dir / "metadata.json");
    }

    if (args.rank.IsMainRank() && !args.checkpoint_root_dir.empty()) {
        const auto latest = args.checkpoint_root_dir / "latest_checkpointed_iteration.txt";
        const auto temporary_latest = args.checkpoint_root_dir / "latest_checkpointed_iteration.txt.tmp";
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
