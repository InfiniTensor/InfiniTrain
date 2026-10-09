#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "infini_train/include/checkpoint/save_planner.h"
#include "infini_train/include/lr_scheduler.h"
#include "infini_train/include/shard_spec.h"

namespace infini_train {
class Optimizer;
class LRScheduler;
class Tensor;
namespace nn {
class Module;
}

struct TrainerState {
    int64_t global_step = 0;
    int64_t consumed_train_samples = 0;
    int64_t n_layer = 0;
    int64_t n_head = 0;
    int64_t n_kv_head = 0;
    int64_t n_embd = 0;
    int64_t original_vocab_size = 0;
    int64_t padded_vocab_size = 0;
    int ddp_size = 1;
    int tp_size = 1;
    int sp_size = 1;
    int pp_size = 1;
    int vpp_size = 1;
};

class Checkpoint {
public:
    static void Save(const std::filesystem::path &checkpoint_dir, const nn::Module &model, const Optimizer *optimizer,
                     const TrainerState &state, const LRScheduler *lr_scheduler);

    static void Load(const std::filesystem::path &checkpoint_dir, nn::Module &model, Optimizer *optimizer,
                     TrainerState &state, LRScheduler *lr_scheduler);

    static void SaveStateDictFile(const std::filesystem::path &path,
                                  const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict);

    static std::unordered_map<std::string, std::shared_ptr<Tensor>>
    LoadStateDictFile(const std::filesystem::path &path);

    struct CheckpointMetadata {
        int version = 0;

        struct TensorEntry {
            std::string key;
            std::string dtype_str;
            std::vector<int64_t> global_shape;
            std::vector<int64_t> local_shape;
            std::vector<int64_t> global_offset;
            std::vector<int> axis_fragmentations;
            std::vector<ShardSegment> segments;
            std::string file;
            uint64_t offset = 0;
            uint64_t byte_size = 0;
            std::vector<int> stored_on_ranks;
            int pp_rank = 0;
        };

        std::vector<TensorEntry> tensors;
        bool has_metadata = false;
    };

    static CheckpointMetadata LoadMetadata(const std::filesystem::path &checkpoint_dir);
    static void SaveMetadataFile(const std::filesystem::path &path, const CheckpointMetadata &metadata);

private:
    struct SavedTensorLocation {
        uint64_t data_offset = 0;
        uint64_t byte_size = 0;
    };
    using SavedTensorLocations = std::unordered_map<std::string, SavedTensorLocation>;

    static SavedTensorLocations
    SaveStateDict(const std::filesystem::path &path,
                  const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict);

    static std::unordered_map<std::string, std::shared_ptr<Tensor>> LoadStateDict(const std::filesystem::path &path);

    static void SaveLocalShard(const std::filesystem::path &checkpoint_dir, const ShardedStateDict &sharded_sd,
                               const std::vector<checkpoint::WriteItem> &write_items,
                               const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict,
                               const std::unordered_map<std::string, std::shared_ptr<Tensor>> &optimizer_state,
                               int global_rank);

    static void SaveTrainerState(const std::filesystem::path &path, const TrainerState &state);
    static TrainerState LoadTrainerState(const std::filesystem::path &path);
};

} // namespace infini_train
