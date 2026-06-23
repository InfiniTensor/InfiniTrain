#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace infini_train {
class Optimizer;
class LRScheduler;
class Tensor;
namespace nn {
class Module;
}

struct TrainerState {
    int64_t global_step = 0;
    int64_t consumed_batches = 0;
    int64_t n_layer = 0;
    int64_t n_head = 0;
    int64_t n_kv_head = 0;
    int64_t n_embd = 0;
    int64_t vocab_size = 0;
    int ddp_size = 1;
    int tp_size = 1;
    int sp_size = 1;
    int pp_size = 1;
};

class Checkpoint {
public:
    static void Save(const std::filesystem::path &checkpoint_dir, const nn::Module &model, const Optimizer *optimizer,
                     const TrainerState &state, bool save_optimizer_state, const LRScheduler *lr_scheduler = nullptr);

    static void Load(const std::filesystem::path &checkpoint_dir, nn::Module &model, Optimizer *optimizer,
                     TrainerState &state, bool load_optimizer_state, LRScheduler *lr_scheduler = nullptr);

private:
    static void SaveStateDict(const std::filesystem::path &path,
                              const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict);

    static std::unordered_map<std::string, std::shared_ptr<Tensor>> LoadStateDict(const std::filesystem::path &path);

    static void SaveTrainerState(const std::filesystem::path &path, const TrainerState &state);
    static TrainerState LoadTrainerState(const std::filesystem::path &path);
};

} // namespace infini_train
