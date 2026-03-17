#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace infini_train {
class Optimizer;
class Tensor;
namespace nn {
class Module;
}

struct TrainerState {
    int64_t global_step = 0;
    int64_t data_batch_idx = 0;
    int64_t data_batch_stride = 1;
    float best_loss = 0.0f;
    double last_lr = 0.0;
    std::string optimizer_type = "unknown";
    std::string checkpoint_format = "bin";

    int ddp_size = 1;
    int tp_size = 1;
    int sp_size = 1;
    int pp_size = 1;
};

struct CheckpointOptions {
    std::string format = "bin";
    bool save_optimizer_state = true;
    std::function<void(const nn::Module &, const std::filesystem::path &)> model_bin_writer;
};

struct CheckpointLoadOptions {
    bool load_optimizer_state = true;
    std::function<void(nn::Module *, const std::filesystem::path &)> model_bin_loader;
};

class Checkpoint {
public:
    static void Save(const std::filesystem::path &checkpoint_dir, const nn::Module &model, const Optimizer &optimizer,
                     const TrainerState &state, const CheckpointOptions &options = {});

    static void Load(const std::filesystem::path &checkpoint_dir, nn::Module *model, Optimizer *optimizer,
                     TrainerState *state, const CheckpointLoadOptions &options = {});

private:
    static void SaveStateDictBinary(const std::filesystem::path &path,
                                    const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict);

    static std::unordered_map<std::string, std::shared_ptr<Tensor>>
    LoadStateDictBinary(const std::filesystem::path &path);

    static void SaveTrainerState(const std::filesystem::path &path, const TrainerState &state);
    static TrainerState LoadTrainerState(const std::filesystem::path &path);
    static std::string InferFormat(const std::filesystem::path &checkpoint_dir);
};

} // namespace infini_train
