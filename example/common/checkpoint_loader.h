#pragma once

#include "infini_train/include/checkpoint.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/optimizer.h"

#include "gflags/gflags.h"

#include <cstdint>
#include <cstring>
#include <filesystem>

#include <functional>
#include <limits>
#include <string>

namespace infini_train {
namespace nn {
class TransformerModel;
}

namespace gpt2 {
std::shared_ptr<nn::TransformerModel> LoadFromLLMC(const std::string &filepath);
void SaveAsLLMC(const std::shared_ptr<nn::TransformerModel> &model, const std::string &filepath);
} // namespace gpt2
namespace llama3 {
std::shared_ptr<nn::TransformerModel> LoadFromLLMC(const std::string &filepath);
void SaveAsLLMC(const std::shared_ptr<nn::TransformerModel> &model, const std::string &filepath);
} // namespace llama3

struct ResumeFromCheckpointArgs {
    fLS::clstring resume_root;
    const nn::parallel::Rank &rank;
    std::shared_ptr<nn::Module> model;
    std::shared_ptr<Optimizer> optimizer;
    DistributedDataLoader &train_loader;
    TrainerState &state;
    DataLoaderIterator &train_iter;
    CheckpointLoadOptions load_options;
};

struct ResumeFromCheckpointResult {
    int global_step = 0;
    float best_loss = std::numeric_limits<float>::infinity();
    size_t data_batch_idx = 0;
};

struct SaveCheckpointArgs {
    std::filesystem::path save_dir;
    int64_t global_step = 0;
    size_t data_batch_idx = 0;
    float best_loss = std::numeric_limits<float>::infinity();
    double last_lr = 0.0;
    std::string optimizer_type;
    std::string checkpoint_format = "bin";
    int ddp_size = 1;
    int tp_size = 1;
    int sp_size = 1;
    int pp_size = 1;
    bool save_optimizer_state = true;
    bool prune_step_checkpoints = false;
    std::filesystem::path checkpoint_root_dir;
    size_t max_checkpoint_keep = 0;
    const nn::parallel::Rank &rank;
    const nn::Module &model;
    const Optimizer &optimizer;
    std::function<void(const nn::Module &, const std::filesystem::path &)> model_bin_writer;
};

ResumeFromCheckpointResult ResumeFromCheckpoint(const ResumeFromCheckpointArgs &args);

void SaveCheckpoint(const SaveCheckpointArgs &args);

} // namespace infini_train
