#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>

#include "infini_train/include/checkpoint/checkpoint.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/optimizer.h"

using namespace infini_train;
namespace nn = infini_train::nn;

namespace infini_train {
class LRScheduler;
}

namespace infini_train::nn {
class TransformerConfig;
}

struct ResumeFromCheckpointArgs {
    std::filesystem::path resume_root;
    const nn::parallel::Rank &rank;
    std::shared_ptr<nn::Module> model;
    std::shared_ptr<Optimizer> optimizer;
    const nn::TransformerConfig &model_config;
    TrainerState &state;
    bool load_optimizer_state;
    std::shared_ptr<LRScheduler> lr_scheduler = nullptr;
};

struct ResumeFromCheckpointResult {
    int global_step = 0;
    size_t consumed_batches = 0;
};

struct SaveCheckpointArgs {
    std::filesystem::path save_dir;
    int64_t global_step = 0;
    size_t consumed_batches = 0;
    int64_t n_layer = 0;
    int64_t n_head = 0;
    int64_t n_kv_head = 0;
    int64_t n_embd = 0;
    int64_t vocab_size = 0;
    int ddp_size = 1;
    int tp_size = 1;
    int sp_size = 1;
    int pp_size = 1;
    bool save_optimizer_state = true;
    std::filesystem::path checkpoint_root_dir;
    size_t max_checkpoint_keep = 0;
    const nn::parallel::Rank &rank;
    const nn::Module &model;
    const Optimizer &optimizer;
    const LRScheduler *lr_scheduler = nullptr;
};

ResumeFromCheckpointResult ResumeFromCheckpoint(const ResumeFromCheckpointArgs &args);

void SaveCheckpoint(const SaveCheckpointArgs &args);
