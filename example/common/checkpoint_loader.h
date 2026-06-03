#pragma once

#include "gflags/gflags.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <string>

#include "infini_train/include/checkpoint.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/optimizer.h"

using namespace infini_train;
namespace nn = infini_train::nn;

struct ResumeFromCheckpointArgs {
    std::filesystem::path resume_root;
    const nn::parallel::Rank &rank;
    std::shared_ptr<nn::Module> model;
    std::shared_ptr<Optimizer> optimizer;
    DistributedDataLoader &train_loader;
    TrainerState &state;
};

struct ResumeFromCheckpointResult {
    int global_step = 0;
    size_t consumed_batches = 0;
};

struct SaveCheckpointArgs {
    std::filesystem::path save_dir;
    int64_t global_step = 0;
    size_t consumed_batches = 0;
    double last_lr = 0.0;
    int ddp_size = 1;
    int tp_size = 1;
    int sp_size = 1;
    int pp_size = 1;
    bool no_save_optim = false;
    bool prune_step_checkpoints = false;
    std::filesystem::path checkpoint_root_dir;
    size_t max_checkpoint_keep = 0;
    const nn::parallel::Rank &rank;
    const nn::Module &model;
    const Optimizer &optimizer;
};

ResumeFromCheckpointResult ResumeFromCheckpoint(const ResumeFromCheckpointArgs &args);

void SaveCheckpoint(const SaveCheckpointArgs &args);
