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
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace infini_train {

float ConvertBF16ToFloat(void *ptr);

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs);

void ReadMatrixAllFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols);

void ReadMatrixRowShardFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols, int64_t row_start,
                             int64_t row_cnt);

void ReadMatrixColShardFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols, int64_t col_start,
                             int64_t col_cnt);

void ReadVectorAllFloat(std::ifstream &ifs, float *dst, int64_t len);

void ReadVectorShardFloat(std::ifstream &ifs, float *dst, int64_t len, int64_t start, int64_t cnt);

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
