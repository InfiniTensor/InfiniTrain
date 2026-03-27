#include "example/common/utils.h"

#include <algorithm>
#include <chrono>

#include "gflags/gflags.h"
#include "gflags/gflags_declare.h"
#include "glog/logging.h"
#include "infini_train/include/nn/parallel/global.h"

namespace infini_train {

float ConvertBF16ToFloat(void *ptr) {
    uint16_t *raw_data = reinterpret_cast<uint16_t *>(ptr);
    uint32_t f32_bits = static_cast<uint32_t>(raw_data[0]) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(f));
    return f;
}

// Model Reader Helper Function
std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

void ReadMatrixAllFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols) {
    const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
    ifs.read(reinterpret_cast<char *>(dst), bytes);
}

// Shard Reader Functions
// Read Row Shard: [row_start : row_start+row_cnt) × [0:cols]
void ReadMatrixRowShardFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols, int64_t row_start,
                             int64_t row_cnt) {
    std::streampos base = ifs.tellg();
    const size_t row_bytes = static_cast<size_t>(cols) * sizeof(float);
    ifs.seekg(base + std::streamoff(row_start * row_bytes));
    // assume row-major
    ifs.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(row_cnt * row_bytes));
    ifs.seekg(base + std::streamoff(rows * row_bytes));
}

// Read Column Shard: [0:rows) × [col_start : col_start+col_cnt)
void ReadMatrixColShardFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols, int64_t col_start,
                             int64_t col_cnt) {
    std::streampos base = ifs.tellg();
    const size_t row_bytes = static_cast<size_t>(cols) * sizeof(float);
    const size_t pick_bytes = static_cast<size_t>(col_cnt) * sizeof(float);
    // assume row-major, need loop
    for (int64_t r = 0; r < rows; ++r) {
        ifs.seekg(base + std::streamoff(r * row_bytes + col_start * sizeof(float)));
        ifs.read(reinterpret_cast<char *>(dst + r * col_cnt), static_cast<std::streamsize>(pick_bytes));
    }
    ifs.seekg(base + std::streamoff(rows * row_bytes));
}

// Read Whole Array
void ReadVectorAllFloat(std::ifstream &ifs, float *dst, int64_t len) {
    ifs.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(len * sizeof(float)));
}

// Read Array Shard: [start : start+cnt)
void ReadVectorShardFloat(std::ifstream &ifs, float *dst, int64_t len, int64_t start, int64_t cnt) {
    std::streampos base = ifs.tellg();
    ifs.seekg(base + std::streamoff(start * sizeof(float)));
    ifs.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(cnt * sizeof(float)));
    ifs.seekg(base + std::streamoff(len * sizeof(float)));
}

ResumeFromCheckpointResult ResumeFromCheckpoint(const ResumeFromCheckpointArgs &args) {
    ResumeFromCheckpointResult result;
    int ddp_world_size = nn::parallel::global::GetDataParallelSize();

    if (args.resume_root.empty()) {
        LOG(INFO) << "No checkpoint specified for resume. Starting training from scratch.";
        return result;
    }

    std::filesystem::path resume_dir = args.resume_root;
    if (args.rank.IsParallel()) {
        const auto rank_dir = resume_dir / std::format("rank_{:06d}", args.rank.GlobalRank());
        if (std::filesystem::exists(rank_dir)) {
            resume_dir = rank_dir;
        }
    }

    Checkpoint::Load(resume_dir, args.model.get(), args.optimizer.get(), &args.state, args.load_options);

    result.global_step = static_cast<int>(args.state.global_step);
    result.best_loss = args.state.best_loss;
    if (args.state.data_batch_stride != static_cast<int64_t>(ddp_world_size) && args.rank.IsMainRank()) {
        LOG(WARNING) << std::format("Checkpoint data_batch_stride {} mismatches current ddp_world_size {}. "
                                    "Proceeding with recorded data_batch_idx {}.",
                                    args.state.data_batch_stride, ddp_world_size, args.state.data_batch_idx);
    }
    result.data_batch_idx = static_cast<size_t>(std::max<int64_t>(args.state.data_batch_idx, 0));
    args.train_iter = args.train_loader.IteratorAtBatchIndex(result.data_batch_idx);
    if (args.rank.IsMainRank()) {
        LOG(INFO) << std::format(
            "Resume training from step {} with best_loss {:.6f}, last_lr {:.3e}, data_batch_idx {}",
            args.state.global_step, args.state.best_loss, args.state.last_lr, args.state.data_batch_idx);
    }

    return result;
}

void SaveCheckpoint(const SaveCheckpointArgs &args) {
    const auto ckpt_start = std::chrono::high_resolution_clock::now();

    TrainerState state;
    state.global_step = args.global_step;
    state.data_batch_idx = static_cast<int64_t>(args.data_batch_idx);
    state.data_batch_stride = args.ddp_size;
    state.best_loss = args.best_loss;
    state.last_lr = args.last_lr;
    state.optimizer_type = args.optimizer_type;
    state.checkpoint_format = args.checkpoint_format;
    state.ddp_size = args.ddp_size;
    state.tp_size = args.tp_size;
    state.sp_size = args.sp_size;
    state.pp_size = args.pp_size;

    CheckpointOptions options;
    options.format = args.checkpoint_format;
    options.save_optimizer_state = args.save_optimizer_state;
    options.model_bin_writer = args.model_bin_writer;
    Checkpoint::Save(args.save_dir, args.model, args.optimizer, state, options);

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

} // namespace infini_train
