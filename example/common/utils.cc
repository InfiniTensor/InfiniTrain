#include "example/common/utils.h"

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

std::tuple<int, float, size_t> ResumeFromCheckpoint(
    const fLS::clstring &flag_resume_root, // resume from this checkpoint directory
    const nn::parallel::Rank &rank,        // rank info for distributed training
    std::shared_ptr<nn::Module> model,     // model to be loaded with checkpoint state
    std::shared_ptr<Optimizer> optimizer,  // some optimizer may not have state, but others may have
    DistributedDataLoader &train_loader,   // distributed dataloader to be resumed
    TrainerState &state,                   // trainer state to be loaded from checkpoint
    DataLoaderIterator
        &train_iter, // dataloader iterator to be set to the correct position according to checkpoint state
    CheckpointLoadOptions model_bin_loader) {
    int global_step = 0;
    float best_loss = std::numeric_limits<float>::infinity();
    size_t data_batch_idx = 0;

    int ddp_world_size = nn::parallel::global::GetDataParallelSize();

    if (flag_resume_root.empty()) {
        LOG(INFO) << "No checkpoint specified for resume. Starting training from scratch.";
        return {global_step, best_loss, data_batch_idx};
    }

    std::filesystem::path resume_dir = flag_resume_root;
    if (rank.IsParallel()) {
        const auto rank_dir = resume_dir / std::format("rank_{:06d}", rank.GlobalRank());
        if (std::filesystem::exists(rank_dir)) {
            resume_dir = rank_dir;
        }
    }

    Checkpoint::Load(resume_dir, model.get(), optimizer.get(), &state, model_bin_loader);

    global_step = static_cast<int>(state.global_step);
    best_loss = state.best_loss;
    if (state.data_batch_stride != static_cast<int64_t>(ddp_world_size) && rank.IsMainRank()) {
        LOG(WARNING) << std::format("Checkpoint data_batch_stride {} mismatches current ddp_world_size {}. "
                                    "Proceeding with recorded data_batch_idx {}.",
                                    state.data_batch_stride, ddp_world_size, state.data_batch_idx);
    }
    data_batch_idx = static_cast<size_t>(std::max<int64_t>(state.data_batch_idx, 0));
    train_iter = train_loader.IteratorAtBatchIndex(data_batch_idx);
    if (rank.IsMainRank()) {
        LOG(INFO) << std::format(
            "Resume training from step {} with best_loss {:.6f}, last_lr {:.3e}, data_batch_idx {}", state.global_step,
            state.best_loss, state.last_lr, state.data_batch_idx);
    }

    return {global_step, best_loss, data_batch_idx};
}

} // namespace infini_train
