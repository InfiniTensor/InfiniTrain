#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "infini_train/include/checkpoint/shard_spec.h"
#include "infini_train/include/datatype.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::checkpoint {

// Physical write description for one local tensor shard.
struct WriteItem {
    std::string key;
    std::string filename;   // "model.ckpt" or "optimizer.ckpt"
    uint64_t offset = 0;    // Planned byte offset in the checkpoint file.
    uint64_t byte_size = 0; // Tensor payload size in bytes.
    DataType dtype = DataType::kFLOAT32;
    std::vector<int64_t> local_shape;
    std::vector<int64_t> global_offset;
    std::vector<int> axis_fragmentations;
    int rank = 0;
};

// Build the local tensor write layout from a ShardedStateDict.
class SavePlanner {
public:
    static std::vector<WriteItem> Plan(const ShardedStateDict &sd, int rank);
};

ShardedStateDict
BuildOptimizerShardedStateDict(const ShardedStateDict &model_state,
                               const std::unordered_map<std::string, std::shared_ptr<Tensor>> &optimizer_state);

// Return the number of payload bytes required by a tensor.
inline uint64_t TensorByteSize(DataType dtype, const std::vector<int64_t> &shape) {
    uint64_t numel = 1;
    for (auto d : shape) { numel *= static_cast<uint64_t>(d); }
    switch (dtype) {
    case DataType::kBFLOAT16:
    case DataType::kFLOAT16:
        return numel * 2;
    case DataType::kFLOAT32:
        return numel * 4;
    case DataType::kFLOAT64:
    case DataType::kINT64:
    case DataType::kUINT64:
        return numel * 8;
    case DataType::kINT32:
    case DataType::kUINT32:
        return numel * 4;
    case DataType::kINT16:
    case DataType::kUINT16:
        return numel * 2;
    case DataType::kINT8:
    case DataType::kUINT8:
    case DataType::kBOOL:
        return numel;
    default:
        return numel * 4;
    }
}

// Compute one rank's balanced interval, including non-divisible dimensions.
inline std::pair<int64_t, int64_t> GetRankSliceRange(int64_t global_size, int world_size, int rank) {
    int64_t per_rank = global_size / world_size;
    int64_t remainder = global_size % world_size;
    int64_t start = rank * per_rank + std::min<int64_t>(rank, remainder);
    int64_t local_size = per_rank + (rank < remainder ? 1 : 0);
    return {start, local_size};
}

} // namespace infini_train::checkpoint
