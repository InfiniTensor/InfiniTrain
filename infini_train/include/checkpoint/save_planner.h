#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "infini_train/include/checkpoint/shard_spec.h"
#include "infini_train/include/datatype.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::checkpoint {

inline constexpr std::string_view kAdamOptimizerPrefix = "adam.";
inline constexpr std::string_view kAdamFirstMomentPrefix = "adam.m.";
inline constexpr std::string_view kAdamSecondMomentPrefix = "adam.v.";
inline constexpr std::string_view kAdamStepKey = "adam.t";

// Physical write description for one local tensor shard.
struct WriteItem {
    std::string key;
    std::string filename;   // "model.ckpt" or "optimizer.ckpt"
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
    return numel * static_cast<uint64_t>(kDataTypeToSize.at(dtype));
}

} // namespace infini_train::checkpoint
