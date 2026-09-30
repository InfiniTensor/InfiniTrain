#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "infini_train/include/checkpoint/checkpoint.h"
#include "infini_train/include/checkpoint/shard_spec.h"
#include "infini_train/include/datatype.h"

namespace infini_train::checkpoint {

// One storage-region transfer from a saved shard into a target local tensor.
struct ReadItem {
    std::string key;
    std::string filename;
    DataType dtype = DataType::kFLOAT32;
    std::vector<int64_t> global_shape;
    uint64_t byte_size = 0;
    uint64_t data_offset = 0;
    int shard_dim = -1;
    int64_t source_offset = 0;
    int64_t target_offset = 0;
    int64_t length = 0;
    std::vector<int64_t> source_shape;
};

// All reads required to materialize one target local tensor.
struct TargetTensorPlan {
    std::string key;
    DataType dtype = DataType::kFLOAT32;
    std::vector<int64_t> global_shape;
    std::vector<int64_t> target_shape;
    int shard_dim = -1;
    int64_t trailing_zero_fill = 0;
    std::vector<ReadItem> reads;
};

// Complete load plan for one rank.
struct LoadPlan {
    std::map<std::string, TargetTensorPlan> tensors;
};

class LoadPlanner {
public:
    // Compute saved-to-target overlaps from explicit global shard coordinates.
    static LoadPlan PlanReshard(const Checkpoint::CheckpointMetadata &metadata,
                                const ShardedStateDict &target_state_dict);
};

} // namespace infini_train::checkpoint
