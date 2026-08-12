#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"

namespace infini_train::checkpoint {

struct ShardSegment {
    int64_t global_offset = 0;
    int64_t local_offset = 0;
    int64_t length = 0;

    bool operator==(const ShardSegment &other) const = default;
};

// Logical tensor shard metadata, aligned with Megatron-LM's ShardedTensor model.
struct ShardedTensor {
    std::string key;
    std::string local_key;
    DataType dtype = DataType::kFLOAT32;
    std::vector<int64_t> global_shape;
    std::vector<int64_t> local_shape;
    std::vector<int64_t> global_offset;
    std::vector<int> axis_fragmentations;
    // Optional disjoint regions along the single fragmented axis. This is used
    // by layouts such as rank-local [Q, K, V], which are not one contiguous
    // slice of the logical global [Q, K, V] tensor.
    std::vector<ShardSegment> segments;

    bool operator==(const ShardedTensor &other) const {
        return key == other.key && local_key == other.local_key && dtype == other.dtype
            && global_shape == other.global_shape && local_shape == other.local_shape
            && global_offset == other.global_offset && axis_fragmentations == other.axis_fragmentations
            && segments == other.segments;
    }
};

struct ShardedStateDict {
    std::map<std::string, ShardedTensor> tensors;

    void Merge(ShardedStateDict &&other) {
        for (auto &[key, info] : other.tensors) {
            const auto display_key = key;
            const auto [_, inserted] = tensors.emplace(std::move(key), std::move(info));
            CHECK(inserted) << "Duplicate sharded state-dict key: " << display_key;
        }
    }
};

} // namespace infini_train::checkpoint
