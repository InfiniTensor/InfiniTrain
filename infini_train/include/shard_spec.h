#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "infini_train/include/datatype.h"

namespace infini_train {

class Tensor;

struct ShardSegment {
  int64_t global_offset = 0;
  int64_t local_offset = 0;
  int64_t length = 0;

  bool operator==(const ShardSegment &other) const = default;
};

// Logical tensor shard metadata, aligned with Megatron-LM's ShardedTensor
// model.
struct ShardedTensor {
  std::string key;
  std::string local_key;
  const Tensor *local_tensor =
      nullptr;  // Checkpoint-only identity; not serialized.
  DataType dtype = DataType::kFLOAT32;
  std::vector<int64_t> global_shape;
  std::vector<int64_t> local_shape;
  std::vector<int64_t> global_offset;
  std::vector<int> axis_fragmentations;
  bool allow_shape_mismatch = false;
  // Optional disjoint regions along the single fragmented axis. This is used
  // by layouts such as rank-local [Q, K, V], which are not one contiguous
  // slice of the logical global [Q, K, V] tensor.
  std::vector<ShardSegment> segments;

  bool operator==(const ShardedTensor &other) const;
};

inline ShardedTensor MakeShardedTensor(std::string key, DataType dtype,
                                       std::vector<int64_t> shape) {
  const auto ndim = shape.size();
  return {.key = std::move(key),
          .dtype = dtype,
          .global_shape = shape,
          .local_shape = std::move(shape),
          .global_offset = std::vector<int64_t>(ndim, 0),
          .axis_fragmentations = std::vector<int>(ndim, 1)};
}

struct ShardedStateDict {
  std::map<std::string, ShardedTensor> tensors;

  void Merge(ShardedStateDict &&other);
};

}  // namespace infini_train
