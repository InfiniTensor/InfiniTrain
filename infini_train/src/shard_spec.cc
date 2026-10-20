#include "infini_train/include/shard_spec.h"

#include <ostream>
#include <utility>

#include "glog/logging.h"

namespace infini_train {

bool ShardedTensor::operator==(const ShardedTensor &other) const {
  return key == other.key && local_key == other.local_key &&
         dtype == other.dtype && global_shape == other.global_shape &&
         local_shape == other.local_shape &&
         global_offset == other.global_offset &&
         axis_fragmentations == other.axis_fragmentations &&
         allow_shape_mismatch == other.allow_shape_mismatch &&
         segments == other.segments;
}

void ShardedStateDict::Merge(ShardedStateDict &&other) {
  for (auto &[key, info] : other.tensors) {
    const auto display_key = key;
    const auto [_, inserted] = tensors.emplace(std::move(key), std::move(info));
    CHECK(inserted) << "Duplicate sharded state-dict key: " << display_key;
  }
}

}  // namespace infini_train
