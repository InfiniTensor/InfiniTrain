#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "infini_train/include/dataset.h"

namespace infini_train {
class Tensor;
}

namespace infini_train::example::cifar10 {

class CIFAR10Dataset : public infini_train::Dataset {
 public:
  static constexpr int64_t kChannels = 3;
  static constexpr int64_t kHeight = 32;
  static constexpr int64_t kWidth = 32;
  static constexpr int64_t kNumClasses = 10;

  CIFAR10Dataset(const std::string &dataset_dir, bool train,
                 bool augment = false);

  std::pair<std::shared_ptr<infini_train::Tensor>,
            std::shared_ptr<infini_train::Tensor>>
  operator[](size_t idx) const override;

  size_t Size() const override;

  // Builds a contiguous [B, 3, 32, 32] FP32 image tensor and [B] INT64 label
  // tensor directly for the requested sample indices.
  std::pair<std::shared_ptr<infini_train::Tensor>,
            std::shared_ptr<infini_train::Tensor>>
  GetBatch(const std::vector<size_t> &indices, bool apply_augment,
           std::mt19937 *rng) const;

  // Generates deterministic shuffled indices for an epoch and returns the
  // non-overlapping shard assigned to `rank` out of `world_size`.
  std::vector<size_t> GetShardedEpochIndices(int epoch, int rank,
                                             int world_size, uint64_t seed,
                                             bool shuffle) const;

 private:
  void LoadBinaryFile(const std::string &path);

  bool train_ = true;
  bool augment_ = false;
  std::vector<uint8_t> raw_images_;  // [N, 3, 32, 32] uint8
  std::vector<uint8_t> raw_labels_;  // [N] uint8
};

}  // namespace infini_train::example::cifar10
