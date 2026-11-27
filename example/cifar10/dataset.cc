#include "example/cifar10/dataset.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/tensor.h"

namespace infini_train::example::cifar10 {
namespace {

constexpr size_t kImageBytes = 3 * 32 * 32;
constexpr size_t kRecordBytes = 1 + kImageBytes;
constexpr float kMean[3] = {0.4914f, 0.4822f, 0.4465f};
constexpr float kStd[3] = {0.2023f, 0.1994f, 0.2010f};
constexpr int kCropPad = 4;

std::string ResolveDataDir(const std::string &dataset_dir) {
  namespace fs = std::filesystem;
  const fs::path base(dataset_dir);
  if (fs::exists(base / "data_batch_1.bin")) {
    return base.string();
  }
  if (fs::exists(base / "cifar-10-batches-bin" / "data_batch_1.bin")) {
    return (base / "cifar-10-batches-bin").string();
  }
  return base.string();
}

void WriteNormalizedImage(const uint8_t *src_chw, float *dst_chw,
                          bool apply_augment, std::mt19937 *rng) {
  int y_offset = 0;
  int x_offset = 0;
  bool hflip = false;
  if (apply_augment && rng != nullptr) {
    std::uniform_int_distribution<int> crop_dist(0, 2 * kCropPad);
    std::uniform_int_distribution<int> flip_dist(0, 1);
    y_offset = crop_dist(*rng);
    x_offset = crop_dist(*rng);
    hflip = (flip_dist(*rng) == 1);
  }

  for (int c = 0; c < 3; ++c) {
    const float mean = kMean[c];
    const float inv_std = 1.0f / kStd[c];
    const uint8_t *src_c = src_chw + c * 32 * 32;
    float *dst_c = dst_chw + c * 32 * 32;
    for (int y = 0; y < 32; ++y) {
      const int sy = apply_augment ? (y + y_offset - kCropPad) : y;
      for (int x = 0; x < 32; ++x) {
        const int out_x = hflip ? (31 - x) : x;
        const int sx = apply_augment ? (x + x_offset - kCropPad) : x;
        float val = 0.0f;
        if (sy >= 0 && sy < 32 && sx >= 0 && sx < 32) {
          val = static_cast<float>(src_c[sy * 32 + sx]) / 255.0f;
        }
        dst_c[y * 32 + out_x] = (val - mean) * inv_std;
      }
    }
  }
}

}  // namespace

CIFAR10Dataset::CIFAR10Dataset(const std::string &dataset_dir, bool train,
                               bool augment)
    : train_(train), augment_(augment) {
  const std::string resolved = ResolveDataDir(dataset_dir);
  if (train_) {
    raw_images_.reserve(50000 * kImageBytes);
    raw_labels_.reserve(50000);
    for (int i = 1; i <= 5; ++i) {
      LoadBinaryFile(resolved + "/data_batch_" + std::to_string(i) + ".bin");
    }
  } else {
    raw_images_.reserve(10000 * kImageBytes);
    raw_labels_.reserve(10000);
    LoadBinaryFile(resolved + "/test_batch.bin");
  }
  CHECK(!raw_labels_.empty())
      << "No CIFAR-10 records loaded from: " << resolved;
}

void CIFAR10Dataset::LoadBinaryFile(const std::string &path) {
  CHECK(std::filesystem::exists(path)) << "CIFAR-10 binary not found: " << path;
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  CHECK(ifs.is_open()) << "Failed to open CIFAR-10 binary: " << path;
  const size_t file_size = static_cast<size_t>(ifs.tellg());
  CHECK_EQ(file_size % kRecordBytes, 0)
      << "Unexpected CIFAR-10 binary file size: " << file_size;
  ifs.seekg(0, std::ios::beg);

  const size_t num_records = file_size / kRecordBytes;
  std::vector<uint8_t> buf(file_size);
  ifs.read(reinterpret_cast<char *>(buf.data()),
           static_cast<std::streamsize>(file_size));

  const size_t old_count = raw_labels_.size();
  raw_labels_.resize(old_count + num_records);
  raw_images_.resize((old_count + num_records) * kImageBytes);

  for (size_t i = 0; i < num_records; ++i) {
    const uint8_t *rec = buf.data() + i * kRecordBytes;
    raw_labels_[old_count + i] = rec[0];
    std::memcpy(raw_images_.data() + (old_count + i) * kImageBytes, rec + 1,
                kImageBytes);
  }
}

std::pair<std::shared_ptr<infini_train::Tensor>,
          std::shared_ptr<infini_train::Tensor>>
CIFAR10Dataset::operator[](size_t idx) const {
  CHECK_LT(idx, raw_labels_.size());
  auto img = std::make_shared<infini_train::Tensor>(
      std::vector<int64_t>{kChannels, kHeight, kWidth},
      infini_train::DataType::kFLOAT32);
  auto lbl = std::make_shared<infini_train::Tensor>(
      std::vector<int64_t>{1}, infini_train::DataType::kINT64);
  WriteNormalizedImage(raw_images_.data() + idx * kImageBytes,
                       static_cast<float *>(img->DataPtr()), false, nullptr);
  *static_cast<int64_t *>(lbl->DataPtr()) =
      static_cast<int64_t>(raw_labels_[idx]);
  return {img, lbl};
}

size_t CIFAR10Dataset::Size() const { return raw_labels_.size(); }

std::pair<std::shared_ptr<infini_train::Tensor>,
          std::shared_ptr<infini_train::Tensor>>
CIFAR10Dataset::GetBatch(const std::vector<size_t> &indices, bool apply_augment,
                         std::mt19937 *rng) const {
  const int64_t bs = static_cast<int64_t>(indices.size());
  auto images = std::make_shared<infini_train::Tensor>(
      std::vector<int64_t>{bs, kChannels, kHeight, kWidth},
      infini_train::DataType::kFLOAT32);
  auto labels = std::make_shared<infini_train::Tensor>(
      std::vector<int64_t>{bs}, infini_train::DataType::kINT64);

  float *img_ptr = static_cast<float *>(images->DataPtr());
  int64_t *lbl_ptr = static_cast<int64_t *>(labels->DataPtr());

  // Generate per-sample seeds deterministically so OpenMP augmentation is
  // reproducible and thread-safe.
  std::vector<uint32_t> sample_seeds(indices.size(), 0);
  if (apply_augment && rng != nullptr) {
    for (size_t i = 0; i < indices.size(); ++i) {
      sample_seeds[i] = (*rng)();
    }
  }

#pragma omp parallel for
  for (int64_t i = 0; i < bs; ++i) {
    const size_t sample_idx = indices[static_cast<size_t>(i)];
    CHECK_LT(sample_idx, raw_labels_.size());
    std::mt19937 local_rng(sample_seeds[static_cast<size_t>(i)]);
    WriteNormalizedImage(raw_images_.data() + sample_idx * kImageBytes,
                         img_ptr + i * kImageBytes, apply_augment,
                         apply_augment ? &local_rng : nullptr);
    lbl_ptr[i] = static_cast<int64_t>(raw_labels_[sample_idx]);
  }
  return {images, labels};
}

std::vector<size_t> CIFAR10Dataset::GetShardedEpochIndices(int epoch, int rank,
                                                           int world_size,
                                                           uint64_t seed,
                                                           bool shuffle) const {
  CHECK_GT(world_size, 0);
  CHECK_GE(rank, 0);
  CHECK_LT(rank, world_size);

  const size_t total = raw_labels_.size();
  std::vector<size_t> all_indices(total);
  std::iota(all_indices.begin(), all_indices.end(), 0);
  if (shuffle) {
    std::mt19937 rng(
        static_cast<uint32_t>(seed + static_cast<uint64_t>(epoch)));
    std::shuffle(all_indices.begin(), all_indices.end(), rng);
  }

  // Non-overlapping strided partition covering the full dataset.
  // For exact equal step counts across ranks in DDP, take floor(total / W)
  // per rank when total % W != 0, or exact total / W when divisible (50000 % 2
  // == 0).
  const size_t per_rank = total / static_cast<size_t>(world_size);
  std::vector<size_t> shard;
  shard.reserve(per_rank);
  for (size_t i = 0; i < per_rank; ++i) {
    shard.push_back(all_indices[i * static_cast<size_t>(world_size) +
                                static_cast<size_t>(rank)]);
  }
  return shard;
}

}  // namespace infini_train::example::cifar10
