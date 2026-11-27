#include "example/cifar10/checkpoint_io.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train::example::cifar10 {
namespace {

using namespace infini_train;

constexpr char kMagic[4] = {'I', 'F', 'T', 'B'};

int32_t DtypeToCode(DataType dt) {
  switch (dt) {
    case DataType::kFLOAT32:
      return 0;
    case DataType::kINT64:
      return 1;
    case DataType::kUINT8:
      return 2;
    default:
      LOG(FATAL) << "Unsupported dtype in checkpoint_io: "
                 << static_cast<int>(dt);
      return -1;
  }
}

DataType CodeToDtype(int32_t code) {
  switch (code) {
    case 0:
      return DataType::kFLOAT32;
    case 1:
      return DataType::kINT64;
    case 2:
      return DataType::kUINT8;
    default:
      LOG(FATAL) << "Unsupported dtype code in checkpoint_io: " << code;
      return DataType::kFLOAT32;
  }
}

}  // namespace

void SaveTensorMap(
    const std::string &path,
    const std::unordered_map<std::string, std::shared_ptr<Tensor>> &tensors) {
  const std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
  std::ofstream ofs(path, std::ios::binary);
  CHECK(ofs.is_open()) << "Failed to open file for writing: " << path;

  std::vector<std::string> keys;
  keys.reserve(tensors.size());
  for (const auto &[k, _] : tensors) {
    keys.push_back(k);
  }
  std::sort(keys.begin(), keys.end());

  ofs.write(kMagic, 4);
  const uint32_t count = static_cast<uint32_t>(keys.size());
  ofs.write(reinterpret_cast<const char *>(&count), sizeof(count));

  for (const auto &key : keys) {
    const auto &t = tensors.at(key);
    auto cpu_t = t->To(Device());
    const uint32_t name_len = static_cast<uint32_t>(key.size());
    const int32_t dtype_code = DtypeToCode(cpu_t.Dtype());
    const uint32_t ndim = static_cast<uint32_t>(cpu_t.Dims().size());
    const uint64_t nbytes = static_cast<uint64_t>(cpu_t.SizeInBytes());

    ofs.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
    ofs.write(key.data(), static_cast<std::streamsize>(name_len));
    ofs.write(reinterpret_cast<const char *>(&dtype_code), sizeof(dtype_code));
    ofs.write(reinterpret_cast<const char *>(&ndim), sizeof(ndim));
    if (ndim > 0) {
      ofs.write(reinterpret_cast<const char *>(cpu_t.Dims().data()),
                static_cast<std::streamsize>(ndim * sizeof(int64_t)));
    }
    ofs.write(reinterpret_cast<const char *>(&nbytes), sizeof(nbytes));
    if (nbytes > 0) {
      ofs.write(static_cast<const char *>(cpu_t.DataPtr()),
                static_cast<std::streamsize>(nbytes));
    }
  }
  CHECK(ofs.good()) << "Error writing tensor map to: " << path;
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> LoadTensorMap(
    const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  CHECK(ifs.is_open()) << "Failed to open tensor map: " << path;

  char magic[4];
  ifs.read(magic, 4);
  CHECK(std::memcmp(magic, kMagic, 4) == 0)
      << "Invalid tensor map magic in: " << path;

  uint32_t count = 0;
  ifs.read(reinterpret_cast<char *>(&count), sizeof(count));

  std::unordered_map<std::string, std::shared_ptr<Tensor>> result;
  result.reserve(count);

  for (uint32_t i = 0; i < count; ++i) {
    uint32_t name_len = 0;
    ifs.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
    std::string name(name_len, '\0');
    ifs.read(name.data(), static_cast<std::streamsize>(name_len));

    int32_t dtype_code = 0;
    uint32_t ndim = 0;
    ifs.read(reinterpret_cast<char *>(&dtype_code), sizeof(dtype_code));
    ifs.read(reinterpret_cast<char *>(&ndim), sizeof(ndim));

    std::vector<int64_t> dims(ndim);
    if (ndim > 0) {
      ifs.read(reinterpret_cast<char *>(dims.data()),
               static_cast<std::streamsize>(ndim * sizeof(int64_t)));
    }
    uint64_t nbytes = 0;
    ifs.read(reinterpret_cast<char *>(&nbytes), sizeof(nbytes));

    auto tensor = std::make_shared<Tensor>(dims, CodeToDtype(dtype_code));
    CHECK_EQ(static_cast<uint64_t>(tensor->SizeInBytes()), nbytes)
        << "Byte size mismatch for tensor: " << name;
    if (nbytes > 0) {
      ifs.read(static_cast<char *>(tensor->DataPtr()),
               static_cast<std::streamsize>(nbytes));
    }
    result.emplace(std::move(name), std::move(tensor));
  }
  CHECK(ifs.good()) << "Error reading tensor map from: " << path;
  return result;
}

void SaveTrainingCheckpoint(const std::string &path, const nn::Module &model,
                            const Optimizer &optimizer,
                            const CheckpointMetadata &meta) {
  std::unordered_map<std::string, std::shared_ptr<Tensor>> all;
  for (const auto &[k, v] : model.StateDict()) {
    all.emplace("model." + k, v);
  }
  for (const auto &[k, v] : optimizer.StateDict()) {
    all.emplace("opt." + k, v);
  }
  auto epoch_t =
      std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kINT64);
  *static_cast<int64_t *>(epoch_t->DataPtr()) = meta.epoch;
  all.emplace("meta.epoch", epoch_t);

  auto step_t =
      std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kINT64);
  *static_cast<int64_t *>(step_t->DataPtr()) = meta.global_step;
  all.emplace("meta.global_step", step_t);

  auto lr_t =
      std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32);
  *static_cast<float *>(lr_t->DataPtr()) = meta.learning_rate;
  all.emplace("meta.learning_rate", lr_t);

  SaveTensorMap(path, all);
}

CheckpointMetadata LoadTrainingCheckpoint(const std::string &path,
                                          nn::Module *model,
                                          Optimizer *optimizer) {
  auto all = LoadTensorMap(path);
  std::unordered_map<std::string, std::shared_ptr<Tensor>> model_sd;
  std::unordered_map<std::string, std::shared_ptr<Tensor>> opt_sd;

  constexpr char kModelPrefix[] = "model.";
  constexpr char kOptPrefix[] = "opt.";
  for (const auto &[k, v] : all) {
    if (k.starts_with(kModelPrefix)) {
      model_sd.emplace(k.substr(sizeof(kModelPrefix) - 1), v);
    } else if (k.starts_with(kOptPrefix)) {
      opt_sd.emplace(k.substr(sizeof(kOptPrefix) - 1), v);
    }
  }
  if (model != nullptr) {
    model->LoadStateDict(model_sd);
  }
  if (optimizer != nullptr && !opt_sd.empty()) {
    optimizer->LoadStateDict(opt_sd);
  }

  CheckpointMetadata meta;
  if (all.contains("meta.epoch")) {
    meta.epoch = *static_cast<const int64_t *>(all.at("meta.epoch")->DataPtr());
  }
  if (all.contains("meta.global_step")) {
    meta.global_step =
        *static_cast<const int64_t *>(all.at("meta.global_step")->DataPtr());
  }
  if (all.contains("meta.learning_rate")) {
    meta.learning_rate =
        *static_cast<const float *>(all.at("meta.learning_rate")->DataPtr());
  }
  return meta;
}

}  // namespace infini_train::example::cifar10
