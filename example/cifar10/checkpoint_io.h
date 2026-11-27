#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace infini_train {
class Optimizer;
class Tensor;
namespace nn {
class Module;
}  // namespace nn
}  // namespace infini_train

namespace infini_train::example::cifar10 {

struct CheckpointMetadata {
  int64_t epoch = 0;
  int64_t global_step = 0;
  float learning_rate = 0.0f;
};

// Saves/loads named tensor maps in a simple portable binary format:
//   magic[4] = "IFTB"
//   uint32_t num_tensors
//   For each tensor:
//     uint32_t name_len, char name[name_len]
//     int32_t dtype (0=FP32, 1=INT64, 2=UINT8)
//     uint32_t ndim, int64_t dims[ndim]
//     uint64_t num_bytes, uint8_t data[num_bytes]
void SaveTensorMap(
    const std::string &path,
    const std::unordered_map<std::string, std::shared_ptr<infini_train::Tensor>>
        &tensors);

std::unordered_map<std::string, std::shared_ptr<infini_train::Tensor>>
LoadTensorMap(const std::string &path);

void SaveTrainingCheckpoint(const std::string &path,
                            const infini_train::nn::Module &model,
                            const infini_train::Optimizer &optimizer,
                            const CheckpointMetadata &meta);

CheckpointMetadata LoadTrainingCheckpoint(const std::string &path,
                                          infini_train::nn::Module *model,
                                          infini_train::Optimizer *optimizer);

}  // namespace infini_train::example::cifar10
