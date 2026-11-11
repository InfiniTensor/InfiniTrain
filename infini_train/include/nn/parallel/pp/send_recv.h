#pragma once

#include <stdint.h>

#include <memory>
#include <vector>

namespace infini_train {
enum class DataType : int8_t;
class Tensor;
class Device;
}  // namespace infini_train

namespace infini_train::nn::parallel {

std::vector<std::shared_ptr<Tensor>> ISend(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors,
    Device target_device, int peer_rank,
    const std::vector<std::vector<int64_t>> &shape, DataType pipeline_dtype);

std::vector<std::shared_ptr<Tensor>> IRecv(
    const std::vector<std::shared_ptr<Tensor>> &outputs, Device src_device,
    int peer_rank, DataType pipeline_dtype);
}  // namespace infini_train::nn::parallel
