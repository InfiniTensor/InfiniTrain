#pragma once

#include <memory>
#include <vector>

namespace infini_train {
class Tensor;
class Device;
} // namespace infini_train

namespace infini_train::nn::parallel {

std::vector<std::shared_ptr<Tensor>> ISend(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                           const Device *target_device, int cur_rank, int peer_rank,
                                           const std::vector<std::vector<int64_t>> &shape);

std::vector<std::shared_ptr<Tensor>> IRecv(const std::vector<std::shared_ptr<Tensor>> &outputs,
                                           const Device *src_device, int cur_rank, int peer_rank);
} // namespace infini_train::nn::parallel
