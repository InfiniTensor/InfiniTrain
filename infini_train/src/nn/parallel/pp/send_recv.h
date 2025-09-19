#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train::nn::pipeline {

std::vector<std::shared_ptr<Tensor>> ISend(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                           const Device *target_device, int cur_rank, int target_rank);

std::vector<std::shared_ptr<Tensor>> IRecv(const std::vector<std::shared_ptr<Tensor>> &outputs,
                                           const Device *src_device, int cur_rank, int src_rank);
} // namespace infini_train::nn::pipeline