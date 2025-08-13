#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train::nn::pipeline {

void Isend(const std::vector<std::shared_ptr<Tensor>> &inputs, int destinations);
void Irecv(const std::vector<std::shared_ptr<Tensor>> &inputs, int source);

} // namespace infini_train::nn::pipeline