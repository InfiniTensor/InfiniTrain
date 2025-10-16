#pragma once

#include <string>

namespace infini_train::nn::parallel {
std::string GetDataParallelProcessGroupName(int thread_rank);

std::string GetTensorParallelProcessGroupName(int thread_rank);
} // namespace infini_train::nn::parallel
