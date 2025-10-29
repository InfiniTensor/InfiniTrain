#pragma once

#include <vector>

namespace infini_train {

float ConvertBF16ToFloat(void *ptr);

std::vector<int> GetDataParallelGroupRanks(int rank);

std::vector<int> GetTensorParallelGroupRanks(int rank);

std::vector<int> GetPipelineParallelGroupRanks(int rank);
} // namespace infini_train
