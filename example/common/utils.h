#pragma once

#include <vector>

namespace infini_train {

float ConvertBF16ToFloat(void *ptr);

std::vector<int> GetDataParallelGroupRanks(int rank);

std::vector<int> GetTensorParallelGroupRanks(int rank);

} // namespace infini_train
