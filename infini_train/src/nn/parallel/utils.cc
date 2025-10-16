#include "infini_train/include/nn/parallel/utils.h"

#include "infini_train/include/nn/parallel/global.h"

namespace infini_train::nn::parallel {

std::string GetDataParallelProcessGroupName(int thread_rank) {
    return "DP" + std::to_string(thread_rank / global::GetDataParallelSize());
}

std::string GetTensorParallelProcessGroupName(int thread_rank) {
    return "TP" + std::to_string(thread_rank % global::GetTensorParallelSize());
}

} // namespace infini_train::nn::parallel
