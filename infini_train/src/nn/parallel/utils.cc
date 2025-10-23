#include "infini_train/include/nn/parallel/utils.h"

#include "infini_train/include/nn/parallel/global.h"

namespace infini_train::nn::parallel {

std::string GetDataParallelProcessGroupName(int thread_rank) {
    // FIXME(zbl): Need a layout definition of parallel coords
    return "DP" + std::to_string(thread_rank % global::GetTensorParallelSize());
}

std::string GetTensorParallelProcessGroupName(int thread_rank) {
    // FIXME(zbl): Need a layout definition of parallel coords
    return "TP" + std::to_string(thread_rank / global::GetTensorParallelSize());
}

} // namespace infini_train::nn::parallel
