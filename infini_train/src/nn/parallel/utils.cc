#include "infini_train/include/nn/parallel/utils.h"

#include <filesystem>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

std::string GetDataParallelProcessGroupName(int global_rank) {
    return "DP" + std::to_string(global::GetGroupId(global::DP, global_rank));
}

std::string GetTensorParallelProcessGroupName(int global_rank) {
    return "TP" + std::to_string(global::GetGroupId(global::TP, global_rank));
}

std::string GetPipelineParallelProcessGroupName(int global_rank) {
    return "PP" + std::to_string(global::GetGroupId(global::PP, global_rank));
}

std::vector<int> GetDataParallelGroupRanks(int global_rank) { return global::GetGroupRanks(global::DP, global_rank); }

std::vector<int> GetTensorParallelGroupRanks(int global_rank) { return global::GetGroupRanks(global::TP, global_rank); }

std::vector<int> GetPipelineParallelGroupRanks(int global_rank) {
    return global::GetGroupRanks(global::PP, global_rank);
}

std::string TensorFileName(const std::string &name, bool tmp) {
    return std::format("{}_{}", name, tmp ? "tmp" : "done");
}

void WriteTensor(std::shared_ptr<Tensor> tensor, const std::string &path) {
    std::string tmp_path = TensorFileName(path, true);

    tensor->SaveAsNpy(tmp_path);

    std::rename(tmp_path.c_str(), TensorFileName(path).c_str());
}

std::shared_ptr<Tensor> ReadTensor(const std::string &path, const Device *device) {
    std::string tensor_path = TensorFileName(path);
    while (std::filesystem::exists(tensor_path) == false) {
        std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }

    auto tensor = Tensor::FromNpy(tensor_path, device);

    std::filesystem::remove(tensor_path);

    return tensor;
}

} // namespace infini_train::nn::parallel
