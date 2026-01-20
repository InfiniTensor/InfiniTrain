#pragma once

#include <memory>
#include <string>
#include <vector>

namespace infini_train {
class Tensor;
class Device;
} // namespace infini_train

namespace infini_train::nn::parallel {
std::string GetDataParallelProcessGroupName(int global_rank);

std::string GetTensorParallelProcessGroupName(int global_rank);

std::string GetPipelineParallelProcessGroupName(int global_rank);

std::vector<int> GetDataParallelGroupRanks(int global_rank);

std::vector<int> GetTensorParallelGroupRanks(int global_rank);

std::vector<int> GetPipelineParallelGroupRanks(int global_rank);

// heterogeneous shared path
static constexpr char kSharedPathPrefix[] = "/nfs/duanchenjie/InfiniTrain_grad_tensors";

std::string TensorFileName(const std::string &name, bool tmp = false);

void WriteTensor(std::shared_ptr<Tensor> tensor, const std::string &path);

std::shared_ptr<Tensor> ReadTensor(const std::string &path, const Device *device);
} // namespace infini_train::nn::parallel
