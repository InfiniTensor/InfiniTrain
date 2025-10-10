#include "infini_train/include/autograd/function_hook.h"

#include <memory>
#include <string>

#include "infini_train/include/nn/parallel/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
namespace {
std::string GetDataParallelFactoryName(const nn::parallel::DistributedDataParallel::Rank &rank) { return "DDP"; }
} // namespace

void AllReducePostAccumulateHook::operator()(const std::shared_ptr<Tensor> &tensor) {
    const auto *device = dynamic_cast<const CudaDevice *>(tensor->GetDevice());
    auto rank = device->rank();
    auto pg = infini_train::nn::parallel::ProcessGroupFactory::Instance()->Get(GetDataParallelFactoryName(rank));
    infini_train::nn::parallel::function::AllReduce(tensor, reduce_op_, pg);
}
} // namespace infini_train::autograd
