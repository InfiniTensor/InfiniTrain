#include "infini_train/include/nn/parallel/distributed_data_parallel.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
namespace {
constexpr char kModuleName[] = "module";
} // namespace

DistributedDataParallel::DistributedDataParallel(std::shared_ptr<nn::Module> module, int device_id) {
    for (auto &param : module->Parameters()) {
        auto device = param->GetDevice();
        CHECK_EQ(device->Index(), device_id) << "All parameters must be on the same device as the module";

        auto ddp_pg
            = ProcessGroupFactory::Instance()->Get(GetDataParallelProcessGroupName(device->rank().thread_rank()));
        auto hook = std::make_unique<infini_train::autograd::AllReducePostAccumulateHook>(function::ReduceOpType::kAvg,
                                                                                          ddp_pg);
        param->RegisterPostAccumulateGradHook(std::move(hook));
    }
    for (auto &buffer : module->Buffers()) {
        CHECK_EQ(buffer->GetDevice()->Index(), device_id) << "All buffers must be on the same device as the module";
    }
    modules_[kModuleName] = std::move(module);
}

std::vector<std::shared_ptr<Tensor>>
DistributedDataParallel::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return modules_[kModuleName]->Forward(input_tensors);
}

} // namespace infini_train::nn::parallel
