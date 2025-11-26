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

DistributedDataParallel::DistributedDataParallel(std::shared_ptr<nn::Module> module, int device_id,
                                                 const ReducerOptions &opts) {
    for (auto &param : module->Parameters()) {
        auto device = param->GetDevice();
        CHECK_EQ(device->Index(), device_id) << "All parameters must be on the same device as the module";
        if (!opts.gradient_bucketing_enabled) {
            auto ddp_pg
                = ProcessGroupFactory::Instance()->Get(GetDataParallelProcessGroupName(device->rank().thread_rank()));
            auto hook = std::make_unique<infini_train::autograd::AllReducePostAccumulateHook>(
                function::ReduceOpType::kAvg, ddp_pg);
            param->RegisterPostAccumulateGradHook(std::move(hook));
        }
    }
    for (auto &buffer : module->Buffers()) {
        CHECK_EQ(buffer->GetDevice()->Index(), device_id) << "All buffers must be on the same device as the module";
    }
    modules_[kModuleName] = std::move(module);

    if (opts.gradient_bucketing_enabled) {
        // Bucket Assignment
        auto params = modules_[kModuleName]->Parameters();
        const size_t first_cap_bytes = opts.first_bucket_cap_mb * kBytesPerMB;
        const size_t normal_cap_bytes = opts.normal_bucket_cap_mb * kBytesPerMB;
        std::vector<size_t> bucket_size_limits = {first_cap_bytes, normal_cap_bytes};
        auto bucket_indices = ComputeBucketAssignmentBySize(params, bucket_size_limits);

        reducer_ = std::make_shared<Reducer>(params, bucket_indices, opts);
        reducer_->AttachHooksToParameters();
    }
}

std::vector<std::shared_ptr<Tensor>>
DistributedDataParallel::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    auto outputs = modules_[kModuleName]->Forward(input_tensors);
    if (reducer_) {
        reducer_->PrepareForBackward();
    }
    return outputs;
}

} // namespace infini_train::nn::parallel
