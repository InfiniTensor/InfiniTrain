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
        CHECK_EQ(param->GetDevice()->Index(), device_id) << "All parameters must be on the same device as the module";
    }
    for (auto &buffer : module->Buffers()) {
        CHECK_EQ(buffer->GetDevice()->Index(), device_id) << "All buffers must be on the same device as the module";
    }
    modules_[kModuleName] = std::move(module);

    // Bucket Assignment
    auto params = modules_[kModuleName]->Parameters();
    const size_t first_cap_bytes = opts.first_bucket_cap_mb * 1024ULL * 1024ULL;
    const size_t normal_cap_bytes = opts.normal_bucket_cap_mb * 1024ULL * 1024ULL;
    std::vector<size_t> bucket_size_limits = {first_cap_bytes, normal_cap_bytes};
    auto bucket_indices = ComputeBucketAssignmentBySize(params, bucket_size_limits);

    reducer_ = std::make_shared<Reducer>(params, bucket_indices, opts);
    reducer_->AttachHooksToParameters();
}

std::vector<std::shared_ptr<Tensor>>
DistributedDataParallel::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    if (reducer_) {
        reducer_->PrepareForBackward();
    }
    return modules_[kModuleName]->Forward(input_tensors);
}

} // namespace infini_train::nn::parallel
