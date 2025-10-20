#include "infini_train/include/nn/parallel/distributed_data_parallel.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
namespace {
constexpr char kModuleName[] = "module";
} // namespace

DistributedDataParallel::Rank::Rank(int process_rank, int thread_rank, int process_size, int thread_size)
    : process_rank_(process_rank), thread_rank_(thread_rank), process_size_(process_size), thread_size_(thread_size) {}

int DistributedDataParallel::Rank::process_rank() const { return process_rank_; }
int DistributedDataParallel::Rank::thread_rank() const { return thread_rank_; }
int DistributedDataParallel::Rank::process_size() const { return process_size_; }
int DistributedDataParallel::Rank::thread_size() const { return thread_size_; }

int DistributedDataParallel::Rank::WorldSize() const { return process_size_ * thread_size_; }

bool DistributedDataParallel::Rank::IsDDP() const { return process_size_ * thread_size_ > 1; }

bool DistributedDataParallel::Rank::IsMainRank() const { return thread_rank_ == 0; }

DistributedDataParallel::DistributedDataParallel(std::shared_ptr<nn::Module> module, int device_id,
                                                 const ReducerOptions &opts,
                                                 std::shared_ptr<CommHookInterface> comm_hook) {
    for (auto &param : module->Parameters()) {
        CHECK_EQ(param->GetDevice()->Index(), device_id) << "All parameters must be on the same device as the module";
        // auto hook =
        // std::make_unique<infini_train::autograd::AllReducePostAccumulateHook>(function::ReduceOpType::kAvg);
        // param->RegisterPostAccumulateGradHook(std::move(hook));
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

    reducer_ = std::make_shared<Reducer>(std::move(params), bucket_indices, comm_hook, opts);
    reducer_->AttachHooksToParameters();
}

std::vector<std::shared_ptr<Tensor>>
DistributedDataParallel::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    if (reducer_) {
        reducer_->PrepareForBackward();
    }
    return modules_[kModuleName]->Forward(input_tensors);
}

void DistributedDataParallel::FinalizeBackward() const {
    if (reducer_) {
        reducer_->FinalizeBackward();
    }
}

} // namespace infini_train::nn::parallel
