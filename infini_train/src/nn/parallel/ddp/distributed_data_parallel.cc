#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
namespace {
constexpr char kModuleName[] = "module";
} // namespace

DistributedDataParallel::DistributedDataParallel(std::shared_ptr<nn::Module> module, int thread_rank,
                                                 const DistributedDataParallelConfig ddp_config)
    : ddp_config_(ddp_config),
      ddp_pg_(ProcessGroupFactory::Instance()->Get(GetDataParallelProcessGroupName(thread_rank))) {
    CHECK(ddp_config_.zero_stage >= 1 && ddp_config_.zero_stage <= 3)
        << "DistributedDataParallel: zero_stage must be in 1/2/3.";
    if (ddp_config_.zero_stage >= 3) {
        LOG(FATAL) << "DistributedDataParallel: ZeRO-3 is not implemented yet.";
    }
    if (!ddp_config_.use_distributed_optimizer && ddp_config_.zero_stage >= 1) {
        LOG(WARNING) << "DistributedDataParallel: zero_stage is ignored because "
                        "use_distributed_optimizer is false.";
        ddp_config_.zero_stage = 1;
    }

    for (auto &param : module->Parameters()) {
        auto device = param->GetDevice();
        CHECK_EQ(device->Index(), thread_rank) << "All parameters must be on the same device as the module";
        if (!ddp_config.gradient_bucketing_enabled && !ddp_config.use_distributed_optimizer) {
            auto hook = std::make_unique<infini_train::autograd::AllReducePostAccumulateHook>(
                function::ReduceOpType::kAvg, ddp_pg_);
            param->RegisterPostAccumulateGradHook(std::move(hook));
        }
    }
    for (auto &buffer : module->Buffers()) {
        CHECK_EQ(buffer->GetDevice()->Index(), thread_rank) << "All buffers must be on the same device as the module";
    }
    modules_[kModuleName] = std::move(module);

    if (ddp_config.use_distributed_optimizer) {
        BuildParamAndGradBuffers();
        RegisterBackwardHooks();
    } else if (ddp_config.gradient_bucketing_enabled) {
        // Bucket Assignment
        auto params = modules_[kModuleName]->Parameters();
        const size_t first_cap_bytes = ddp_config.first_bucket_cap_mb * kBytesPerMB;
        const size_t normal_cap_bytes = ddp_config.normal_bucket_cap_mb * kBytesPerMB;
        std::vector<size_t> bucket_size_limits = {first_cap_bytes, normal_cap_bytes};
        auto bucket_indices = ComputeBucketAssignmentBySize(params, bucket_size_limits);

        reducer_ = std::make_shared<Reducer>(params, bucket_indices, ddp_config);
        reducer_->AttachHooksToParameters();
    }
}

void DistributedDataParallel::BuildParamAndGradBuffers() {
    // (param_dtype, grad_dtype)
    using DTypePair = std::pair<DataType, DataType>;
    std::map<DTypePair, std::vector<std::shared_ptr<Tensor>>> dtype_to_params;

    for (auto param : modules_[kModuleName]->Parameters()) {
        if (!param->requires_grad()) {
            continue;
        }
        auto param_dtype = param->Dtype();
        auto grad_dtype = param->grad() ? param->grad()->Dtype() : param_dtype;
        dtype_to_params[{param_dtype, grad_dtype}].push_back(param);
    }

    param_grad_buffers_.clear();
    param_grad_buffers_.reserve(dtype_to_params.size());

    for (auto &kv : dtype_to_params) {
        auto [param_dtype, grad_dtype] = kv.first;
        auto param_list = kv.second;

        if (param_list.empty()) {
            continue;
        }

        // At the point, zero_stage is already aligned with use_distributed_optimizer.
        auto buffer = std::make_shared<ParamAndGradBuffer>(param_list, param_dtype, grad_dtype, ddp_pg_, ddp_config_);

        param_grad_buffers_.push_back(buffer);
    }

    // TODO(zbl): option for disable bucketing
    bucket_groups_ = PartitionBuckets(param_grad_buffers_, /*force_single_bucket_group=*/false);

    if (ddp_config_.use_distributed_optimizer && ddp_config_.overlap_param_gather) {
        auto num_bucket_groups = bucket_groups_.size();
        for (auto i = num_bucket_groups - 1; i > 0; --i) {
            bucket_groups_[i]->SetNextParamGatherBucketGroup(bucket_groups_[i - 1]);
        }
    }

    param_to_bucket_group_.clear();
    for (auto &group : bucket_groups_) {
        for (auto &bucket : group->buckets()) {
            for (auto &param : bucket->params()) {
                auto inserted = param_to_bucket_group_.emplace(param.get(), group).second;
                if (!inserted) {
                    LOG(FATAL) << "Parameter appears in more than one bucket group.";
                }
            }
        }
    }

    LOG(INFO) << "DDP BuildParamAndGradBuffers: "
              << "dtype_groups=" << dtype_to_params.size() << ", param_grad_buffers=" << param_grad_buffers_.size()
              << ", bucket_groups=" << bucket_groups_.size();
}

void DistributedDataParallel::RegisterBackwardHooks() {
    if (ddp_config_.zero_stage >= 2) {
        auto &module = modules_.at(kModuleName);
        for (auto &param : module->Parameters()) {
            if (!param->requires_grad()) {
                continue;
            }
            auto it = param_to_bucket_group_.find(param.get());
            if (it == param_to_bucket_group_.end()) {
                continue;
            }
            std::weak_ptr<ParamAndGradBucketGroup> weak_group = it->second;
            param->SetGradAccumulateBypass(
                [weak_group, param](const std::shared_ptr<Tensor> &grad_output, bool overwrite, float learning_rate) {
                    if (auto group = weak_group.lock()) {
                        group->AccumulateParamGrad(param, grad_output, overwrite, learning_rate);
                        if (group->config().overlap_grad_reduce) {
                            group->RegisterGradReady(param);
                        }
                        return true;
                    }
                    return false;
                });
        }
        return;
    }

    class DDPPostAccumulateHook final : public autograd::PostAccumulateGradHook {
    public:
        DDPPostAccumulateHook(DistributedDataParallel *ddp, const std::weak_ptr<Tensor> param)
            : ddp_(ddp), param_(param) {}

        void operator()(const std::shared_ptr<Tensor> &) override {
            if (auto param = param_.lock()) {
                ddp_->OnGradReady(param);
            }
        }

    private:
        DistributedDataParallel *ddp_;
        std::weak_ptr<Tensor> param_;
    };

    auto &module = modules_.at(kModuleName);
    for (auto &param : module->Parameters()) {
        if (!param->requires_grad()) {
            continue;
        }

        auto hook = std::make_unique<DDPPostAccumulateHook>(this, param);
        param->RegisterPostAccumulateGradHook(std::move(hook));
    }
}

void DistributedDataParallel::OnGradReady(const std::shared_ptr<Tensor> &param) {
    auto it = param_to_bucket_group_.find(param.get());
    if (it != param_to_bucket_group_.end()) {
        CHECK(param->requires_grad());
        if (ddp_config_.overlap_grad_reduce && (ddp_config_.zero_stage < 2)) {
            CHECK(param->grad()) << "param.grad being None is not safe when overlap_grad_reduce is True";
        }

        if (ddp_config_.overlap_grad_reduce) {
            it->second->RegisterGradReady(param);
        }
    }
}

std::vector<std::shared_ptr<Tensor>>
DistributedDataParallel::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    auto outputs = (*modules_[kModuleName])(input_tensors);
    if (reducer_) {
        reducer_->PrepareForBackward();
    }
    if (ddp_config_.use_distributed_optimizer) {
        for (auto buffer : param_grad_buffers_) { buffer->RebindGradViews(); }
    }
    return outputs;
}
} // namespace infini_train::nn::parallel
