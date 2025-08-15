#include "infini_train/include/nn/parallel/tensor_parallel.h"

#include <format>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/nn/parallel/scatter_gather.h"

namespace infini_train::nn::parallel {

namespace {
// Comm Kernel Call Functions
std::shared_ptr<Tensor> GatherAlongLastDim(const std::shared_ptr<Tensor> &tensor, const TensorParallelGroup &tp_group) {
    int world_size = tp_group.WorldSize();
    CHECK_GT(world_size, 0) << "Tensor Parallel group not initialized";
    if (world_size == 1) {
        // Bypass the function if we are using only 1 GPU.
        return {tensor};
    }

    auto it = std::find(tp_group.devices.begin(), tp_group.devices.end(), tensor->GetDevice());
    CHECK(it != tp_group.devices.end()) << "Tensor device not in TensorParallelGroup";

    std::vector<int64_t> output_shape = tensor->Dims();
    out_shape.back() *= world_size;
    auto gathered_output = std::make_shared<Tensor>(output_shape, tensor->Dtype(), device);

#ifdef USE_NCCL
    auto kernel = Dispatcher::Instance().GetKernel({input_device_->Type(), "CommNcclAllGather"});
    return {kernel.Call<void, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>(gathered_output, tensor)};
#else
    LOG(FATAL) << "AllGather requires NCCL enabled";
    return {};
#endif
    // AllGather gather along dim 0 by default
    auto output_list = gathered_output->split(tensor->Dims()[0], 0);
    auto output = nn::functional::Concat(output_list, -1);

    return output;
}

std::shared_ptr<Tensor> SplitAlongLastDim(const std::shared_ptr<Tensor> &tensor, const TensorParallelGroup &tp_group) {
    int world_size = tp_group.WorldSize();
    CHECK_GT(world_size, 0) << "Tensor Parallel group not initialized";
    if (world_size == 1) {
        // Bypass the function if we are using only 1 GPU.
        return {tensor};
    }
    auto last_dim_size = tensor->Dims().back() / world_size;
    auto shards = tensor->Split(last_dim_size, -1);
    auto rank = tp_group_.RankOf(input_device_);
    return shards[rank]->Contiguous();
}

std::shared_ptr<Tensor> Reduce(const std::shared_ptr<Tensor> &tensor, const TensorParallelGroup &tp_group) {
    int world_size = tp_group.WorldSize();
    CHECK_GT(world_size, 0) << "Tensor Parallel group not initialized";
    if (world_size == 1) {
        // Bypass the function if we are using only 1 GPU.
        return {tensor};
    }

    auto it = std::find(tp_group.devices.begin(), tp_group.devices.end(), tensor->GetDevice());
    CHECK(it != tp_group.devices.end()) << "Tensor device not in TensorParallelGroup";

#ifdef USE_NCCL
    auto kernel = Dispatcher::Instance().GetKernel({input_device_->Type(), "CommNcclAllReduceLocal"});
    kernel.Call<void, std::shared_ptr<Tensor>>(tensor);
    return {tensor};
#else
    LOG(FATAL) << "AllReduce requires NCCL enabled";
    return {};
#endif
}

// Autograd Function definitions
class CopyToTPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "CopyToTPRegionFunction";

    explicit CopyToTPRegion(TensorParallelGroup tp_group) : autograd::Function(kType), tp_group_(tp_group) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        // Each rank should get the same `input`
        // No need to perform Broadcast-like copy
        return {input_tensors[0]};
    };

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        return Reduce(grad_outputs[0], tp_group_);
    };

private:
    TensorParallelGroup tp_group_;
}

class GatherFromTPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "GatherFromTPRegionFunction";

    explicit GatherFromTPRegion(TensorParallelGroup tp_group) : autograd::Function(kType), tp_group_(tp_group) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        return GatherAlongLastDim(input_tensors[0], tp_group_);
    };

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        // Each rank should get the same `full_grad_output`
        // Perform local split to get corresponding shard
        return SplitAlongLastDim(grad_outputs[0], tp_group_);
    };

private:
    TensorParallelGroup tp_group_;
}

class ScatterToTPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "ScatterToTPRegionFunction";

    explicit ScatterToTPRegion(TensorParallelGroup tp_group) : autograd::Function(kType), tp_group_(tp_group) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        // Each rank should get the same `input`
        // Perform local split to get corresponding shard
        return SplitAlongLastDim(input_tensors[0], tp_group_);
    };

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        return GatherAlongLastDim(grad_outputs[0], tp_group_);
    };

private:
    TensorParallelGroup tp_group_;
}

class ReduceFromTPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "ReduceFromTPRegionFunction";

    explicit ReduceFromTPRegion(TensorParallelGroup tp_group) : autograd::Function(kType), tp_group_(tp_group) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        // Perform AllReduceSum to get full output
        return Reduce(input_tensors[0], tp_group_);
    };

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        return {grad_outputs[0]};
    };

private:
    TensorParallelGroup tp_group_;
}

// Comm Helper Functions
std::shared_ptr<Tensor>
CopyToTPRegion(const std::shared_ptr<Tensor> &input, TensorParallelGroup tp_group) {
    return std::make_shared<autograd::CopyToTPRegion>(tp_group)->Apply({input})[0];
}

std::shared_ptr<Tensor> GatherFromTPRegion(const std::shared_ptr<Tensor> &input, TensorParallelGroup tp_group) {
    return std::make_shared<autograd::GatherFromTPRegion>(tp_group)->Apply({input})[0];
}

std::shared_ptr<Tensor> ScatterToTPRegion(const std::shared_ptr<Tensor> &input, TensorParallelGroup tp_group) {
    return std::make_shared<autograd::ScatterToTPRegion>(tp_group)->Apply({input})[0];
}

std::shared_ptr<Tensor> ReduceFromTPRegion(const std::shared_ptr<Tensor> &input, TensorParallelGroup tp_group) {
    return std::make_shared<autograd::ReduceFromTPRegion>(tp_group)->Apply({input})[0];
}

} // namespace

ColumnParallelLinear::ColumnParallelLinear(int64_t in_features, int64_t out_features, bool bias,
                                           TensorParallelGroup tp_group, bool gather_output, bool input_is_parallel,
                                           bool skip_bias_add)
    : Module(kType), tp_group_(tp_group), bias_(bias), gather_output_(gather_output),
      input_is_parallel_(input_is_parallel), skip_bias_add_(skip_bias_add) {
    CHECK_GT(tp_group.WorldSize(), 0) << "No available devices found";
    CHECK_EQ(out_features % tp_group_.WorldSize(), 0)
        << "out_features must be divisible by TP world size for ColumnParallel";

    output_size_per_partition_ = out_features / tp_group_.WorldSize();

    // init params shards on local rank
    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{output_size_per_partition_, in_features}, DataType::kFLOAT32,
                                   device_)
              ->RequiresGrad();
    if (bias) {
        parameters_[kParamBiasName]
            = std::make_shared<Tensor>(std::vector<int64_t>{output_size_per_partition_}, DataType::kFLOAT32, device_)
                  ->RequiresGrad();
    }
}

std::vector<std::shared_ptr<Tensor>>
ColumnParallelLinear::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1) << "ColumnParallelLinear takes exactly one input";
    auto input = input_is_parallel_ ? input_tensors[0] : CopyToTPRegion(input_tensors[0], tp_group_);

    auto sharded_output = std::make_shared<autograd::Linear>()->Apply(
        (bias_ && !skip_bias_add_)
            ? std::vector<std::shared_ptr<Tensor>>{input, parameters_.at(kParamWeightName), parameters_[kParamBiasName]}
            : std::vector<std::shared_ptr<Tensor>>{input, parameters_.at(kParamWeightName)})[0];

    std::shared_ptr<Tensor> output = gather_output_ ? GatherFromTPRegion(sharded_output, tp_group_) : sharded_output;

    return skip_bias_add_ ? {output, bias_ ? parameters_.at(kParamBiasName) : nullptr} : {output};
}

RowParallelLinear::RowParallelLinear(int64_t in_features, int64_t out_features, bool bias, TensorParallelGroup tp_group,
                                     bool reduce_output, bool input_is_parallel, bool skip_bias_add)
    : Module(kType), tp_group_(tp_group), bias_(bias), reduce_output_(reduce_output),
      input_is_parallel_(input_is_parallel), skip_bias_add_(skip_bias_add) {
    CHECK_GT(tp_group.WorldSize(), 0) << "No available devices found";
    CHECK_EQ(in_features % tp_group_.WorldSize(), 0)
        << "in_features must be divisible by TP world size for RowParallel";
    input_size_per_partition_ = in_features / tp_group_.WorldSize();

    // init params shards on local rank
    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{out_features, input_size_per_partition_}, DataType::kFLOAT32,
                                   device_)
              ->RequiresGrad();
    if (bias) {
        parameters_[kParamBiasName]
            = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32, device_)->RequiresGrad();
    }
}

std::vector<std::shared_ptr<Tensor>>
RowParallelLinear::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1) << "RowParallelLinear takes exactly one input";
    auto input = input_is_parallel_ ? input_tensors[0] : ScatterToTPRegion(input_tensors[0], tp_group_);

    auto sharded_output = input->Matmul(parameters_.at(kParamWeightName));
    auto output = reduce_output_ ? ReduceFromTPRegion(sharded_output, tp_group_) : output;

    if (bias_ && !skip_bias_add_) {
        output = sharded_output->Add(parameters_[kParamBiasName]);
    }

    return skip_bias_add_ ? {output, bias_ ? parameters_.at(kParamBiasName) : nullptr} : {output};
}

} // namespace infini_train::nn::parallel
