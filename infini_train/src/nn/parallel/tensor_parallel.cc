#include "infini_train/include/nn/parallel/tensor_parallel.h"

#include <format>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/autograd/sparse.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/nn/parallel/scatter_gather.h"

namespace infini_train::nn::parallel {

// namespace {
// Comm Kernel Call Functions
std::shared_ptr<Tensor> GatherAlongLastDim(const std::shared_ptr<Tensor> &tensor, const TensorParallelGroup &tp_group) {
    int world_size = tp_group.WorldSize();
    CHECK_GT(world_size, 0) << "Tensor Parallel group not initialized";
    if (world_size == 1) {
        // Bypass the function if we are using only 1 GPU.
        return tensor;
    }

    auto it = std::find(tp_group.devices.begin(), tp_group.devices.end(), tensor->GetDevice());
    CHECK(it != tp_group.devices.end()) << "Tensor device not in TensorParallelGroup";

    std::vector<int64_t> output_shape = tensor->Dims();
    output_shape[0] *= world_size;
    auto gathered_output = std::make_shared<Tensor>(output_shape, tensor->Dtype(), tensor->GetDevice());

#ifdef USE_NCCL
    auto kernel = Dispatcher::Instance().GetKernel({tensor->GetDevice()->Type(), "CommNcclAllGather"});
    kernel.Call<void>(gathered_output, tensor);
#else
    LOG(FATAL) << "AllGather requires NCCL enabled";
    return nullptr;
#endif
    // AllGather gather along dim 0 by default
    auto output_list = gathered_output->Split(tensor->Dims()[0], 0);
    auto output = nn::function::Concat(output_list, -1)->Contiguous();

    return output;
}

std::shared_ptr<Tensor> SplitAlongLastDim(const std::shared_ptr<Tensor> &tensor, const TensorParallelGroup &tp_group) {
    int world_size = tp_group.WorldSize();
    CHECK_GT(world_size, 0) << "Tensor Parallel group not initialized";
    if (world_size == 1) {
        // Bypass the function if we are using only 1 GPU.
        return tensor;
    }

    CHECK_EQ(tp_group.RankOf(tensor->GetDevice()), tp_group.Rank()) << "Tensor is not on device of current rank.";
    tensor->GetDevice()->SetDevice();

    auto last_dim_size = tensor->Dims().back() / world_size;
    auto shards = tensor->Split(last_dim_size, -1);
    return shards[tp_group.Rank()]->Contiguous();
}

std::shared_ptr<Tensor> Reduce(const std::shared_ptr<Tensor> &tensor, const TensorParallelGroup &tp_group) {
    int world_size = tp_group.WorldSize();
    CHECK_GT(world_size, 0) << "Tensor Parallel group not initialized";
    if (world_size == 1) {
        // Bypass the function if we are using only 1 GPU.
        return tensor;
    }

    auto it = std::find(tp_group.devices.begin(), tp_group.devices.end(), tensor->GetDevice());
    CHECK(it != tp_group.devices.end()) << "Tensor device not in TensorParallelGroup";

#ifdef USE_NCCL
    auto kernel = Dispatcher::Instance().GetKernel({tensor->GetDevice()->Type(), "CommNcclAllReduceLocal"});
    kernel.Call<void>(tensor);
    return tensor;
#else
    LOG(FATAL) << "AllReduce requires NCCL enabled";
    return nullptr;
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
        return {Reduce(grad_outputs[0], tp_group_)};
    };

private:
    TensorParallelGroup tp_group_;
};

class GatherFromTPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "GatherFromTPRegionFunction";

    explicit GatherFromTPRegion(TensorParallelGroup tp_group) : autograd::Function(kType), tp_group_(tp_group) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        return {GatherAlongLastDim(input_tensors[0], tp_group_)};
    };

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        // Each rank should get the same `full_grad_output`
        // Perform local split to get corresponding shard
        return {SplitAlongLastDim(grad_outputs[0], tp_group_)};
    };

private:
    TensorParallelGroup tp_group_;
};

class ScatterToTPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "ScatterToTPRegionFunction";

    explicit ScatterToTPRegion(TensorParallelGroup tp_group) : autograd::Function(kType), tp_group_(tp_group) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        // Each rank should get the same `input`
        // Perform local split to get corresponding shard
        return {SplitAlongLastDim(input_tensors[0], tp_group_)};
    };

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        return {GatherAlongLastDim(grad_outputs[0], tp_group_)};
    };

private:
    TensorParallelGroup tp_group_;
};

class ReduceFromTPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "ReduceFromTPRegionFunction";

    explicit ReduceFromTPRegion(TensorParallelGroup tp_group) : autograd::Function(kType), tp_group_(tp_group) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        // Perform AllReduceSum to get full output
        return {Reduce(input_tensors[0], tp_group_)};
    };

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        return {grad_outputs[0]};
    };

private:
    TensorParallelGroup tp_group_;
};

// Comm Helper Functions
std::vector<std::shared_ptr<Tensor>> CopyToTPRegionFunc(const std::shared_ptr<Tensor> &input,
                                                        TensorParallelGroup tp_group) {
    return std::make_shared<CopyToTPRegion>(tp_group)->Apply({input});
}

std::vector<std::shared_ptr<Tensor>> GatherFromTPRegionFunc(const std::shared_ptr<Tensor> &input,
                                                            TensorParallelGroup tp_group) {
    return std::make_shared<GatherFromTPRegion>(tp_group)->Apply({input});
}

std::vector<std::shared_ptr<Tensor>> ScatterToTPRegionFunc(const std::shared_ptr<Tensor> &input,
                                                           TensorParallelGroup tp_group) {
    return std::make_shared<ScatterToTPRegion>(tp_group)->Apply({input});
}

std::vector<std::shared_ptr<Tensor>> ReduceFromTPRegionFunc(const std::shared_ptr<Tensor> &input,
                                                            TensorParallelGroup tp_group) {
    return std::make_shared<ReduceFromTPRegion>(tp_group)->Apply({input});
}

void LinearResetParameters(std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias = nullptr) {
    init::KaimingUniform(weight, sqrt(5.0f));
    if (bias) {
        const auto [fan_in, _] = init::CalculateFanInAndFanOut(weight);
        const float bound = fan_in > 0 ? 1.0 / sqrt(fan_in) : 0.0;
        init::Uniform(bias, -bound, bound);
    }
}

// } // namespace

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

    LinearResetParameters(parameters_[kParamWeightName], bias ? parameters_[kParamBiasName] : nullptr);
}

std::vector<std::shared_ptr<Tensor>>
ColumnParallelLinear::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1) << "ColumnParallelLinear takes exactly one input";
    auto input = input_is_parallel_ ? input_tensors[0] : CopyToTPRegionFunc(input_tensors[0], tp_group_)[0];

    auto sharded_output = std::make_shared<autograd::Linear>()->Apply(
        (bias_ && !skip_bias_add_)
            ? std::vector<std::shared_ptr<Tensor>>{input, parameters_.at(kParamWeightName), parameters_[kParamBiasName]}
            : std::vector<std::shared_ptr<Tensor>>{input, parameters_.at(kParamWeightName)})[0];

    std::shared_ptr<Tensor> output
        = gather_output_ ? GatherFromTPRegionFunc(sharded_output, tp_group_)[0] : sharded_output;

    return skip_bias_add_
             ? std::vector<std::shared_ptr<Tensor>>{output, bias_ ? parameters_.at(kParamBiasName) : nullptr}
             : std::vector<std::shared_ptr<Tensor>>{output};
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

    LinearResetParameters(parameters_[kParamWeightName], bias ? parameters_[kParamBiasName] : nullptr);
}

std::vector<std::shared_ptr<Tensor>>
RowParallelLinear::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1) << "RowParallelLinear takes exactly one input";
    auto input = input_is_parallel_ ? input_tensors[0] : ScatterToTPRegionFunc(input_tensors[0], tp_group_)[0];

    auto sharded_output = std::make_shared<autograd::Linear>()->Apply(
        std::vector<std::shared_ptr<Tensor>>{input, parameters_.at(kParamWeightName)})[0];

    auto output = reduce_output_ ? ReduceFromTPRegionFunc(sharded_output, tp_group_)[0] : sharded_output;

    if (bias_ && !skip_bias_add_) {
        output = output->Add(parameters_[kParamBiasName]);
    }

    return skip_bias_add_
             ? std::vector<std::shared_ptr<Tensor>>{output, bias_ ? parameters_.at(kParamBiasName) : nullptr}
             : std::vector<std::shared_ptr<Tensor>>{output};
}

VocabParallelEmbedding::VocabParallelEmbedding(int64_t num_embeddings, int64_t embedding_dim,
                                               TensorParallelGroup tp_group)
    : Module(kType), tp_group_(tp_group), vocab_size_global_(num_embeddings), embedding_dim_(embedding_dim) {
    CHECK_GT(tp_group_.WorldSize(), 0) << "No available devices found for VocabParallelEmbedding";
    CHECK_GT(num_embeddings, 0);
    CHECK_GT(embedding_dim, 0);
    // NOTE(zbl): Assume num_embeddings already been padded to multiple of world size as in Megatron-LM
    CHECK_EQ(num_embeddings % tp_group_.WorldSize(), 0)
        << "num_embeddings must be divisible by TP world size for VocabParallelEmbedding";

    auto world_size = tp_group_.WorldSize();
    auto rank = tp_group_.Rank();

    vocab_size_per_partition_ = num_embeddings / world_size;
    vocab_start_index_ = static_cast<int64_t>(rank) * vocab_size_per_partition_;
    vocab_end_index_ = vocab_start_index_ + vocab_size_per_partition_;

    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{vocab_size_per_partition_, embedding_dim_}, DataType::kFLOAT32,
                                   device_)
              ->RequiresGrad();
}

std::vector<std::shared_ptr<Tensor>>
VocabParallelEmbedding::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1) << "VocabParallelEmbedding takes exactly one input (token ids)";
    auto tokens = input_tensors[0];

    CHECK(tokens->Dtype() == DataType::kINT64 || tokens->Dtype() == DataType::kINT32)
        << "VocabParallelEmbedding expects integer token ids";

    auto world_size = tp_group_.WorldSize();
    auto masked_input = tokens;
    std::shared_ptr<Tensor> input_mask = nullptr;
    if (world_size > 1) {
        // TODO(zbl): BinaryScalar only support float scalars now
        // input_mask: same shape as input [B, T]
        input_mask
            = (tokens < static_cast<float>(vocab_start_index_)) | (tokens >= static_cast<float>(vocab_end_index_));
        masked_input = tokens - static_cast<float>(vocab_start_index_);
        masked_input = masked_input->MaskedFill(input_mask, 0);
    }

    auto local_output
        = std::make_shared<autograd::Embedding>()->Apply({masked_input, parameters_[kParamWeightName]})[0];

    if (world_size > 1) {
        // TODO(zbl): MaskedFill itself does not support row-wise mask, needs implementation
        //            For now, use RepeatInterleave + MaskedFill to work around
        auto MaskRowsLike = [](const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &row_mask,
                               float value) -> std::shared_ptr<Tensor> {
            const auto &input_shape = input->Dims();
            const int64_t H = input_shape.back();
            const int64_t rows = input->NumElements() / H;
            std::shared_ptr<Tensor> mask1d;
            if (row_mask->NumElements() == rows) {
                mask1d = (row_mask->Dims().size() == 1) ? row_mask : row_mask->View({rows});
            } else {
                LOG(FATAL) << "row_mask must have product of all dims except last (expected " << rows << ", got "
                           << row_mask->NumElements() << ")";
            }

            auto fullMask
                = std::make_shared<Tensor>(mask1d->RepeatInterleave(H, 0)->View(input_shape)->To(input->Dtype()));

            return input->MaskedFill(fullMask, value);
        };
        // Same as `local_output[input_mask, :] = 0.0`
        local_output = MaskRowsLike(local_output, input_mask, 0.0f);
    }

    auto output = ReduceFromTPRegionFunc(local_output, tp_group_)[0];

    return {output};
}

} // namespace infini_train::nn::parallel
