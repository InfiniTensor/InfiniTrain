#include "infini_train/include/nn/parallel/tensor_parallel.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/autograd/sparse.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

// NOTE(zbl): Reserved for VocabParallelEmbedding, since rank is needed in its constructor before any Device exists
//            On other occasions, should use Device::Rank()
thread_local int tp_rank = 0;

namespace {
// Comm Kernel Call Functions
std::shared_ptr<Tensor> GatherAlongFirstDim(const std::shared_ptr<Tensor> &tensor) {
    int world_size = global::GetTensorParallelSize();
    CHECK_GT(world_size, 0) << "Tensor Parallel group not initialized";
    if (world_size == 1) {
        // Bypass the function if we are using only 1 GPU.
        return tensor;
    }

    auto device = tensor->GetDevice();
    auto tp_group = ProcessGroupFactory::Instance()->Get(GetTensorParallelProcessGroupName(device.Rank().GlobalRank()));

    std::vector<int64_t> output_shape = tensor->Dims();
    output_shape[0] *= world_size;
    auto gathered_output = std::make_shared<Tensor>(output_shape, tensor->Dtype(), device);

    tp_group->AllGather(gathered_output, tensor, false);
    return gathered_output;
}

std::shared_ptr<Tensor> GatherAlongLastDim(const std::shared_ptr<Tensor> &tensor) {
    int world_size = global::GetTensorParallelSize();
    CHECK_GT(world_size, 0) << "Tensor Parallel group not initialized";
    if (world_size == 1) {
        // Bypass the function if we are using only 1 GPU.
        return tensor;
    }

    auto device = tensor->GetDevice();
    auto tp_group = ProcessGroupFactory::Instance()->Get(GetTensorParallelProcessGroupName(device.Rank().GlobalRank()));

    std::vector<int64_t> output_shape = tensor->Dims();
    output_shape[0] *= world_size;
    auto gathered_output = std::make_shared<Tensor>(output_shape, tensor->Dtype(), device);

    tp_group->AllGather(gathered_output, tensor, false);

    // AllGather gather along dim 0 by default
    auto output_list = gathered_output->Split(tensor->Dims()[0], 0);
    auto output = nn::function::Concat(output_list, -1)->Contiguous();

    return output;
}

std::shared_ptr<Tensor> SplitAlongLastDim(const std::shared_ptr<Tensor> &tensor) {
    int world_size = global::GetTensorParallelSize();
    CHECK_GT(world_size, 0) << "Tensor Parallel group not initialized";
    if (world_size == 1) {
        // Bypass the function if we are using only 1 GPU.
        return tensor;
    }

    auto device = tensor->GetDevice();
    auto tp_group = ProcessGroupFactory::Instance()->Get(GetTensorParallelProcessGroupName(device.Rank().GlobalRank()));
    auto rank = tp_group->GetGroupRank(device.Rank().GlobalRank());

    auto last_dim_size = tensor->Dims().back() / world_size;
    auto shards = tensor->Split(last_dim_size, -1);
    return shards[rank]->Contiguous();
}

std::shared_ptr<Tensor> Reduce(const std::shared_ptr<Tensor> &tensor) {
    int world_size = global::GetTensorParallelSize();
    CHECK_GT(world_size, 0) << "Tensor Parallel group not initialized";
    if (world_size == 1) {
        // Bypass the function if we are using only 1 GPU.
        return tensor;
    }

    auto device = tensor->GetDevice();
    auto tp_group = ProcessGroupFactory::Instance()->Get(GetTensorParallelProcessGroupName(device.Rank().GlobalRank()));

    auto output = std::make_shared<Tensor>(*tensor);

    tp_group->AllReduce(output, function::ReduceOpType::kSum, false);
    return output;
}

std::shared_ptr<Tensor> ReduceScatterAlongFirstDim(const std::shared_ptr<Tensor> &tensor) {
    int world_size = global::GetTensorParallelSize();
    CHECK_GT(world_size, 0) << "Tensor Parallel group not initialized";
    if (world_size == 1) {
        // Bypass the function if we are using only 1 GPU.
        return tensor;
    }

    auto device = tensor->GetDevice();
    auto tp_group = ProcessGroupFactory::Instance()->Get(GetTensorParallelProcessGroupName(device.Rank().GlobalRank()));

    auto output_shape = tensor->Dims();
    CHECK_EQ(output_shape[0] % world_size, 0) << "First dimension of the tensor should be divisible by TP world size";
    output_shape[0] /= world_size;

    auto output = std::make_shared<Tensor>(output_shape, tensor->Dtype(), device);

    tp_group->ReduceScatter(output, tensor, function::ReduceOpType::kSum, false);

    return output;
}

// Autograd Function definitions
class CopyToTPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "CopyToTPRegionFunction";

    explicit CopyToTPRegion() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        // Each rank should get the same `input`
        // No need to perform Broadcast-like copy
        return {std::make_shared<Tensor>(*input_tensors[0])};
    };

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        return {Reduce(grad_outputs[0])};
    };
};

class GatherFromTPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "GatherFromTPRegionFunction";

    explicit GatherFromTPRegion() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        return {GatherAlongLastDim(input_tensors[0])};
    };

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        // Each rank should get the same `full_grad_output`
        // Perform local split to get corresponding shard
        return {SplitAlongLastDim(grad_outputs[0])};
    };
};

class ScatterToTPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "ScatterToTPRegionFunction";

    explicit ScatterToTPRegion() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        // Each rank should get the same `input`
        // Perform local split to get corresponding shard
        return {SplitAlongLastDim(input_tensors[0])};
    };

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        return {GatherAlongLastDim(grad_outputs[0])};
    };
};

class ReduceFromTPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "ReduceFromTPRegionFunction";

    explicit ReduceFromTPRegion() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        // Perform AllReduceSum to get full output
        return {Reduce(input_tensors[0])};
    };

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        return {std::make_shared<Tensor>(*grad_outputs[0])};
    };
};

class ReduceScatterToSPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "ReduceScatterToSPRegionFunction";

    explicit ReduceScatterToSPRegion() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        // FIXME(zbl): Megatron-LM keeps tensor as [S, B, H] by default
        return {ReduceScatterAlongFirstDim(input_tensors[0]->Transpose(0, 1))->Transpose(0, 1)};
    };

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        // FIXME(zbl): Megatron-LM keeps tensor as [S, B, H] by default
        return {GatherAlongFirstDim(grad_outputs[0]->Transpose(0, 1))->Transpose(0, 1)};
    };
};

class GatherFromSPRegion : public autograd::Function {
public:
    static constexpr char kType[] = "GatherFromSPRegionFunction";

    explicit GatherFromSPRegion() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        // FIXME(zbl): Megatron-LM keeps tensor as [S, B, H] by default
        return {GatherAlongFirstDim(input_tensors[0]->Transpose(0, 1))->Transpose(0, 1)};
    };

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        // FIXME(zbl): Megatron-LM keeps tensor as [S, B, H] by default
        return {ReduceScatterAlongFirstDim(grad_outputs[0]->Transpose(0, 1))->Transpose(0, 1)};
    };
};

void LinearResetParameters(std::shared_ptr<Tensor> weight, std::shared_ptr<Tensor> bias = nullptr) {
    init::KaimingUniform(weight, sqrt(5.0f));
    if (bias) {
        const auto [fan_in, _] = init::CalculateFanInAndFanOut(weight);
        const float bound = fan_in > 0 ? 1.0 / sqrt(fan_in) : 0.0;
        init::Uniform(bias, -bound, bound);
    }
}

// Comm Helper Functions
std::vector<std::shared_ptr<Tensor>> CopyToTPRegionFunc(const std::shared_ptr<Tensor> &input) {
    return std::make_shared<CopyToTPRegion>()->Apply({input});
}

std::vector<std::shared_ptr<Tensor>> GatherFromTPRegionFunc(const std::shared_ptr<Tensor> &input) {
    return std::make_shared<GatherFromTPRegion>()->Apply({input});
}

std::vector<std::shared_ptr<Tensor>> ScatterToTPRegionFunc(const std::shared_ptr<Tensor> &input) {
    return std::make_shared<ScatterToTPRegion>()->Apply({input});
}

std::vector<std::shared_ptr<Tensor>> ReduceFromTPRegionFunc(const std::shared_ptr<Tensor> &input) {
    return std::make_shared<ReduceFromTPRegion>()->Apply({input});
}

std::vector<std::shared_ptr<Tensor>> ReduceScatterToSPRegionFunc(const std::shared_ptr<Tensor> &input) {
    return std::make_shared<ReduceScatterToSPRegion>()->Apply({input});
}

std::vector<std::shared_ptr<Tensor>> GatherFromSPRegionFunc(const std::shared_ptr<Tensor> &input) {
    return std::make_shared<GatherFromSPRegion>()->Apply({input});
}
} // namespace

ColumnParallelLinear::ColumnParallelLinear(int64_t in_features, int64_t out_features, bool bias, bool gather_output,
                                           bool input_is_parallel, bool skip_bias_add, bool sequence_parallel)
    : CloneableModule(kType), bias_(bias), gather_output_(gather_output), input_is_parallel_(input_is_parallel),
      skip_bias_add_(skip_bias_add), sequence_parallel_(sequence_parallel) {
    auto tp_size = global::GetTensorParallelSize();
    CHECK_GT(tp_size, 0) << "No available devices found";
    CHECK_EQ(out_features % tp_size, 0) << "out_features must be divisible by TP world size for ColumnParallel";

    output_size_per_partition_ = out_features / tp_size;

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
    auto input
        = (input_is_parallel_ || sequence_parallel_) ? input_tensors[0] : CopyToTPRegionFunc(input_tensors[0])[0];

    if (sequence_parallel_) {
        input = GatherFromSPRegionFunc(input)[0];
    }

    auto sharded_output = std::make_shared<autograd::Linear>()->Apply(
        (bias_ && !skip_bias_add_)
            ? std::vector<std::shared_ptr<Tensor>>{input, parameters_.at(kParamWeightName), parameters_[kParamBiasName]}
            : std::vector<std::shared_ptr<Tensor>>{input, parameters_.at(kParamWeightName)})[0];

    std::shared_ptr<Tensor> output = gather_output_ ? GatherFromTPRegionFunc(sharded_output)[0] : sharded_output;

    return skip_bias_add_
             ? std::vector<std::shared_ptr<Tensor>>{output, bias_ ? parameters_.at(kParamBiasName) : nullptr}
             : std::vector<std::shared_ptr<Tensor>>{output};
}

RowParallelLinear::RowParallelLinear(int64_t in_features, int64_t out_features, bool bias, bool reduce_output,
                                     bool input_is_parallel, bool skip_bias_add, bool sequence_parallel)
    : CloneableModule(kType), bias_(bias), reduce_output_(reduce_output), input_is_parallel_(input_is_parallel),
      skip_bias_add_(skip_bias_add), sequence_parallel_(sequence_parallel) {
    auto tp_size = global::GetTensorParallelSize();
    CHECK_GT(tp_size, 0) << "No available devices found";
    CHECK_EQ(in_features % tp_size, 0) << "in_features must be divisible by TP world size for RowParallel";
    input_size_per_partition_ = in_features / tp_size;

    if (!input_is_parallel_ && sequence_parallel_) {
        LOG(FATAL) << "To enable `sequence_parallel`, `input_is_parallel` must be `True`";
    }

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
    auto input = input_is_parallel_ ? input_tensors[0] : ScatterToTPRegionFunc(input_tensors[0])[0];

    auto sharded_output = std::make_shared<autograd::Linear>()->Apply(
        std::vector<std::shared_ptr<Tensor>>{input, parameters_.at(kParamWeightName)})[0];

    auto output = reduce_output_ ? (sequence_parallel_ ? ReduceScatterToSPRegionFunc(sharded_output)[0]
                                                       : ReduceFromTPRegionFunc(sharded_output)[0])
                                 : sharded_output;

    if (bias_ && !skip_bias_add_) {
        output = output->Add(parameters_[kParamBiasName]);
    }

    return skip_bias_add_
             ? std::vector<std::shared_ptr<Tensor>>{output, bias_ ? parameters_.at(kParamBiasName) : nullptr}
             : std::vector<std::shared_ptr<Tensor>>{output};
}

VocabParallelEmbedding::VocabParallelEmbedding(int64_t num_embeddings, int64_t embedding_dim,
                                               bool reduce_scatter_embeddings)
    : CloneableModule(kType), vocab_size_global_(num_embeddings), embedding_dim_(embedding_dim),
      reduce_scatter_embeddings_(reduce_scatter_embeddings) {
    auto tp_size = global::GetTensorParallelSize();
    CHECK_GT(tp_size, 0) << "No available devices found for VocabParallelEmbedding";
    CHECK_GT(num_embeddings, 0);
    CHECK_GT(embedding_dim, 0);
    // NOTE(zbl): Assume num_embeddings already been padded to multiple of world size as in Megatron-LM
    CHECK_EQ(num_embeddings % tp_size, 0)
        << "num_embeddings must be divisible by TP world size for VocabParallelEmbedding";

    vocab_size_per_partition_ = num_embeddings / tp_size;
    vocab_start_index_ = static_cast<int64_t>(tp_rank) * vocab_size_per_partition_;
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

    auto tp_size = global::GetTensorParallelSize();
    auto masked_input = tokens;
    std::shared_ptr<Tensor> input_mask = nullptr;
    if (tp_size > 1) {
        // TODO(zbl): BinaryScalar only support float scalars now
        // input_mask: same shape as input [B, T]
        input_mask
            = (tokens < static_cast<float>(vocab_start_index_)) | (tokens >= static_cast<float>(vocab_end_index_));
        masked_input = tokens - static_cast<float>(vocab_start_index_);
        masked_input = masked_input->MaskedFill(input_mask, 0);
    }

    auto local_output
        = std::make_shared<autograd::Embedding>()->Apply({masked_input, parameters_[kParamWeightName]})[0];

    if (tp_size > 1) {
        // NOTE(zbl): Already extend MaskedFill to support row-wise mask
        //            Same as `local_output[input_mask, :] = 0.0`
        local_output = local_output->MaskedFill(std::make_shared<Tensor>(input_mask->To(local_output->Dtype())), 0.0f);
    }

    auto output = reduce_scatter_embeddings_ ? ReduceScatterToSPRegionFunc(local_output)[0]
                                             : ReduceFromTPRegionFunc(local_output)[0];

    return {output};
}

std::vector<std::shared_ptr<Tensor>>
VocabParallelCrossEntropy::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2) << kType << " expects {logits, target}";
    // NOTE(zbl): CrossEntropy originally requires FP32 in autocast context. Here we explicitly upcast logits to FP32
    //            at the beginning of the forward pass, in alignment with Megatron-LM's behavior. Ref:
    //            https://github.com/NVIDIA/Megatron-LM/blob/e07c4a4450b6faa187a1ef4ec082a35ad7d2f085/megatron/core/tensor_parallel/cross_entropy.py#L28
    auto logits = std::make_shared<Tensor>(input_tensors[0]->To(DataType::kFLOAT32));
    auto target = input_tensors[1];

    auto dtype = logits->Dtype();
    auto device = logits->GetDevice();

    CHECK(target->Dtype() == DataType::kINT64) << "target must be int64";
    CHECK_GE(label_smoothing_, 0.0f);
    CHECK_LT(label_smoothing_, 1.0f);

    int tp_size = global::GetTensorParallelSize();
    const ProcessGroup *tp_group = nullptr;
    int rank = 0;
    if (tp_size > 1) {
        tp_group = ProcessGroupFactory::Instance()->Get(GetTensorParallelProcessGroupName(device.Rank().GlobalRank()));
        rank = tp_group->GetGroupRank(device.Rank().GlobalRank());
    }

    vocab_size_local_ = logits->Dims().back();
    vocab_size_global_ = static_cast<int64_t>(vocab_size_local_) * tp_size;
    vocab_size_original_ = vocab_size_original_ == 0 ? vocab_size_global_ : vocab_size_original_;
    CHECK_LE(vocab_size_original_, vocab_size_global_) << "Original vocab size should be <= padded vocab size";

    // rows = product of all dims except last
    rows_ = logits->NumElements() / vocab_size_local_;

    // 0. Mask out the padded part to -inf
    int64_t vocab_start = static_cast<int64_t>(rank) * vocab_size_local_;
    int64_t vocab_end = vocab_start + vocab_size_local_;

    auto col_ids = nn::init::Arange(0, vocab_size_local_, DataType::kINT64, device);
    auto global_ids = (tp_size > 1) ? col_ids->Add(static_cast<float>(vocab_start)) : col_ids;
    auto valid_mask_local
        = std::make_shared<Tensor>((global_ids < static_cast<float>(vocab_size_original_))->To(logits->Dtype()))
              ->View({1, vocab_size_local_});

    auto logits_masked = logits->MaskedFill(1 - valid_mask_local, -std::numeric_limits<float>::infinity());

    // 1. Calculate global max value
    auto local_max = logits_masked->Max(-1);
    auto global_max = local_max;
    if (tp_size > 1) {
        tp_group->AllReduce(global_max, function::ReduceOpType::kMax, false);
    }
    auto shifted = logits_masked->Sub(global_max->Unsqueeze(-1));

    // 2. Prepare vocab range and mask
    std::shared_ptr<Tensor> target_mask;
    std::shared_ptr<Tensor> masked_target;
    if (tp_size > 1) {
        target_mask = (target < static_cast<float>(vocab_start)) | (target >= static_cast<float>(vocab_end));
        masked_target = target - static_cast<float>(vocab_start);
        masked_target = masked_target->MaskedFill(target_mask, 0);
    } else {
        target_mask = std::make_shared<Tensor>(target->Dims(), DataType::kINT64, target->GetDevice());
        target_mask->Fill(0);
        masked_target = target;
    }

    // 3, Calculate exp(shifted) and global sum_exp
    auto exp_local = shifted->Exp();
    auto sum_exp_local = exp_local->Sum(-1);
    auto sum_exp = (tp_size > 1) ? ReduceFromTPRegionFunc(sum_exp_local)[0] : sum_exp_local;

    // 4. Perform Softmax（local shards but normalize globally）
    auto softmax_local = exp_local->Div(sum_exp->Unsqueeze(-1));

    // 5. Perform allreduce to get global predicted_logit
    auto pred_local = shifted->Gather(-1, masked_target->Unsqueeze(-1))->Squeeze(-1);
    if (tp_size > 1) {
        pred_local = pred_local->MaskedFill(std::make_shared<Tensor>(target_mask->To(pred_local->Dtype())), 0.0f);
    }
    auto predicted = (tp_size > 1) ? ReduceFromTPRegionFunc(pred_local)[0] : pred_local;

    // 6. loss = log(sum_exp) - predicted_logit
    auto log_sum_exp = sum_exp->Log();
    auto loss = log_sum_exp->Sub(predicted);

    // 7. Label smoothing（According to Megatron-LM）
    // TODO(zbl): adjust smoothing coef according to vocab_size_original
    if (label_smoothing_ > 0.0f) {
        // mean_logp over *valid tokens only*:
        // mean_logp = (sum_i_in_valid shifted_i) / vocab_size_original_ - log_sum_exp
        auto shifted2d = shifted->View({rows_, vocab_size_local_});
        auto sum_shifted_valid_local = (shifted2d->Mul(valid_mask_local))->Sum(-1);

        auto sum_shifted_valid
            = (tp_size > 1) ? ReduceFromTPRegionFunc(sum_shifted_valid_local)[0] : sum_shifted_valid_local;

        auto mean_logp = sum_shifted_valid->Mul(1.f / static_cast<float>(vocab_size_original_))->Sub(log_sum_exp);

        float smoothing = label_smoothing_;
        loss = loss->Mul(1.0f - smoothing)->Sub(mean_logp->Mul(smoothing));
    }

    // 8. Save for backward
    saved_tensors_ = {softmax_local, target_mask, masked_target, valid_mask_local};

    return {loss};
}

std::vector<std::shared_ptr<Tensor>>
VocabParallelCrossEntropy::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);

    auto grad_output = grad_outputs[0];
    auto softmax_local = saved_tensors_[0];
    auto target_mask = std::make_shared<Tensor>(saved_tensors_[1]->To(softmax_local->Dtype()));
    auto masked_target = saved_tensors_[2];
    auto valid_mask_local = saved_tensors_[3];

    auto device = grad_output->GetDevice().type();
    auto grad_input = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {device, "VocabParallelCrossEntropyBackward"}, grad_output, softmax_local, target_mask, masked_target,
        valid_mask_local, vocab_size_local_, vocab_size_original_, label_smoothing_);
    return {grad_input, nullptr};
}

std::vector<std::shared_ptr<Tensor>>
VocabParallelCrossEntropyLoss::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    auto logits = input_tensors[0];
    auto target = input_tensors[1];

    auto loss_tensor = std::make_shared<VocabParallelCrossEntropy>(vocab_size_original_, label_smoothing_)
                           ->Apply(std::vector<std::shared_ptr<Tensor>>{logits, target})[0];
    // NOTE(zbl): loss should be a scalar
    std::shared_ptr<Tensor> scalar = loss_tensor->View({static_cast<int64_t>(loss_tensor->NumElements())})->Mean(0);
    return {scalar};
}
} // namespace infini_train::nn::parallel
