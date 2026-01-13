#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/process_group.h"

namespace infini_train {
class Tensor;
class Device;
} // namespace infini_train

namespace infini_train::nn::parallel {

// NOTE(zbl): Reserved for VocabParallelEmbedding, since rank is needed in its constructor before any Device exists
//            On other occasions, should use Device::Rank()
extern thread_local int tp_rank;

class ColumnParallelLinear : public nn::CloneableModule<ColumnParallelLinear> {
public:
    static constexpr char kType[] = "ColumnParallelLinear";

    static constexpr char kParamWeightName[] = "weight";
    static constexpr char kParamBiasName[] = "bias";

    ColumnParallelLinear(int64_t in_features, int64_t out_features, bool bias, bool gather_output,
                         bool input_is_parallel, bool skip_bias_add, bool sequence_parallel = false);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    bool bias_ = true;
    bool gather_output_ = false;     // whether to return full local output tensor after forward (need gather)
    bool input_is_parallel_ = false; // will perform an autograd-aware copy when false
    bool skip_bias_add_ = false;     // will return {out, bias} if true (for fusion purpose)
    bool sequence_parallel_ = false; // whether to enable sequence parallel

    int64_t output_size_per_partition_ = 0;
};

class RowParallelLinear : public nn::CloneableModule<RowParallelLinear> {
public:
    static constexpr char kType[] = "RowParallelLinear";

    static constexpr char kParamWeightName[] = "weight";
    static constexpr char kParamBiasName[] = "bias";

    RowParallelLinear(int64_t in_features, int64_t out_features, bool bias, bool reduce_output, bool input_is_parallel,
                      bool skip_bias_add, bool sequence_parallel = false);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    bool bias_ = true;
    bool reduce_output_ = false;     // whether to return full local output tensor after forward (need reduce)
    bool input_is_parallel_ = false; // will perform an autograd-aware copy when false
    bool skip_bias_add_ = false;     // will return {out, bias} if true (for fusion purpose)
    bool sequence_parallel_ = false; // whether to enable sequence parallel

    int64_t input_size_per_partition_ = 0;
};

class VocabParallelEmbedding : public nn::CloneableModule<VocabParallelEmbedding> {
public:
    static constexpr char kType[] = "VocabParallelEmbedding";
    static constexpr char kParamWeightName[] = "weight";

    VocabParallelEmbedding(int64_t num_embeddings, int64_t embedding_dim, bool reduce_scatter_embeddings);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    bool reduce_scatter_embeddings_ = false; // whether to perform ReduceScatter after embedding lookup

    int64_t vocab_size_global_ = 0;
    int64_t embedding_dim_ = 0;

    int64_t vocab_size_per_partition_ = 0;
    int64_t vocab_start_index_ = 0;
    int64_t vocab_end_index_ = 0;
};

class VocabParallelCrossEntropy : public autograd::Function {
public:
    static constexpr char kType[] = "VocabParallelCrossEntropyFunction";

    VocabParallelCrossEntropy(int64_t vocab_size_original = 0, float label_smoothing = 0.f)
        : autograd::Function(kType), vocab_size_original_(vocab_size_original), label_smoothing_(label_smoothing) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    float label_smoothing_ = 0.0f;

    int64_t rows_ = 0;
    int64_t vocab_size_local_ = 0;
    int64_t vocab_size_global_ = 0;
    int64_t vocab_size_original_ = 0; // For padded situations
};

class VocabParallelCrossEntropyLoss : public nn::CloneableModule<VocabParallelCrossEntropyLoss> {
public:
    static constexpr char kType[] = "VocabParallelCrossEntropyLoss";
    VocabParallelCrossEntropyLoss(int64_t vocab_size_original = 0, float label_smoothing = 0.f)
        : CloneableModule(kType), vocab_size_original_(vocab_size_original), label_smoothing_(label_smoothing){};

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    float label_smoothing_ = 0.0f;
    int64_t vocab_size_original_ = 0; // For padded situations
};
} // namespace infini_train::nn::parallel
