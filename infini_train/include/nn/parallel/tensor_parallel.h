#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

struct TensorParallelGroup {
    // TODO(zbl): check torch.distributed.ProcessGroup
    std::vector<const Device *> devices;
    int rank;

    int WorldSize() const { return static_cast<int>(devices.size()); }
    int Rank() const { return rank; }
    int RankOf(const Device *d) const {
        auto it = std::find(devices.begin(), devices.end(), d);
        if (it != devices.end()) {
            return static_cast<int>(std::distance(devices.begin(), it));
        } else {
            return -1;
        }
    }
};

std::vector<std::shared_ptr<Tensor>> GatherFromTPRegionFunc(const std::shared_ptr<Tensor> &input,
                                                            TensorParallelGroup tp_group);

class ColumnParallelLinear : public Module {
public:
    static constexpr char kType[] = "ColumnParallelLinear";

    static constexpr char kParamWeightName[] = "weight";
    static constexpr char kParamBiasName[] = "bias";

    ColumnParallelLinear(int64_t in_features, int64_t out_features, bool bias, TensorParallelGroup tp_group,
                         bool gather_output, bool input_is_parallel, bool skip_bias_add);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    TensorParallelGroup tp_group_;

    bool bias_ = true;
    bool gather_output_ = false;     // whether to return full local output tensor after forward (need gather)
    bool input_is_parallel_ = false; // will perform an autograd-aware copy when false
    bool skip_bias_add_ = false;     // will return {out, bias} if true (for fusion purpose)

    int64_t output_size_per_partition_ = 0;
};

class RowParallelLinear : public Module {
public:
    static constexpr char kType[] = "RowParallelLinear";

    static constexpr char kParamWeightName[] = "weight";
    static constexpr char kParamBiasName[] = "bias";

    RowParallelLinear(int64_t in_features, int64_t out_features, bool bias, TensorParallelGroup tp_group,
                      bool reduce_output, bool input_is_parallel, bool skip_bias_add);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    TensorParallelGroup tp_group_;

    bool bias_ = true;
    bool reduce_output_ = false;     // whether to return full local output tensor after forward (need reduce)
    bool input_is_parallel_ = false; // will perform an autograd-aware copy when false
    bool skip_bias_add_ = false;     // will return {out, bias} if true (for fusion purpose)

    int64_t input_size_per_partition_ = 0;
};

class VocabParallelEmbedding : public Module {
public:
    static constexpr char kType[] = "VocabParallelEmbedding";
    static constexpr char kParamWeightName[] = "weight";

    VocabParallelEmbedding(int64_t num_embeddings, int64_t embedding_dim, TensorParallelGroup tp_group);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    TensorParallelGroup tp_group_;

    int64_t vocab_size_global_ = 0;
    int64_t embedding_dim_ = 0;

    int64_t vocab_size_per_partition_ = 0;
    int64_t vocab_start_index_ = 0;
    int64_t vocab_end_index_ = 0;
};
} // namespace infini_train::nn::parallel
