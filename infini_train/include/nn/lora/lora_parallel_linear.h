#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/lora/lora_config.h"
#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
class Device;
} // namespace infini_train

namespace infini_train::nn::lora {

// LoRA wrapper for ColumnParallelLinear
// Weight shape: [out_features_per_partition, in_features]
// LoRA A: [rank, in_features] - replicated across TP ranks
// LoRA B: [out_features_per_partition, rank] - sharded like base weight
class LoRAColumnParallelLinear : public nn::CloneableModule<LoRAColumnParallelLinear> {
public:
    static constexpr char kType[] = "LoRAColumnParallelLinear";

    static constexpr char kParamWeightName[] = "weight";
    static constexpr char kParamBiasName[] = "bias";
    static constexpr char kParamLoraAName[] = "lora_A";
    static constexpr char kParamLoraBName[] = "lora_B";

    // Constructor wrapping existing ColumnParallelLinear
    LoRAColumnParallelLinear(std::shared_ptr<nn::Module> base_module, const LoRAConfig &config, int64_t in_features,
                             int64_t out_features);

    // Constructor wrapping existing ColumnParallelLinear (auto-infer dimensions from weight)
    LoRAColumnParallelLinear(std::shared_ptr<nn::Module> base_module, const LoRAConfig &config);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void MergeWeights();
    void UnmergeWeights();
    bool IsMerged() const;

    std::vector<std::shared_ptr<Tensor>> LoRAParameters() const;
    std::vector<std::shared_ptr<Tensor>> Parameters() const override;

    int64_t in_features() const;
    int64_t out_features() const;
    int64_t rank() const;

private:
    void InitLoRAWeights();
    void FreezeBaseWeights();

    LoRAConfig config_;
    int64_t in_features_;
    int64_t out_features_;
    int64_t out_features_per_partition_;
    bool bias_;
    bool gather_output_;
    bool input_is_parallel_;
    bool skip_bias_add_;
    bool sequence_parallel_;
    bool merged_ = false;

    std::shared_ptr<Tensor> original_weight_;
};

// LoRA wrapper for RowParallelLinear
// Weight shape: [out_features, in_features_per_partition]
// LoRA A: [rank, in_features_per_partition] - sharded like base weight
// LoRA B: [out_features, rank] - replicated, but gradient needs AllReduce
class LoRARowParallelLinear : public nn::CloneableModule<LoRARowParallelLinear> {
public:
    static constexpr char kType[] = "LoRARowParallelLinear";

    static constexpr char kParamWeightName[] = "weight";
    static constexpr char kParamBiasName[] = "bias";
    static constexpr char kParamLoraAName[] = "lora_A";
    static constexpr char kParamLoraBName[] = "lora_B";

    // Constructor wrapping existing RowParallelLinear
    LoRARowParallelLinear(std::shared_ptr<nn::Module> base_module, const LoRAConfig &config, int64_t in_features,
                          int64_t out_features);

    // Constructor wrapping existing RowParallelLinear (auto-infer dimensions from weight)
    LoRARowParallelLinear(std::shared_ptr<nn::Module> base_module, const LoRAConfig &config);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void MergeWeights();
    void UnmergeWeights();
    bool IsMerged() const;

    std::vector<std::shared_ptr<Tensor>> LoRAParameters() const;
    std::vector<std::shared_ptr<Tensor>> Parameters() const override;

    int64_t in_features() const;
    int64_t out_features() const;
    int64_t rank() const;

private:
    void InitLoRAWeights();
    void FreezeBaseWeights();

    LoRAConfig config_;
    int64_t in_features_;
    int64_t out_features_;
    int64_t in_features_per_partition_;
    bool bias_;
    bool reduce_output_;
    bool input_is_parallel_;
    bool skip_bias_add_;
    bool sequence_parallel_;
    bool merged_ = false;

    std::shared_ptr<Tensor> original_weight_;
};

} // namespace infini_train::nn::lora
