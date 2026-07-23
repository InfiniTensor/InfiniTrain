#include "infini_train/include/nn/modules/transformer/moe/experts.h"

#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/transformer/mlp.h"
#include "infini_train/include/nn/modules/transformer/moe/moe_utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::moe {

SequentialMLP::SequentialMLP(int64_t num_local_experts, const TransformerConfig &config)
    : CloneableModule(kType), num_local_experts_(num_local_experts) {
    const auto &moe_config = RequireMoEConfig(config);
    CHECK(moe_config.expert_impl == MoEConfig::ExpertImpl::kSequential);

    CHECK_GT(num_local_experts_, 0);

    for (int64_t expert_idx = 0; expert_idx < num_local_experts_; ++expert_idx) {
        modules_[std::string(kExpertNamePrefix) + std::to_string(expert_idx)] = std::make_shared<MLP>(config);
    }
}

std::vector<std::shared_ptr<Tensor>> SequentialMLP::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 3);
    auto permuted_input = input_tensors[0];
    auto tokens_per_expert = input_tensors[1];
    auto permuted_probs = input_tensors[2];
    CHECK(tokens_per_expert->Dtype() == DataType::kINT64);
    CHECK(tokens_per_expert->Dims() == (std::vector<int64_t>{num_local_experts_}));
    CHECK(tokens_per_expert->GetDevice().IsCPU());
    CHECK_EQ(permuted_input->Dims().size(), 2);
    CHECK(permuted_probs->Dims() == (std::vector<int64_t>{permuted_input->Dims()[0]}));
    CHECK(permuted_probs->GetDevice() == permuted_input->GetDevice());
    const auto *tokens_per_expert_ptr = static_cast<const int64_t *>(tokens_per_expert->DataPtr());

    std::vector<std::shared_ptr<Tensor>> expert_outputs;
    int64_t start = 0;
    for (int64_t expert_idx = 0; expert_idx < num_local_experts_; ++expert_idx) {
        const int64_t num_tokens_for_expert = tokens_per_expert_ptr[expert_idx];
        const int64_t end = start + num_tokens_for_expert;
        if (num_tokens_for_expert == 0) {
            start = end;
            continue;
        }

        auto expert_input = permuted_input->Slice(0, start, end);
        auto expert_probs = permuted_probs->Slice(0, start, end)->View({num_tokens_for_expert, 1});
        auto expert_name = std::string(kExpertNamePrefix) + std::to_string(expert_idx);
        auto expert_output = (*modules_.at(expert_name))({expert_input})[0];
        expert_outputs.push_back(expert_output * expert_probs);
        start = end;
    }
    CHECK_EQ(start, permuted_input->Dims()[0]);
    std::shared_ptr<Tensor> permuted_expert_output;
    if (expert_outputs.empty()) {
        auto output_dtype = permuted_input->Dtype();
        const auto autocast_context = GetCurrentAutocastContext();
        if (autocast_context.enabled) {
            output_dtype = autocast_context.autocast_dtype;
        }
        permuted_expert_output
            = std::make_shared<Tensor>(permuted_input->Dims(), output_dtype, permuted_input->GetDevice());
    } else {
        permuted_expert_output
            = expert_outputs.size() == 1 ? expert_outputs[0] : nn::function::Concat(expert_outputs, 0);
    }
    return {permuted_expert_output};
}

} // namespace infini_train::nn::moe
