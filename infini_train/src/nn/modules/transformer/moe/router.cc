#include "infini_train/include/nn/modules/transformer/moe/router.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/autograd/moe.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/transformer/moe/moe_utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::moe {

TopKRouter::TopKRouter(const TransformerConfig &config) : CloneableModule(kType), config_(config) {
    const auto &moe_config = RequireMoEConfig(config_);
    CHECK(moe_config.router_type == MoERouterType::kTopK);
    CHECK_EQ(moe_config.router_topk, 1) << "Current InfiniTrain MoE implementation supports top-1 routing only";
    CHECK_GT(moe_config.num_experts, 0);

    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{moe_config.num_experts, config_.n_embd}, DataType::kFLOAT32,
                                   device_)
              ->RequiresGrad();
    init::KaimingUniform(parameters_[kParamWeightName]);

    if (config_.add_bias_linear) {
        parameters_[kParamBiasName]
            = std::make_shared<Tensor>(std::vector<int64_t>{moe_config.num_experts}, DataType::kFLOAT32, device_)
                  ->RequiresGrad();
        parameters_[kParamBiasName]->Fill(0.0f);
    }
}

std::vector<std::shared_ptr<Tensor>> TopKRouter::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    std::vector<std::shared_ptr<Tensor>> linear_inputs{input_tensors[0], parameters_.at(kParamWeightName)};
    if (parameters_.contains(kParamBiasName)) {
        linear_inputs.push_back(parameters_.at(kParamBiasName));
    }

    auto logits = std::make_shared<autograd::Linear>()->Apply(linear_inputs)[0];
    auto scores = function::Softmax(logits, -1);
    auto routing_probs = std::make_shared<autograd::Top1Mask>()->Apply({scores})[0];
    return {routing_probs};
}

} // namespace infini_train::nn::moe
