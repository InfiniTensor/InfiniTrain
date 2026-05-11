#include "infini_train/include/nn/modules/transformer/moe/moe_layer.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/transformer/moe/experts.h"
#include "infini_train/include/nn/modules/transformer/moe/moe_utils.h"
#include "infini_train/include/nn/modules/transformer/moe/router.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::moe {

MoELayer::MoELayer(const TransformerConfig &config) : CloneableModule(kType), config_(config) {
    const auto &moe_config = RequireMoEConfig(config_);
    CHECK(config_.ffn_type == FFNType::kMoE);
    CHECK(moe_config.dispatcher_type == MoEDispatcherType::kLocal)
        << "Current InfiniTrain MoE implementation supports local dispatch only";

    modules_[kRouterLayerName] = std::make_shared<TopKRouter>(config_);
    modules_[kExpertsLayerName] = std::make_shared<SequentialMLP>(config_);
}

std::vector<std::shared_ptr<Tensor>> MoELayer::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    auto hidden_states = input_tensors[0];
    auto routing_probs = (*modules_.at(kRouterLayerName))({hidden_states})[0];
    return (*modules_.at(kExpertsLayerName))({hidden_states, routing_probs});
}

} // namespace infini_train::nn::moe
