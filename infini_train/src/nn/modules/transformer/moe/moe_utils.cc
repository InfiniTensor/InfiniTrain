#include "infini_train/include/nn/modules/transformer/moe/moe_utils.h"

#include "glog/logging.h"

#include "infini_train/include/autograd/local_token_dispatcher.h"
#include "infini_train/include/autograd/scatter.h"
#include "infini_train/include/autograd/topk.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/functional.h"

namespace infini_train::nn::moe {

std::vector<std::shared_ptr<Tensor>>
TopkRoutingWithScoreFunction(const std::shared_ptr<Tensor> &logits, int64_t topk, bool use_pre_softmax,
                             std::optional<float> scaling_factor,
                             const MoEConfig::RouterScoreFunction &score_function) {

    // Megatron TopKRouter returns dense tensors:
    //   routing_probs: [num_tokens, num_experts]
    //   routing_map:   [num_tokens, num_experts], bool
    std::shared_ptr<Tensor> top_probs;
    std::shared_ptr<Tensor> top_indices;

    if (score_function == MoEConfig::RouterScoreFunction::kSoftmax) {
        if (use_pre_softmax) {
            auto scores = function::Softmax(logits, -1);
            auto topk_function = std::make_shared<autograd::TopK>(topk);
            top_probs = topk_function->Apply({scores})[0];
            top_indices = topk_function->TopIndices();
        } else {
            auto topk_function = std::make_shared<autograd::TopK>(topk);
            auto top_scores = topk_function->Apply({logits})[0];
            top_indices = topk_function->TopIndices();
            top_probs = function::Softmax(top_scores, -1);
        }
    } else if (score_function == MoEConfig::RouterScoreFunction::kSigmoid) {
        auto sigmoid_scores = function::Sigmoid(logits);
        auto topk_function = std::make_shared<autograd::TopK>(topk);
        top_probs = topk_function->Apply({sigmoid_scores})[0];
        top_indices = topk_function->TopIndices();
        if (topk > 1) {
            top_probs = top_probs / (top_probs->Sum(-1, true) + 1e-20f);
        }
    } else {
        LOG(FATAL) << "Unsupported MoE router score function";
    }

    if (scaling_factor.has_value()) {
        top_probs = top_probs * scaling_factor.value();
    }

    auto routing_probs = std::make_shared<autograd::Scatter>(logits->Dims())->Apply({top_probs, top_indices})[0];
    auto routing_map_values = std::make_shared<Tensor>(top_indices->Equals(top_indices)->To(DataType::kBOOL));
    auto routing_map = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {logits->GetDevice().type(), "ScatterForward"}, routing_map_values, top_indices, logits->Dims());
    return {routing_probs, routing_map};
}

const MoEConfig &RequireMoEConfig(const TransformerConfig &config) {
    CHECK(config.moe_config.has_value()) << "MoE layer requires TransformerConfig::moe_config";
    return config.moe_config.value();
}

} // namespace infini_train::nn::moe
