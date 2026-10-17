#include "infini_train/include/nn/modules/transformer/moe/moe_layer.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/transformer/moe/experts.h"
#include "infini_train/include/nn/modules/transformer/moe/moe_utils.h"
#include "infini_train/include/nn/modules/transformer/moe/router.h"
#include "infini_train/include/nn/modules/transformer/moe/token_dispatcher.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::moe {
namespace {

std::vector<int64_t> LocalExpertIds(int64_t num_local_experts, int global_rank) {
    const auto expert_ranks = parallel::GetExpertParallelGroupRanks(global_rank);
    const auto rank_it = std::find(expert_ranks.begin(), expert_ranks.end(), global_rank);
    CHECK(rank_it != expert_ranks.end());
    const int64_t ep_rank = std::distance(expert_ranks.begin(), rank_it);
    const int64_t local_expert_start = ep_rank * num_local_experts;
    std::vector<int64_t> local_expert_ids(num_local_experts);
    std::iota(local_expert_ids.begin(), local_expert_ids.end(), local_expert_start);
    return local_expert_ids;
}

} // namespace

MoELayer::MoELayer(const TransformerConfig &config) : CloneableModule(kType), config_(config) {
    const auto &moe_config = RequireMoEConfig(config_);
    CHECK(config_.ffn_type == FFNType::kMoE);
    CHECK(moe_config.token_dispatcher_type == MoEConfig::TokenDispatcherType::kAllGather)
        << "Current InfiniTrain MoE implementation supports AllGather dispatcher only";
    const int expert_parallel_size = parallel::global::GetExpertParallelSize();
    if (expert_parallel_size > 1) {
        CHECK_EQ(parallel::global::GetExpertTensorParallelSize(), 1)
            << "MoE expert parallelism currently requires expert_tensor_parallel_size=1";
    }
    CHECK_GT(moe_config.num_experts, 0);
    CHECK_EQ(moe_config.num_experts % expert_parallel_size, 0);

    num_local_experts_ = moe_config.num_experts / expert_parallel_size;
    CHECK_GT(num_local_experts_, 0);

    modules_[kRouterLayerName] = std::make_shared<TopKRouter>(config_);
    modules_[kExpertsLayerName] = std::make_shared<SequentialMLP>(num_local_experts_, config_);
}

std::vector<std::shared_ptr<Tensor>> MoELayer::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    auto hidden_states = input_tensors[0];

    // routing
    auto router_output = (*modules_.at(kRouterLayerName))({hidden_states});
    CHECK_EQ(router_output.size(), 2);
    const int64_t num_global_experts = RequireMoEConfig(config_).num_experts;
    const int64_t num_tokens = router_output[0]->NumElements() / num_global_experts;
    auto routing_map = router_output[1]->View({num_tokens, num_global_experts});
    auto probs = router_output[0]->View({num_tokens, num_global_experts});

    // init dispatcher
    const auto &device = hidden_states->GetDevice();
    const int global_rank = device.Rank().GlobalRank();
    const auto local_expert_ids = LocalExpertIds(num_local_experts_, global_rank);
    const auto group_ranks = parallel::GetExpertTensorAndExpertParallelGroupRanks(global_rank);
    const parallel::ProcessGroup *process_group = nullptr;
    if (group_ranks.size() > 1) {
        process_group
            = parallel::ProcessGroupFactory::Instance(device.type())
                  ->GetOrCreate(parallel::GetExpertTensorAndExpertParallelProcessGroupName(global_rank), group_ranks);
    }
    MoEAllGatherTokenDispatcher dispatcher(num_local_experts_, local_expert_ids, config_, process_group);

    // dispatch
    auto preprocessed = dispatcher.DispatchPreprocess(hidden_states, routing_map, probs);
    auto gathered = dispatcher.TokenDispatch(preprocessed[0], preprocessed[1]);
    auto dispatch = dispatcher.DispatchPostprocess(gathered[0], gathered[1]);

    // expert computation
    auto expert_output = (*modules_.at(kExpertsLayerName))(
        {dispatch.permuted_input, dispatch.metadata.tokens_per_expert, dispatch.permuted_probs})[0];

    // combine
    auto unpermuted = dispatcher.CombinePreprocess(expert_output);
    auto combined = dispatcher.TokenCombine(unpermuted);
    return {dispatcher.CombinePostprocess(combined)};
}

} // namespace infini_train::nn::moe
