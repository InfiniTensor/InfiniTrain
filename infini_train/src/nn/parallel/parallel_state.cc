#include "infini_train/include/nn/parallel/parallel_state.h"

#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/utils.h"

namespace infini_train::nn::parallel {
namespace {

using ProcessGroupName = std::string (*)(int);

void CreateProcessGroups(const std::vector<std::vector<int>> &groups, ProcessGroupName group_name,
                         ProcessGroupFactory *process_group_factory) {
    for (const auto &ranks : groups) {
        if (ranks.size() > 1) {
            process_group_factory->GetOrCreate(group_name(ranks.front()), ranks);
        }
    }
}

} // namespace

void InitializeModelParallel(const ModelParallelInitOptions &options) {
    CHECK(options.backend != Device::DeviceType::kInvalid) << "Model-parallel backend must be valid";
    if (global::GetWorldSize() == 1) {
        return;
    }

    auto *process_group_factory = ProcessGroupFactory::Instance(options.backend);
    const auto &dense_rank_generator = global::GetDenseRankGenerator();

    // All processes must execute this sequence in the same order. The default
    // world group is created first by ProcessGroupFactory's constructor.
    CreateProcessGroups(dense_rank_generator.GetRanks(global::DP), GetDataParallelProcessGroupName,
                        process_group_factory);
    CreateProcessGroups(dense_rank_generator.GetRanks(global::TP), GetTensorParallelProcessGroupName,
                        process_group_factory);
    CreateProcessGroups(dense_rank_generator.GetRanks(global::PP), GetPipelineParallelProcessGroupName,
                        process_group_factory);

    if (!options.initialize_expert_groups) {
        return;
    }

    const auto &expert_rank_generator = global::GetExpertRankGenerator();
    CreateProcessGroups(expert_rank_generator.GetRanks(global::DP), GetExpertDataParallelProcessGroupName,
                        process_group_factory);
    CreateProcessGroups(expert_rank_generator.GetRanks(global::TP), GetExpertTensorParallelProcessGroupName,
                        process_group_factory);
    CreateProcessGroups(expert_rank_generator.GetRanks(global::EP), GetExpertParallelProcessGroupName,
                        process_group_factory);
    CreateProcessGroups(expert_rank_generator.GetRanks({global::TP, global::EP}),
                        GetExpertTensorAndExpertParallelProcessGroupName, process_group_factory);
}

} // namespace infini_train::nn::parallel
