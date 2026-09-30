#include "infini_train/include/checkpoint/save_planner.h"

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::checkpoint {

ShardedStateDict
BuildOptimizerShardedStateDict(const ShardedStateDict &model_state,
                               const std::unordered_map<std::string, std::shared_ptr<Tensor>> &optimizer_state) {
    ShardedStateDict result;
    for (const auto &[key, tensor] : optimizer_state) {
        if (key == kAdamStepKey) {
            auto info = MakeShardedTensor(key, tensor->Dtype(), tensor->Dims());
            info.local_key = key;
            result.tensors.emplace(key, std::move(info));
            continue;
        }

        std::string parameter_key;
        if (key.starts_with(kAdamFirstMomentPrefix)) {
            parameter_key = key.substr(kAdamFirstMomentPrefix.size());
        } else if (key.starts_with(kAdamSecondMomentPrefix)) {
            parameter_key = key.substr(kAdamSecondMomentPrefix.size());
        } else {
            CHECK(false) << "Unsupported optimizer state key: " << key;
        }

        auto model_it = model_state.tensors.find(parameter_key);
        CHECK(model_it != model_state.tensors.end())
            << "Optimizer state " << key << " has no matching named model parameter. "
            << "Optimizer resharding requires named parameters.";
        auto info = model_it->second;
        info.key = key;
        info.local_key = key;
        info.dtype = tensor->Dtype();
        result.tensors.emplace(key, std::move(info));
    }
    return result;
}

std::vector<WriteItem> SavePlanner::Plan(const ShardedStateDict &sd, int rank) {
    std::vector<WriteItem> items;
    for (auto &[key, info] : sd.tensors) {
        bool is_optimizer = key.starts_with(kAdamOptimizerPrefix);
        WriteItem item;
        item.key = key;
        item.filename = is_optimizer ? "optimizer.ckpt" : "model.ckpt";
        item.byte_size = TensorByteSize(info.dtype, info.local_shape);
        item.dtype = info.dtype;
        item.local_shape = info.local_shape;
        item.global_offset = info.global_offset;
        item.axis_fragmentations = info.axis_fragmentations;
        item.rank = rank;

        items.push_back(std::move(item));
    }

    return items;
}

} // namespace infini_train::checkpoint
