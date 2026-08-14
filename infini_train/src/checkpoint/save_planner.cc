#include "infini_train/include/checkpoint/save_planner.h"

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::checkpoint {

ShardedStateDict
BuildOptimizerShardedStateDict(const ShardedStateDict &model_state,
                               const std::unordered_map<std::string, std::shared_ptr<Tensor>> &optimizer_state) {
    ShardedStateDict result;
    for (const auto &[key, tensor] : optimizer_state) {
        if (key == "adam.t") {
            ShardedTensor info;
            info.key = key;
            info.local_key = key;
            info.dtype = tensor->Dtype();
            info.global_shape = tensor->Dims();
            info.local_shape = tensor->Dims();
            info.global_offset.assign(tensor->Dims().size(), 0);
            info.axis_fragmentations.assign(tensor->Dims().size(), 1);
            result.tensors.emplace(key, std::move(info));
            continue;
        }

        std::string parameter_key;
        if (key.starts_with("adam.m.")) {
            parameter_key = key.substr(std::string("adam.m.").size());
        } else if (key.starts_with("adam.v.")) {
            parameter_key = key.substr(std::string("adam.v.").size());
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
    uint64_t model_offset = 0;
    uint64_t optim_offset = 0;

    for (auto &[key, info] : sd.tensors) {
        bool is_optimizer = key.starts_with("adam.");
        uint64_t &offset = is_optimizer ? optim_offset : model_offset;

        WriteItem item;
        item.key = key;
        item.filename = is_optimizer ? "optimizer.ckpt" : "model.ckpt";
        item.offset = offset;
        item.byte_size = TensorByteSize(info.dtype, info.local_shape);
        item.dtype = info.dtype;
        item.local_shape = info.local_shape;
        item.global_offset = info.global_offset;
        item.axis_fragmentations = info.axis_fragmentations;
        item.rank = rank;

        items.push_back(std::move(item));
        offset += items.back().byte_size;
    }

    return items;
}

} // namespace infini_train::checkpoint
