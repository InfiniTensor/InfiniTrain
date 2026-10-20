#include "infini_train/include/checkpoint/save_planner.h"

#include <map>
#include <ostream>
#include <string_view>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/checkpoint/constants.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train::checkpoint {

ShardedStateDict BuildOptimizerShardedStateDict(
    const ShardedStateDict &model_state,
    const std::unordered_map<std::string, std::shared_ptr<Tensor>>
        &optimizer_state,
    const NamedParameterList &optimizer_parameters) {
  std::map<const Tensor *, std::string> global_key_by_tensor;
  for (const auto &[key, info] : model_state.tensors) {
    if (info.local_tensor != nullptr &&
        !global_key_by_tensor.contains(info.local_tensor)) {
      global_key_by_tensor.emplace(info.local_tensor, key);
    }
  }
  std::unordered_map<std::string, const Tensor *> tensor_by_optimizer_name;
  for (const auto &[name, parameter] : optimizer_parameters) {
    CHECK(parameter != nullptr) << "Null optimizer parameter: " << name;
    CHECK(tensor_by_optimizer_name.emplace(name, parameter.get()).second)
        << "Duplicate optimizer parameter name: " << name;
  }

  ShardedStateDict result;
  for (const auto &[local_key, tensor] : optimizer_state) {
    if (local_key == optimizers::kAdamStepKey) {
      auto info = MakeShardedTensor(local_key, tensor->Dtype(), tensor->Dims());
      info.local_key = local_key;
      result.tensors.emplace(local_key, std::move(info));
      continue;
    }

    std::string_view state_prefix;
    if (local_key.starts_with(optimizers::kAdamFirstMomentPrefix)) {
      state_prefix = optimizers::kAdamFirstMomentPrefix;
    } else if (local_key.starts_with(optimizers::kAdamSecondMomentPrefix)) {
      state_prefix = optimizers::kAdamSecondMomentPrefix;
    } else {
      CHECK(false) << "Unsupported optimizer state key: " << local_key;
    }
    const auto local_parameter_name = local_key.substr(state_prefix.size());
    std::string global_parameter_key;
    if (!optimizer_parameters.empty()) {
      CHECK(tensor_by_optimizer_name.contains(local_parameter_name))
          << "Optimizer state has no parameter identity mapping: " << local_key;
      const auto *parameter = tensor_by_optimizer_name.at(local_parameter_name);
      CHECK(global_key_by_tensor.contains(parameter))
          << "Optimizer parameter is absent from model sharded state: "
          << local_parameter_name;
      global_parameter_key = global_key_by_tensor.at(parameter);
    } else {
      // Compatibility for metadata-only planner callers. Production save/load
      // always supplies identities.
      CHECK(model_state.tensors.contains(local_parameter_name))
          << "Optimizer state has no matching model metadata: " << local_key;
      global_parameter_key = local_parameter_name;
    }
    const auto global_key = std::string(state_prefix) + global_parameter_key;

    auto info = model_state.tensors.at(global_parameter_key);
    info.key = global_key;
    info.local_key = local_key;
    info.local_tensor = nullptr;
    info.dtype = tensor->Dtype();
    CHECK(result.tensors.emplace(global_key, std::move(info)).second)
        << "Duplicate global optimizer state key: " << global_key;
  }
  return result;
}

std::vector<WriteItem> SavePlanner::Plan(const ShardedStateDict &sd, int rank) {
  std::vector<WriteItem> items;
  for (auto &[key, info] : sd.tensors) {
    bool is_optimizer = key.starts_with(optimizers::kAdamOptimizerPrefix);
    WriteItem item;
    item.key = key;
    item.filename =
        is_optimizer ? kOptimizerCheckpointFilename : kModelCheckpointFilename;
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

}  // namespace infini_train::checkpoint
