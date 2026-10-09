#pragma once

#include <memory>
#include <string>
#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/shard_spec.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
struct TransformerConfig;

class MLP : public infini_train::nn::CloneableModule<MLP> {
 public:
  static constexpr char kType[] = "MLP";
  static constexpr char kCFcLayerName[] = "c_fc";
  static constexpr char kGeluLayerName[] = "gelu";
  static constexpr char kCProjLayerName[] = "c_proj";

  static constexpr char kSwiGLULayerName[] = "swiglu";

  explicit MLP(const TransformerConfig &config);

  std::vector<std::shared_ptr<infini_train::Tensor>> Forward(
      const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;

  ShardedStateDict BuildShardedStateDict(
      const std::string &prefix = "") const override;
};
}  // namespace infini_train::nn
