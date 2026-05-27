#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::moe {

std::vector<std::shared_ptr<Tensor>> TopkRoutingWithScoreFunction(const std::shared_ptr<Tensor> &logits, int64_t topk,
                                                                  bool use_pre_softmax,
                                                                  std::optional<float> scaling_factor,
                                                                  const MoEConfig::RouterScoreFunction &score_function);

const MoEConfig &RequireMoEConfig(const TransformerConfig &config);

} // namespace infini_train::nn::moe
