#pragma once

#include "infini_train/include/nn/modules/transformer/transformer_config.h"

namespace infini_train::nn::moe {

const MoEConfig &RequireMoEConfig(const TransformerConfig &config);

} // namespace infini_train::nn::moe
