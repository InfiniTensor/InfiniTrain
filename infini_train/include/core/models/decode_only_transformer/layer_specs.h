#pragma once

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_config.h"

namespace infini_train::nn {

ModuleSpec BuildDecoderOnlyTransformerSpec(const TransformerConfig &config, ModuleSpec first_stage, ModuleSpec chunk,
                                           ModuleSpec last_stage);
} // namespace infini_train::nn
