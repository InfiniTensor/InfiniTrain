#include "infini_train/include/core/models/decode_only_transformer/layer_specs.h"

#include <cmath>

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/core/transformer/transformer_model.h"
#include "infini_train/include/nn/modules/transformer.h"

namespace infini_train::nn {

ModuleSpec BuildDecoderOnlyTransformerSpec(const TransformerConfig &config, ModuleSpec first_stage, ModuleSpec layer,
                                           ModuleSpec last_stage) {
    ModuleSpec spec(typeid(TransformerModel));
    spec.WithSubmodule(TransformerFirstStage::kType, first_stage)
        .WithSubmodule(TransformerLayer::kType, layer)
        .WithSubmodule(TransformerLastStage::kType, last_stage);

    return spec;
}
} // namespace infini_train::nn
