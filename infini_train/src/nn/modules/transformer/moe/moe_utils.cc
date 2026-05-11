#include "infini_train/include/nn/modules/transformer/moe/moe_utils.h"

#include "glog/logging.h"

namespace infini_train::nn::moe {

const MoEConfig &RequireMoEConfig(const TransformerConfig &config) {
    CHECK(config.moe_config.has_value()) << "MoE layer requires TransformerConfig::moe_config";
    return config.moe_config.value();
}

} // namespace infini_train::nn::moe
