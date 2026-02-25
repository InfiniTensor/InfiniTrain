#pragma once

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_config.h"

namespace infini_train::nn {
class TransformerKernel;

class LLaMA3ChunkABI : public TransformerChunkABI {
public:
    static constexpr char kHLayerName[] = "h";
    static constexpr char kFreqsCisName[] = "freqs_cis";

    LLaMA3ChunkABI(const TransformerConfig &config, int start_layer, int end_layer,
                   std::shared_ptr<TransformerKernel> kernel);

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &x) override;

private:
    TransformerConfig config_;
};
} // namespace infini_train::nn