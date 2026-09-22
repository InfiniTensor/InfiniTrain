#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/transformer/transformer.h"

namespace fm9gv {

// FM9GV keeps the reusable LLM implementation in TransformerModel while
// owning the model-specific embedding replacement path here.
class Model : public infini_train::nn::TransformerModel {
public:
    explicit Model(const infini_train::nn::TransformerConfig &config);

    // Mirrors torch.nn.Module.forward: input_ids is required; visual states and
    // their token indices are optional. With only input_ids this is a text LLM.
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs) override;
};

} // namespace fm9gv
