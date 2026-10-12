#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"

namespace infini_train {
class Tensor;
}

namespace fm9gv {

class Model;
class Resampler;
class VisionTransformer;

// FM9G4B-V's model-specific composition. The reusable text transformer remains
// in Model; this module only owns the vision encoder, resampler and multimodal
// data flow used by the official model.
class MultimodalModel : public infini_train::nn::CloneableModule<MultimodalModel> {
public:
    static constexpr char kType[] = "FM9GVMultimodalModel";
    static constexpr char kLLMName[] = "llm";
    static constexpr char kVPMName[] = "vpm";
    static constexpr char kResamplerName[] = "resampler";

    MultimodalModel(std::shared_ptr<Model> llm, std::shared_ptr<VisionTransformer> vpm,
                    std::shared_ptr<Resampler> resampler);

    // Supported forms:
    //   {input_ids}
    //   {input_ids, vision_hidden_states, image_indices}
    //   {input_ids, pixel_values, target_sizes, image_indices}
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs) override;

    std::shared_ptr<infini_train::Tensor> EncodeVision(const std::shared_ptr<infini_train::Tensor> &pixel_values,
                                                       const std::shared_ptr<infini_train::Tensor> &target_sizes);
    std::vector<std::shared_ptr<infini_train::Tensor>>
    ForwardWithVisionHiddenStates(const std::shared_ptr<infini_train::Tensor> &input_ids,
                                  const std::shared_ptr<infini_train::Tensor> &vision_hidden_states,
                                  const std::shared_ptr<infini_train::Tensor> &image_indices);

    void SetVisionRequiresGrad(bool requires_grad);
    void SetLLMRequiresGrad(bool requires_grad);
};

} // namespace fm9gv
