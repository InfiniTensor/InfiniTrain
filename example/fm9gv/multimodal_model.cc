#include "example/fm9gv/multimodal_model.h"

#include <utility>

#include "glog/logging.h"

#include "example/fm9gv/model.h"
#include "example/fm9gv/resampler.h"
#include "example/fm9gv/vision_encoder.h"
#include "infini_train/include/tensor.h"

namespace fm9gv {

using infini_train::Tensor;

MultimodalModel::MultimodalModel(std::shared_ptr<Model> llm, std::shared_ptr<VisionTransformer> vpm,
                                 std::shared_ptr<Resampler> resampler)
    : CloneableModule(kType) {
    CHECK(llm != nullptr);
    CHECK(vpm != nullptr);
    CHECK(resampler != nullptr);
    modules_[kLLMName] = std::move(llm);
    modules_[kVPMName] = std::move(vpm);
    modules_[kResamplerName] = std::move(resampler);
}

std::shared_ptr<Tensor> MultimodalModel::EncodeVision(const std::shared_ptr<Tensor> &pixel_values,
                                                      const std::shared_ptr<Tensor> &target_sizes) {
    auto features = (*modules_[kVPMName])({pixel_values, target_sizes})[0];
    return (*modules_[kResamplerName])({features, target_sizes})[0];
}

std::vector<std::shared_ptr<Tensor>>
MultimodalModel::ForwardWithVisionHiddenStates(const std::shared_ptr<Tensor> &input_ids,
                                               const std::shared_ptr<Tensor> &vision_hidden_states,
                                               const std::shared_ptr<Tensor> &image_indices) {
    CHECK_EQ(input_ids->Dims().size(), 2);
    CHECK_EQ(vision_hidden_states->Dims().size(), 3);
    CHECK_EQ(image_indices->Dims().size(), 2);
    CHECK_EQ(input_ids->Dims()[0], vision_hidden_states->Dims()[0])
        << "The current FM9GV training path supports one image per sequence";
    CHECK_EQ(input_ids->Dims()[0], image_indices->Dims()[0]);
    CHECK_EQ(vision_hidden_states->Dims()[1], image_indices->Dims()[1]);
    return (*modules_[kLLMName])({input_ids, vision_hidden_states, image_indices});
}

std::vector<std::shared_ptr<Tensor>> MultimodalModel::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK(inputs.size() == 1 || inputs.size() == 3 || inputs.size() == 4)
        << "FM9GV multimodal forward expects {input_ids}, "
           "{input_ids, vision_hidden_states, image_indices}, or "
           "{input_ids, pixel_values, target_sizes, image_indices}";
    if (inputs.size() == 1) {
        return (*modules_[kLLMName])({inputs[0]});
    }
    if (inputs.size() == 3) {
        return ForwardWithVisionHiddenStates(inputs[0], inputs[1], inputs[2]);
    }
    return ForwardWithVisionHiddenStates(inputs[0], EncodeVision(inputs[1], inputs[2]), inputs[3]);
}

void MultimodalModel::SetVisionRequiresGrad(bool requires_grad) {
    for (const auto &parameter : modules_[kVPMName]->Parameters()) { parameter->set_requires_grad(requires_grad); }
}

void MultimodalModel::SetLLMRequiresGrad(bool requires_grad) {
    for (const auto &parameter : modules_[kLLMName]->Parameters()) { parameter->set_requires_grad(requires_grad); }
}

} // namespace fm9gv
