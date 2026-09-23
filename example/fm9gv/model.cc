#include "example/fm9gv/model.h"

#include <string>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

namespace fm9gv {

using infini_train::Tensor;
namespace nn = infini_train::nn;

Model::Model(const nn::TransformerConfig &config) : nn::TransformerModel(config) {}

std::vector<std::shared_ptr<Tensor>> Model::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK(inputs.size() == 1 || inputs.size() == 3)
        << "FM9GV forward expects input_ids or {input_ids, vision_hidden_states, image_indices}";
    if (inputs.size() == 1) { return nn::TransformerModel::Forward(inputs); }

    CHECK_EQ(nn::parallel::global::GetPipelineParallelSize(), 1)
        << "FM9GV multimodal embedding replacement currently requires PP=1";
    CHECK_EQ(nn::parallel::global::GetVirtualPipelineParallelSize(), 1)
        << "FM9GV multimodal embedding replacement currently requires VPP=1";

    auto hidden = (*modules_[kPPFirstStageName])({inputs[0]})[0];
    auto image_indices = inputs[2]->Unsqueeze(2)->RepeatInterleave(inputs[1]->Dims().back(), 2);
    hidden = nn::function::Scatter(hidden, 1, image_indices, inputs[1]);
    hidden = (*modules_[std::string(kPPChunkNamePrefix) + "0"])({hidden})[0];
    return (*modules_[kPPLastStageName])({hidden});
}

} // namespace fm9gv
