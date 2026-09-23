#include "example/fm9gv/resampler_ops.h"

#include <utility>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace fm9gv {

using infini_train::Dispatcher;
using infini_train::Tensor;

InterpolateFunction::InterpolateFunction(std::vector<int64_t> size, std::string mode,
                                         std::optional<bool> align_corners, bool antialias)
    : Function("Interpolate"), size_(std::move(size)), mode_(std::move(mode)),
      align_corners_(align_corners), antialias_(antialias) {}

std::vector<std::shared_ptr<Tensor>>
InterpolateFunction::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK_EQ(inputs.size(), 1);
    input_dims_ = inputs[0]->Dims();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {inputs[0]->GetDevice().type(), "InterpolateForward"}, inputs[0], size_, mode_, align_corners_,
        antialias_)};
}

void InterpolateFunction::SetupContext(const std::vector<std::shared_ptr<Tensor>> &,
                                       const std::vector<std::shared_ptr<Tensor>> &) {}

std::vector<std::shared_ptr<Tensor>>
InterpolateFunction::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {grad_outputs[0]->GetDevice().type(), "InterpolateBackward"}, grad_outputs[0], input_dims_, mode_,
        align_corners_, antialias_)};
}

RoiAlignFunction::RoiAlignFunction(std::vector<int64_t> output_size, double spatial_scale,
                                   int64_t sampling_ratio, bool aligned)
    : Function("RoiAlign"), output_size_(std::move(output_size)), spatial_scale_(spatial_scale),
      sampling_ratio_(sampling_ratio), aligned_(aligned) {}

std::vector<std::shared_ptr<Tensor>>
RoiAlignFunction::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK_EQ(inputs.size(), 2);
    input_dims_ = inputs[0]->Dims();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {inputs[0]->GetDevice().type(), "RoiAlignForward"}, inputs[0], inputs[1], output_size_, spatial_scale_,
        sampling_ratio_, aligned_)};
}

void RoiAlignFunction::SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                                    const std::vector<std::shared_ptr<Tensor>> &) {
    ctx_.SaveForBackward({inputs[1]});
}

std::vector<std::shared_ptr<Tensor>>
RoiAlignFunction::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto boxes = ctx_.GetSavedTensors()[0];
    auto grad_input = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {grad_outputs[0]->GetDevice().type(), "RoiAlignBackward"}, grad_outputs[0], boxes, input_dims_,
        output_size_, spatial_scale_, sampling_ratio_, aligned_);
    return {grad_input, nullptr};
}

std::shared_ptr<Tensor> Interpolate(const std::shared_ptr<Tensor> &input,
                                    const std::optional<std::vector<int64_t>> &size,
                                    const std::optional<std::vector<double>> &scale_factor,
                                    const std::string &mode, std::optional<bool> align_corners,
                                    std::optional<bool> recompute_scale_factor, bool antialias) {
    CHECK(size.has_value()) << "DCU interpolate currently requires an explicit size";
    CHECK(!scale_factor.has_value()) << "Specify either size or scale_factor, not both";
    CHECK(!recompute_scale_factor.value_or(false));
    CHECK_EQ(size->size(), 2);
    CHECK_EQ(mode, "bilinear");
    CHECK(!align_corners.value_or(false));
    CHECK(!antialias);
    return std::make_shared<InterpolateFunction>(*size, mode, align_corners, antialias)->Apply({input})[0];
}

std::shared_ptr<Tensor> RoiAlign(const std::shared_ptr<Tensor> &input,
                                 const std::shared_ptr<Tensor> &boxes,
                                 const std::vector<int64_t> &output_size, double spatial_scale,
                                 int64_t sampling_ratio, bool aligned) {
    CHECK_EQ(output_size.size(), 2);
    return std::make_shared<RoiAlignFunction>(output_size, spatial_scale, sampling_ratio, aligned)
        ->Apply({input, boxes})[0];
}

} // namespace fm9gv
