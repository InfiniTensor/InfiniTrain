#include "infini_train/include/autograd/conv2d.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Conv2d::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_GE(input_tensors.size(), 2);
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    std::shared_ptr<Tensor> bias = nullptr;
    if (input_tensors.size() == 3) {
        bias = input_tensors[2];
    }

    auto device = input->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {device, "Conv2dForward"}, input, weight, bias, stride_, padding_)};
}

void Conv2d::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    bool need_input = ctx_.needs_input_grad().size() > 0 && ctx_.needs_input_grad()[0];
    bool need_weight = ctx_.needs_input_grad().size() > 1 && ctx_.needs_input_grad()[1];

    ctx_.SaveForBackward({need_weight ? input : nullptr, need_input ? weight : nullptr});

    input_dims_ = input->Dims();
    LOG(ERROR) << "SetupContext input_dims_ size: " << input_dims_.size();
    in_channels_ = weight->Dims()[1];
    out_channels_ = weight->Dims()[0];
    kernel_size_ = weight->Dims()[2];
    bias_ = input_tensors.size() == 3;
}

std::vector<std::shared_ptr<Tensor>> Conv2d::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    auto saved_tensors = ctx_.GetSavedTensors();
    LOG(ERROR) << "Backward input_dims_ size: " << input_dims_.size();
    CHECK_EQ(saved_tensors.size(), 2);
    const auto &input = saved_tensors[0];
    const auto &weight = saved_tensors[1];
    LOG(ERROR) << "Backward weight ptr: " << weight.get();
    if (weight) {
        LOG(ERROR) << "Backward weight dims size: " << weight->Dims().size();
    } else {
        LOG(ERROR) << "Backward weight is null!";
    }
    const auto &grad_output = grad_outputs[0];

    CHECK(!ctx_.needs_input_grad().empty()) << "needs_input_grad not populated in Conv2d::Backward";
    bool need_grad_input = ctx_.needs_input_grad()[0];
    bool need_grad_weight = ctx_.needs_input_grad().size() > 1 && ctx_.needs_input_grad()[1];
    bool need_grad_bias = bias_ && ctx_.needs_input_grad().size() > 2 && ctx_.needs_input_grad()[2];

    auto device = grad_output->GetDevice().type();

    std::shared_ptr<Tensor> grad_input = nullptr;
    std::shared_ptr<Tensor> grad_weight = nullptr;
    std::shared_ptr<Tensor> grad_bias = nullptr;

    if (need_grad_input) {
        grad_input = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
            {device, "Conv2dBackwardInput"}, weight, grad_output, input_dims_, stride_, padding_);
    }
    if (need_grad_weight && weight) {
    grad_weight = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {device, "Conv2dBackwardWeight"}, input, grad_output, weight->Dims(), stride_, padding_);
    }
    if (need_grad_bias) {
        grad_bias = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
            {device, "Conv2dBackwardBias"}, grad_output, out_channels_);
    }

    if (bias_) {
        return {grad_input, grad_weight, grad_bias};
    } else {
        return {grad_input, grad_weight};
    }
}
} // namespace infini_train::autograd