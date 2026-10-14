#include "infini_train/include/autograd/conv2d.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> Conv2d::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK(input_tensors.size() == 2 || input_tensors.size() == 3);
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    const auto bias = input_tensors.size() == 3 ? input_tensors[2] : nullptr;
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({input->GetDevice().type(), "Conv2dForward"}, input,
                                                                 weight, bias, stride_, padding_)};
}

void Conv2d::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &needs_input_grad = ctx_.needs_input_grad();
    const bool need_input = !needs_input_grad.empty() && needs_input_grad[0];
    const bool need_weight = needs_input_grad.size() > 1 && needs_input_grad[1];
    ctx_.SaveForBackward({need_weight ? input_tensors[0] : nullptr, need_input ? input_tensors[1] : nullptr});
    bias_ = input_tensors.size() == 3;
    input_dims_ = input_tensors[0]->Dims();
    weight_dims_ = input_tensors[1]->Dims();
}

std::vector<std::shared_ptr<Tensor>> Conv2d::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];
    const auto saved = ctx_.GetSavedTensors();
    const bool need_input = ctx_.needs_input_grad()[0];
    const bool need_weight = ctx_.needs_input_grad()[1];
    const bool need_bias = bias_ && ctx_.needs_input_grad()[2];
    const auto device = grad_output->GetDevice().type();

    std::shared_ptr<Tensor> grad_input;
    std::shared_ptr<Tensor> grad_weight;
    std::shared_ptr<Tensor> grad_bias;
    if (need_input) {
        grad_input = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "Conv2dBackwardInput"}, saved[1],
                                                                          grad_output, input_dims_, stride_, padding_);
    }
    if (need_weight) {
        grad_weight = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
            {device, "Conv2dBackwardWeight"}, saved[0], grad_output, weight_dims_, stride_, padding_);
    }
    if (need_bias) {
        grad_bias = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "Conv2dBackwardBias"}, grad_output);
    }
    return bias_ ? std::vector<std::shared_ptr<Tensor>>{grad_input, grad_weight, grad_bias}
                 : std::vector<std::shared_ptr<Tensor>>{grad_input, grad_weight};
}

} // namespace infini_train::autograd
