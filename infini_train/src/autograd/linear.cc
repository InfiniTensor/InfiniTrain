#include "infini_train/include/autograd/linear.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
namespace linear {
void ForwardOut(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &bias) {
    Dispatcher::Instance().Call<void>({input->GetDevice().type(), "LinearForwardOut"}, input, weight, output, true,
                                      bias);
}

std::shared_ptr<Tensor> Forward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                const std::shared_ptr<Tensor> &bias) {
    return Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({input->GetDevice().type(), "LinearForward"}, input,
                                                                weight, true, bias);
}

std::shared_ptr<Tensor> BackwardInput(const std::shared_ptr<Tensor> &weight, const std::shared_ptr<Tensor> &grad_output,
                                      const std::vector<int64_t> &input_dims) {
    return Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {grad_output->GetDevice().type(), "LinearBackwardInput"}, weight, grad_output, true, weight->Dims()[1],
        weight->Dims()[0], input_dims);
}

void BackwardInputOut(const std::shared_ptr<Tensor> &weight, const std::shared_ptr<Tensor> &grad_output,
                      const std::shared_ptr<Tensor> &grad_input) {
    Dispatcher::Instance().Call<void>({grad_output->GetDevice().type(), "LinearBackwardInputOut"}, weight, grad_output,
                                      grad_input, true, weight->Dims()[1], weight->Dims()[0]);
}

std::shared_ptr<Tensor> BackwardWeight(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &grad_output,
                                       int64_t in_features, int64_t out_features) {
    return Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {grad_output->GetDevice().type(), "LinearBackwardWeight"}, input, grad_output, true, in_features, out_features);
}

std::shared_ptr<Tensor> BackwardBias(const std::shared_ptr<Tensor> &grad_output, int64_t out_features) {
    return Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({grad_output->GetDevice().type(), "LinearBackwardBias"},
                                                                grad_output, out_features);
}
} // namespace linear

std::vector<std::shared_ptr<Tensor>> Linear::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_GE(input_tensors.size(), 2);
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    const auto &bias = input_tensors.size() == 3 ? input_tensors[2] : nullptr;

    return {linear::Forward(input, weight, bias)};
}

void Linear::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    bool need_input = ctx_.needs_input_grad().size() > 0 && ctx_.needs_input_grad()[0];
    bool need_weight = ctx_.needs_input_grad().size() > 1 && ctx_.needs_input_grad()[1];

    // grad_input needs weight, grad_weight needs input
    ctx_.SaveForBackward({need_weight ? input : nullptr, need_input ? weight : nullptr});

    bias_ = input_tensors.size() == 3;
    in_features_ = weight->Dims()[1];
    out_features_ = weight->Dims()[0];
    input_dims_ = input->Dims();
}

std::vector<std::shared_ptr<Tensor>> Linear::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    auto saved_tensors = ctx_.GetSavedTensors();
    CHECK_EQ(saved_tensors.size(), 2);
    const auto &input = saved_tensors[0];
    const auto &weight = saved_tensors[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    CHECK(!ctx_.needs_input_grad().empty()) << "needs_input_grad not populated in Linear::Backward";
    bool need_grad_input = ctx_.needs_input_grad()[0];
    bool need_grad_weight = ctx_.needs_input_grad().size() > 1 && ctx_.needs_input_grad()[1];
    bool need_grad_bias = bias_ && ctx_.needs_input_grad().size() > 2 && ctx_.needs_input_grad()[2];

    std::shared_ptr<Tensor> grad_input = nullptr;
    std::shared_ptr<Tensor> grad_weight = nullptr;
    std::shared_ptr<Tensor> grad_bias = nullptr;

    if (need_grad_input) {
        grad_input = linear::BackwardInput(weight, grad_output, input_dims_);
    }
    if (need_grad_weight) {
        grad_weight = linear::BackwardWeight(input, grad_output, in_features_, out_features_);
    }
    if (need_grad_bias) {
        grad_bias = linear::BackwardBias(grad_output, out_features_);
    }

    if (bias_) {
        return {grad_input, grad_weight, grad_bias};
    } else {
        return {grad_input, grad_weight};
    }
}
} // namespace infini_train::autograd
