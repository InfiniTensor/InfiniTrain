#include "infini_train/include/autograd/activations.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> ReLU::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK_EQ(inputs.size(), 1);
    CHECK(inputs[0]);
    CHECK(inputs[0]->Dtype() == DataType::kFLOAT32) << "ReLU supports FP32 only";
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({inputs[0]->GetDevice().type(), "ReLUForward"},
                                                                 inputs[0])};
}

void ReLU::SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                        const std::vector<std::shared_ptr<Tensor>> &) {
    ctx_.SaveForBackward({inputs[0]});
}

std::vector<std::shared_ptr<Tensor>> ReLU::Backward(const std::vector<std::shared_ptr<Tensor>> &grads) {
    CHECK_EQ(grads.size(), 1);
    const auto input = ctx_.GetSavedTensors()[0];
    CHECK(grads[0]);
    CHECK(grads[0]->Dims() == input->Dims());
    CHECK(grads[0]->Dtype() == input->Dtype());
    CHECK(grads[0]->GetDevice() == input->GetDevice());
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({input->GetDevice().type(), "ReLUBackward"}, input,
                                                                 grads[0])};
}

std::vector<std::shared_ptr<Tensor>> Sigmoid::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "SigmoidForward"}, input)};
}

void Sigmoid::SetupContext(const std::vector<std::shared_ptr<Tensor>> &,
                           const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &output = output_tensors[0];
    ctx_.SaveForBackward({output});
}

std::vector<std::shared_ptr<Tensor>> Sigmoid::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    auto saved_tensors = ctx_.GetSavedTensors();
    CHECK_EQ(saved_tensors.size(), 1);
    const auto &output = saved_tensors[0];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = output->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "SigmoidBackward"}, output, grad_output)};
}

std::vector<std::shared_ptr<Tensor>> SwiGLU::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];
    CHECK_GT(input->Dims().size(), 0);
    CHECK_EQ(input->Dims().back() % 2, 0) << "SwiGLU expects an even last dimension";

    auto device = input->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "SwiGLUForward"}, input)};
}

void SwiGLU::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &) {
    ctx_.SaveForBackward({input_tensors[0]});
}

std::vector<std::shared_ptr<Tensor>> SwiGLU::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    auto saved_tensors = ctx_.GetSavedTensors();
    CHECK_EQ(saved_tensors.size(), 1);
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &input = saved_tensors[0];
    const auto &grad_output = grad_outputs[0];

    auto device = input->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "SwiGLUBackward"}, input, grad_output)};
}
} // namespace infini_train::autograd
