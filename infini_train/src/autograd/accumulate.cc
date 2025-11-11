#include "infini_train/include/autograd/accumulate.h"

#include "glog/logging.h"

#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
AccumulateGrad::AccumulateGrad(std::shared_ptr<Tensor> tensor, float learning_rate)
    : tensor_(tensor), learning_rate_(learning_rate) {}

std::vector<std::shared_ptr<Tensor>> AccumulateGrad::Forward(const std::vector<std::shared_ptr<Tensor>> &) {
    LOG(FATAL) << "AccumulateGrad::Forward shall not be called directly!";
    return {};
}

std::vector<std::shared_ptr<Tensor>>
AccumulateGrad::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    auto grad_output = grad_outputs[0];

    auto grad = tensor_->grad();
    auto device = tensor_->GetDevice();
    device->SetDevice();

    if (grad_output) {
        if (grad) {
            auto kernel = Dispatcher::Instance().GetKernel({device->Type(), "AccumulateGrad"});
            kernel.Call<void>(grad_output, learning_rate_, grad);
        } else {
            auto new_grad = std::make_shared<Tensor>(*grad_output.get(), 0, grad_output->Dims());
            tensor_->set_grad(std::move(new_grad));
        }
        auto hook = tensor_->post_accumulate_grad_hook();
        if (hook != nullptr) {
            (*hook)(tensor_->grad());
        }
        tensor_->ResetAccumulator();
    }
    return {};
}
} // namespace infini_train::autograd
