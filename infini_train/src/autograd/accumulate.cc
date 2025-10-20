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
    if (grad_output) {
        if (grad) {
            if (tensor_->ConsumeGradOverwriteFlag()) {
                tensor_->grad()->CopyFrom(grad_output);
            } else {
                auto device = grad->GetDevice();
                device->SetDevice();
                auto kernel = Dispatcher::Instance().GetKernel({device->Type(), "AccumulateGrad"});
                kernel.Call<void>(grad_output, learning_rate_, grad);
            }
        } else {
            auto new_grad = std::make_shared<Tensor>(tensor_->Dims(), grad_output->Dtype(), grad_output->GetDevice());
            // FIXME(zbl): build from tensor_.buffer
            new_grad->CopyFrom(grad_output);
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
