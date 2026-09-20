#include "infini_train/include/autograd/conv2d.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"


namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Conv2d::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_GE(input_tensors.size(), 2);
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    const auto &bias = input_tensors.size() == 3 ? input_tensors[2] : nullptr;
    
    // 按照输入设备分发到CPU和CUDA的Conv2dForward kernel
    auto device = input->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "Conv2dForward"}, input, weight, bias, stride_, padding_)};

}

void Conv2d::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors, 
                        const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    const auto &weight = input_tensors[1];
    bool need_input = ctx_.needs_input_grad().size() > 0 && ctx_.needs_input_grad()[0];
    bool need_weight = ctx_.needs_input_grad().size() > 1 && ctx_.needs_input_grad()[1];

    // 反向1grad_input需要weight, 反向2grad_weight需要input
    // 哪边不需要梯度就存nullptr, 省显存/内存
    ctx_.SaveForBackward({need_weight ? input : nullptr, need_input ? weight : nullptr});

    bias_ = input_tensors.size() == 3;
    input_dims_ = input->Dims();
    weight_dims_ = weight->Dims();
}


std::vector<std::shared_ptr<Tensor>> Conv2d::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    auto saved_tensors = ctx_.GetSavedTensors();
    CHECK_EQ(saved_tensors.size(), 2);
    const auto &input = saved_tensors[0];
    const auto &weight = saved_tensors[1];

    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    CHECK(!ctx_.needs_input_grad().empty()) << "needs_input_grad not populated in Conv2d::Backward";
    bool need_grad_input = ctx_.needs_input_grad()[0];
    bool need_grad_weight = ctx_.needs_input_grad().size() > 1 && ctx_.needs_input_grad()[1];
    bool need_grad_bias = bias_ && ctx_.needs_input_grad().size() > 2 && ctx_.needs_input_grad()[2];

    auto device = grad_output->GetDevice().type();

    std::shared_ptr<Tensor> grad_input = nullptr;
    std::shared_ptr<Tensor> grad_weight = nullptr;
    std::shared_ptr<Tensor> grad_bias = nullptr;

    // 反向1： 输入的梯度
    if (need_grad_input) {
        grad_input = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "Conv2dBackwardInput"}, weight, grad_output, input_dims_, stride_, padding_);
    }

    // 反向2：权重的梯度
    if (need_grad_weight) {
        grad_weight = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "Conv2dBackwardWeight"}, input, grad_output, weight_dims_, stride_, padding_);
    }

    // 反向3：偏置的梯度
    if (need_grad_bias) {
        grad_bias = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "Conv2dBackwardBias"}, grad_output, weight_dims_[0]);
    }


    if (bias_) {
        return {grad_input, grad_weight, grad_bias};
    } else {
        return {grad_input, grad_weight};
    }
}



}