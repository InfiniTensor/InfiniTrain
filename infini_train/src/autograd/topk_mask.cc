#include "infini_train/include/autograd/topk_mask.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> TopKMask::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    CHECK_GT(topk_, 0);
    const auto &input = input_tensors[0];
    auto device = input->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "TopKMaskForward"}, input, topk_)};
}

void TopKMask::SetupContext(const std::vector<std::shared_ptr<Tensor>> &,
                            const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    saved_tensors_ = {output_tensors[0]};
}

std::vector<std::shared_ptr<Tensor>> TopKMask::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];
    const auto &mask_values = saved_tensors_[0];
    auto device = grad_output->GetDevice().type();
    return {
        Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "TopKMaskBackward"}, grad_output, mask_values)};
}

} // namespace infini_train::autograd
