#include "infini_train/include/autograd/topk.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> TopK::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    CHECK_GT(topk_, 0);
    const auto &input = input_tensors[0];
    auto device = input->GetDevice().type();
    auto topk_outputs = Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {device, "TopKForward"}, input, topk_, dim_, largest_, sorted_);
    CHECK_EQ(topk_outputs.size(), 2);
    return topk_outputs;
}

void TopK::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                        const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    CHECK_EQ(output_tensors.size(), 2);
    input_dims_ = input_tensors[0]->Dims();
    ctx_.SaveForBackward({output_tensors[1]});
    ctx_.MarkNonDifferentiable({output_tensors[1]});
}

std::vector<std::shared_ptr<Tensor>> TopK::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 2);
    const auto &top_grad = grad_outputs[0];
    auto saved_tensors = ctx_.GetSavedTensors();
    const auto &top_indices = saved_tensors[0];
    auto device = top_grad->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "TopKBackward"}, top_grad, top_indices,
                                                                 input_dims_, dim_)};
}

} // namespace infini_train::autograd
