#include "infini_train/include/autograd/matmul.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Matmul::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &input1 = input_tensors[0];
    const auto &input2 = input_tensors[1];

    auto device = input1->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "MatmulForward"}, input1, input2)};
}

void Matmul::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &input1 = input_tensors[0];
    const auto &input2 = input_tensors[1];
    const auto &output = output_tensors[0];

    // grad_input1 = grad_output @ input2^T, so input2 is needed
    // grad_input2 = grad_output^T @ input1, so input1 is needed
    bool need_grad_input1 = needs_input_grad_.size() > 0 && needs_input_grad_[0];
    bool need_grad_input2 = needs_input_grad_.size() > 1 && needs_input_grad_[1];

    SaveForBackward({need_grad_input2 ? input1 : nullptr, need_grad_input1 ? input2 : nullptr});
    input1_dims_ = input1->Dims();
    input2_dims_ = input2->Dims();
    out_features_ = output->Dims()[0];
}

std::vector<std::shared_ptr<Tensor>> Matmul::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(SavedTensorsSize(), 2);
    const auto &input1 = GetSavedTensor(0);
    const auto &input2 = GetSavedTensor(1);
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    CHECK(!needs_input_grad_.empty()) << "needs_input_grad_ not populated in Matmul::Backward";
    bool need_grad_input1 = needs_input_grad_.size() > 0 && needs_input_grad_[0];
    bool need_grad_input2 = needs_input_grad_.size() > 1 && needs_input_grad_[1];

    auto device = grad_output->GetDevice().type();

    std::shared_ptr<Tensor> grad_input = nullptr;
    std::shared_ptr<Tensor> grad_other = nullptr;

    if (need_grad_input1) {
        CHECK(input2 != nullptr) << "input2 not saved but need_grad_input1 is true";
        grad_input = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "MatmulBackwardInput"}, input2,
                                                                          grad_output, input1_dims_);
    }
    if (need_grad_input2) {
        CHECK(input1 != nullptr) << "input1 not saved but need_grad_input2 is true";
        grad_other = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({device, "MatmulBackwardOther"}, input1,
                                                                          grad_output, input2_dims_);
    }

    return {grad_input, grad_other};
}
} // namespace infini_train::autograd
