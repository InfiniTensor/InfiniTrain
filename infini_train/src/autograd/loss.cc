#include "infini_train/include/autograd/loss.h"

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
namespace {
int64_t CountValidTargets(const std::shared_ptr<Tensor> &target, int64_t ignore_index, int64_t num_classes) {
    auto target_cpu = target->To(Device());
    int64_t valid_count = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(target_cpu.NumElements()); ++i) {
        int64_t value = 0;
        if (target_cpu.Dtype() == DataType::kUINT8) {
            value = static_cast<const uint8_t *>(target_cpu.DataPtr())[i];
        } else if (target_cpu.Dtype() == DataType::kINT64) {
            value = static_cast<const int64_t *>(target_cpu.DataPtr())[i];
        } else {
            LOG(FATAL) << "Unsupported target data type: " << static_cast<int>(target_cpu.Dtype());
        }
        if (value == ignore_index) { continue; }
        CHECK_GE(value, 0);
        CHECK_LT(value, num_classes);
        ++valid_count;
    }
    return valid_count;
}
} // namespace

std::vector<std::shared_ptr<Tensor>> CrossEntropy::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 2);
    const auto &input = input_tensors[0];
    const auto &target = input_tensors[1];

    valid_count_ = CountValidTargets(target, ignore_index_, input->Dims().back());
    auto device = input->GetDevice().type();
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {device, "CrossEntropyForward"}, input, target, ignore_index_, valid_count_)};
}

void CrossEntropy::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                const std::vector<std::shared_ptr<Tensor>> &) {
    const auto &input = input_tensors[0];
    const auto &target = input_tensors[1];
    ctx_.SaveForBackward({input, target});
}

std::vector<std::shared_ptr<Tensor>> CrossEntropy::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    auto saved_tensors = ctx_.GetSavedTensors();
    CHECK_EQ(saved_tensors.size(), 2);
    const auto &input = saved_tensors[0];
    const auto &target = saved_tensors[1];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = input->GetDevice().type();
    auto grad_input = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {device, "CrossEntropyBackward"}, input, target, grad_output, ignore_index_, valid_count_);
    return {grad_input, nullptr};
}
} // namespace infini_train::autograd
