#include "infini_train/include/autograd/comm.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {

Scatter::Scatter(const std::vector<Device> &target_gpus, int64_t dim,
                 const infini_train::nn::parallel::ProcessGroup *pg)
    : autograd::Function(kType), target_gpus_(target_gpus), dim_(dim),
      pg_(pg ? pg : infini_train::nn::parallel::ProcessGroupFactory::Instance()->GetDefaultProcessGroup()) {}

std::vector<std::shared_ptr<Tensor>> Scatter::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    const auto &input = input_tensors[0];
    std::vector<std::shared_ptr<Tensor>> output_tensors;
    auto device = input->GetDevice()->Type();
    output_tensors = pg_->Scatter(input, target_gpus_, dim_);
    return output_tensors;
}

void Scatter::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                           const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    input_device_ = input_tensors[0]->GetDevice();
}

std::vector<std::shared_ptr<Tensor>> Scatter::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    return std::make_shared<Gather>(input_device_, dim_)->Apply(grad_outputs);
}

Gather::Gather(Device target_device, int64_t dim, const infini_train::nn::parallel::ProcessGroup *pg)
    : autograd::Function(kType), target_device_(target_device), dim_(dim),
      pg_(pg ? pg : infini_train::nn::parallel::ProcessGroupFactory::Instance()->GetDefaultProcessGroup()) {}

std::vector<std::shared_ptr<Tensor>> Gather::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    for (const auto &tensor : input_tensors) {
        CHECK_NE(static_cast<int>(tensor->GetDevice()->Type()), static_cast<int>(Device::DeviceType::kCPU))
            << "Gather function not implemented for CPU tensors";
    }
    if (dim_ == 0 && input_tensors[0]->Dims().size() == 0) {
        // FIXME(dcj): Here it is assumed that all tensors involved in the gather operation have the same shape.
        unsqueezed_scalar_ = true;
        LOG(WARNING) << "Was asked to gather along dimension 0, but all "
                        "input tensors were scalars; will instead unsqueeze "
                        "and return a vector.";
        // TODO(dcj): do unsqueeze here
    } else {
        unsqueezed_scalar_ = false;
    }
    auto device = input_tensors[0]->GetDevice()->Type();
    return {pg_->Gather(input_tensors, target_device_, dim_)};
}

void Gather::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    for (const auto &tensor : input_tensors) { input_gpus_.push_back(tensor->GetDevice()); }
}

std::vector<std::shared_ptr<Tensor>> Gather::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // TODO(dcj): do squeeze here if unsqueezed_scalar_ is true
    return std::make_shared<Scatter>(std::vector<Device>{input_gpus_}, dim_)->Apply(grad_outputs);
}

Broadcast::Broadcast(const std::vector<Device> &target_gpus, const infini_train::nn::parallel::ProcessGroup *pg)
    : autograd::Function(kType), target_gpus_(target_gpus),
      pg_(pg ? pg : infini_train::nn::parallel::ProcessGroupFactory::Instance()->GetDefaultProcessGroup()) {}

std::vector<std::shared_ptr<Tensor>> Broadcast::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    if (target_gpus_.size() == 0) {
        return {};
    }

    input_device_ = input_tensors[0]->GetDevice();

    for (const auto &tensor : input_tensors) {
        CHECK(!tensor->GetDevice()->IsCPU()) << "Broadcast function not implemented for CPU tensors";
        CHECK(tensor->GetDevice()->Type() == input_device_->Type())
            << "Broadcast function not implemented for tensors on different device type";
    }

    // TODO(dcj): mark non differentiable
    return pg_->BroadCast(input_tensors);
}

void Broadcast::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                             const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    num_inputs_ = input_tensors.size();
}

std::vector<std::shared_ptr<Tensor>> Broadcast::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    return std::make_shared<ReduceAddCoalesced>(input_device_, num_inputs_)->Apply(grad_outputs);
}

ReduceAddCoalesced::ReduceAddCoalesced(Device destination, int64_t num_inputs,
                                       const infini_train::nn::parallel::ProcessGroup *pg)
    : autograd::Function(kType), destination_(destination), num_inputs_(num_inputs),
      pg_(pg ? pg : infini_train::nn::parallel::ProcessGroupFactory::Instance()->GetDefaultProcessGroup()) {}

std::vector<std::shared_ptr<Tensor>>
ReduceAddCoalesced::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    std::vector<std::vector<std::shared_ptr<Tensor>>> tensor_reshaped(input_tensors.size() / num_inputs_);
    for (auto device_idx = 0; device_idx < input_tensors.size() / num_inputs_; ++device_idx) {
        tensor_reshaped[device_idx].resize(num_inputs_);
        for (auto tensor_idx = 0; tensor_idx < num_inputs_; ++tensor_idx) {
            tensor_reshaped[device_idx][tensor_idx] = input_tensors[device_idx * num_inputs_ + tensor_idx];
        }
    }

    return pg_->ReduceAddCoalesced(tensor_reshaped, destination_);
}

void ReduceAddCoalesced::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    for (const auto &tensor : input_tensors) { target_gpus_.push_back(tensor->GetDevice()); }
}

std::vector<std::shared_ptr<Tensor>>
ReduceAddCoalesced::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    return std::make_shared<Broadcast>(target_gpus_)->Apply(grad_outputs);
}
} // namespace infini_train::autograd
