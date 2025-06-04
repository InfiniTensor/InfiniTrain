#include "infini_train/include/nn/modules/module.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
namespace {
class Broadcast : public autograd::Function {
public:
    static constexpr char kType[] = "BroadcastFunction";

    explicit Broadcast(const std::vector<const Device *> &target_gpus)
        : autograd::Function(kType), target_gpus_(target_gpus) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    std::vector<const Device *> target_gpus_;
    int64_t num_inputs_ = 0;
    const Device *input_device_ = nullptr;
};

class ReduceAddCoalesced : public autograd::Function {
public:
    static constexpr char kType[] = "ReduceAddCoalescedFunction";

    explicit ReduceAddCoalesced(const Device *destination, int64_t num_inputs)
        : autograd::Function(kType), destination_(destination), num_inputs_(num_inputs) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const Device *destination_ = nullptr;
    std::vector<const Device *> target_gpus_;
    int64_t num_inputs_ = 0;
};

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
    auto kernel = Dispatcher::Instance().GetKernel({input_device_->Type(), "CommNcclBroadcast"});
    return kernel.Call<std::vector<std::shared_ptr<Tensor>>>(input_tensors, target_gpus_);
}

void Broadcast::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                             const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    num_inputs_ = input_tensors.size();
}

std::vector<std::shared_ptr<Tensor>> Broadcast::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // FIXME(dcj): implement this
    return std::make_shared<ReduceAddCoalesced>(input_device_, num_inputs_)->Apply(grad_outputs);
}

std::vector<std::shared_ptr<Tensor>>
ReduceAddCoalesced::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    std::vector<std::vector<std::shared_ptr<Tensor>>> tensor_reshaped(input_tensors.size() / num_inputs_);
    for (auto device_idx = 0; device_idx < input_tensors.size() / num_inputs_; ++device_idx) {
        tensor_reshaped[device_idx].resize(num_inputs_);
        for (auto tensor_idx = 0; tensor_idx < num_inputs_; ++tensor_idx) {
            tensor_reshaped[device_idx][tensor_idx] = input_tensors[device_idx * num_inputs_ + tensor_idx];
        }
    }

    auto kernel
        = Dispatcher::Instance().GetKernel({input_tensors[0]->GetDevice()->Type(), "CommNcclReduceAddCoalesced"});
    return kernel.Call<std::vector<std::shared_ptr<Tensor>>>(tensor_reshaped, destination_);
}

void ReduceAddCoalesced::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    for (const auto &tensor : input_tensors) { target_gpus_.push_back(tensor->GetDevice()); }
}

std::vector<std::shared_ptr<Tensor>>
ReduceAddCoalesced::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // FIXME(dcj): implement this
    return std::make_shared<Broadcast>(target_gpus_)->Apply(grad_outputs);
}

std::vector<std::vector<std::shared_ptr<Tensor>>>
BroadcastCoalescedReshape(const std::vector<std::shared_ptr<Tensor>> &tensors,
                          const std::vector<const Device *> &devices) {
    auto tensor_copies = std::make_shared<Broadcast>(devices)->Apply(tensors);
    std::vector<std::vector<std::shared_ptr<Tensor>>> tensor_copies_reshaped(devices.size());
    for (int replica_idx = 0; replica_idx < devices.size(); ++replica_idx) {
        tensor_copies_reshaped[replica_idx].resize(tensors.size());
        for (int tensor_idx = 0; tensor_idx < tensors.size(); ++tensor_idx) {
            tensor_copies_reshaped[replica_idx][tensor_idx] = tensor_copies[replica_idx * tensors.size() + tensor_idx];
        }
    }
    return tensor_copies_reshaped;
}
} // namespace

// FIXME(dcj): should replicate buffer_
std::vector<std::shared_ptr<Module>> Replicate(const std::shared_ptr<Module> &network,
                                               const std::vector<const Device *> &devices) {
    const int num_replicas = devices.size();

    // FIXME(dcj): replicate buffer_!!!!!!!!!!!!!
    auto params = network->Parameters();
    std::unordered_map<Tensor *, int> param_indices;
    for (int idx = 0; idx < params.size(); ++idx) { param_indices[params[idx].get()] = idx; }
    auto param_copies = BroadcastCoalescedReshape(params, devices);

    auto modules = network->modules();
    std::vector<std::vector<std::shared_ptr<Module>>> module_copies(num_replicas);
    std::unordered_map<Module *, int> module_indices;

    for (int idx = 0; idx < modules.size(); ++idx) {
        auto &module = modules[idx];
        module_indices[module.get()] = idx;
        for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
            module_copies[replica_idx].push_back(module->ReplicateForDataParallel(replica_idx));
        }
    }

    for (int idx = 0; idx < modules.size(); ++idx) {
        auto &module = modules[idx];
        for (auto &[name, child] : module->modules_) {
            const auto module_idx = module_indices[child.get()];
            for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
                auto &replica = module_copies[replica_idx][idx];
                replica->modules_[name] = module_copies[replica_idx][module_idx];
            }
        }
        for (auto &[name, param] : module->parameters_) {
            const auto param_idx = param_indices[param.get()];
            for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
                auto &replica = module_copies[replica_idx][idx];
                replica->parameters_[name] = param_copies[replica_idx][param_idx];
            }
        }
    }

    std::vector<std::shared_ptr<Module>> replicas;
    for (int idx = 0; idx < num_replicas; ++idx) { replicas.push_back(std::move(module_copies[idx][0])); }
    return replicas;
}
} // namespace infini_train::nn::parallel
