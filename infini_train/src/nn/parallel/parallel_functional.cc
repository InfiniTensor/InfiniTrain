#include "infini_train/include/nn/parallel/parallel_functional.h"

#include <memory>
#include <unordered_map>
#include <vector>

#include "infini_train/include/autograd/comm.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel::function {

std::shared_ptr<Work> AllReduce(const std::shared_ptr<Tensor> &tensor, ReduceOpType reduce_op, const ProcessGroup *pg,
                                bool async_op) {
    auto device = tensor->GetDevice().type();
    if (pg == nullptr) {
        pg = ProcessGroupFactory::Instance(device)->GetDefaultProcessGroup();
    }
    return pg->AllReduce(tensor, reduce_op, async_op);
}

std::shared_ptr<Work> AllGather(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input,
                                const ProcessGroup *pg, bool async_op) {
    auto device = output->GetDevice().type();
    if (pg == nullptr) {
        pg = ProcessGroupFactory::Instance(device)->GetDefaultProcessGroup();
    }
    return pg->AllGather(output, input, async_op);
}

std::shared_ptr<Work> ReduceScatter(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &input,
                                    ReduceOpType reduce_op, const ProcessGroup *pg, bool async_op) {
    auto device = output->GetDevice().type();
    if (pg == nullptr) {
        pg = ProcessGroupFactory::Instance(device)->GetDefaultProcessGroup();
    }
    return pg->ReduceScatter(output, input, reduce_op, async_op);
}

std::vector<std::vector<std::shared_ptr<Tensor>>> Scatter(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                                          const std::vector<Device> &devices, int dim) {
    std::vector<std::vector<std::shared_ptr<Tensor>>> output_tensors;
    for (const auto &tensor : input_tensors) {
        output_tensors.emplace_back(std::make_shared<autograd::comm::Scatter>(devices, dim)->Apply({tensor}));
    }
    std::vector<std::vector<std::shared_ptr<Tensor>>> transposed_output_tensors;
    transposed_output_tensors.resize(devices.size());
    for (int i = 0; i < devices.size(); ++i) {
        transposed_output_tensors[i].resize(input_tensors.size());
        for (int j = 0; j < input_tensors.size(); ++j) { transposed_output_tensors[i][j] = output_tensors[j][i]; }
    }
    return transposed_output_tensors;
}

std::vector<std::shared_ptr<Tensor>> Gather(const std::vector<std::vector<std::shared_ptr<Tensor>>> &tensors,
                                            Device target_device, int dim) {
    std::vector<std::shared_ptr<Tensor>> gather_tensors;
    for (const auto &tensor : tensors) { gather_tensors.push_back(tensor[0]); }
    return std::make_shared<autograd::comm::Gather>(target_device, dim)->Apply(gather_tensors);
}

std::vector<std::vector<std::shared_ptr<Tensor>>>
BroadcastCoalescedReshape(const std::vector<std::shared_ptr<Tensor>> &tensors, const std::vector<Device> &devices) {
    if (tensors.empty()) {
        return {};
    }
    auto tensor_copies = std::make_shared<autograd::comm::Broadcast>(devices)->Apply(tensors);
    std::vector<std::vector<std::shared_ptr<Tensor>>> tensor_copies_reshaped(devices.size());
    for (int replica_idx = 0; replica_idx < devices.size(); ++replica_idx) {
        tensor_copies_reshaped[replica_idx].resize(tensors.size());
        for (int tensor_idx = 0; tensor_idx < tensors.size(); ++tensor_idx) {
            tensor_copies_reshaped[replica_idx][tensor_idx] = tensor_copies[replica_idx * tensors.size() + tensor_idx];
        }
    }
    return tensor_copies_reshaped;
}

std::vector<std::shared_ptr<Module>> Replicate(const std::shared_ptr<Module> &network,
                                               const std::vector<Device> &devices) {
    const int num_replicas = devices.size();

    // FIXME(dcj): Parameters function need deduplication
    auto params = network->Parameters();
    std::unordered_map<Tensor *, int> param_indices;
    for (int idx = 0; idx < params.size(); ++idx) { param_indices[params[idx].get()] = idx; }
    auto param_copies = BroadcastCoalescedReshape(params, devices);
    for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
        for (auto param : param_copies[replica_idx]) {
            param->RequiresGrad();
            // FIXME(dcj): maybe wrong in dp(need autograd reduce)
            param->set_is_leaf(true);
        }
    }

    auto buffers = network->Buffers();
    std::unordered_map<Tensor *, int> buffer_indices;
    for (int idx = 0; idx < buffers.size(); ++idx) { buffer_indices[buffers[idx].get()] = idx; }
    auto buffer_copies = BroadcastCoalescedReshape(buffers, devices);

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
        for (const auto &name : module->module_order_) {
            const auto &child = module->modules_.at(name);
            for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
                auto &replica = module_copies[replica_idx][idx];
                replica->RegisterModule(name,
                                        child ? module_copies[replica_idx][module_indices.at(child.get())] : nullptr);
            }
        }
        for (const auto &name : module->parameter_order_) {
            const auto &param = module->parameters_.at(name);
            for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
                auto &replica = module_copies[replica_idx][idx];
                replica->RegisterParameter(name,
                                           param ? param_copies[replica_idx][param_indices.at(param.get())] : nullptr);
            }
        }
        for (const auto &name : module->buffer_order_) {
            const auto &buffer = module->buffers_.at(name);
            const bool persistent = !module->non_persistent_buffers_.contains(name);
            for (int replica_idx = 0; replica_idx < num_replicas; ++replica_idx) {
                auto &replica = module_copies[replica_idx][idx];
                replica->RegisterBuffer(
                    name, buffer ? buffer_copies[replica_idx][buffer_indices.at(buffer.get())] : nullptr, persistent);
            }
        }
    }

    std::vector<std::shared_ptr<Module>> replicas;
    for (int idx = 0; idx < num_replicas; ++idx) { replicas.push_back(std::move(module_copies[idx][0])); }
    return replicas;
}
} // namespace infini_train::nn::parallel::function
