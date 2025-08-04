#include "infini_train/include/nn/parallel/distributed_data_parallel.h"

#include <format>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/nn/parallel/scatter_gather.h"

namespace infini_train::nn::parallel {
namespace {
constexpr char kModuleName[] = "module";

void AllReduce(const std::vector<std::vector<std::shared_ptr<Tensor>>> &grads) {
    CHECK_GT(grads.size(), 0);
    auto device = grads[0][0]->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "CommNcclAllReduce"});
    // FIXME(dcj): should auto deduce the grads's type
    kernel.Call<void, std::vector<std::vector<std::shared_ptr<Tensor>>>>(grads);
}

void AllReduceGradients(const std::vector<std::shared_ptr<Module>> &replicas,
                        const std::vector<const Device *> &devices) {
    CHECK_GT(replicas.size(), 0);

    const auto &params = replicas[0]->Parameters();
    std::vector<std::vector<std::shared_ptr<Tensor>>> grads(replicas.size());

    for (int replica_idx = 0; replica_idx < replicas.size(); ++replica_idx) {
        grads[replica_idx].resize(params.size());
        for (int param_idx = 0; param_idx < params.size(); ++param_idx) {
            grads[replica_idx][param_idx] = replicas[replica_idx]->Parameters()[param_idx]->grad();
        }
    }

    AllReduce(grads);
}

std::vector<std::vector<std::shared_ptr<Tensor>>>
ParallelApply(const std::vector<std::shared_ptr<Module>> &modules,
              const std::vector<std::vector<std::shared_ptr<Tensor>>> &inputs,
              const std::vector<const Device *> &devices) {
    CHECK_EQ(modules.size(), inputs.size()) << std::format(
        "The number of modules {} is not equal to the number of inputs {}", modules.size(), inputs.size());
    CHECK_EQ(modules.size(), devices.size());

    // pre-allocate results so we do not need lock in the worker threads
    std::vector<std::optional<std::vector<std::shared_ptr<Tensor>>>> results(modules.size(), std::nullopt);

    auto worker = [&](const std::shared_ptr<Module> &module, const std::vector<std::shared_ptr<Tensor>> &inputs,
                      const Device *device, int idx) {
        device->SetDevice();
        auto output = module->Forward(inputs);
        results[idx] = output;
    };

    if (modules.size() > 1) {
        std::vector<std::thread> threads;
        for (int idx = 0; idx < modules.size(); ++idx) {
            threads.emplace_back(worker, modules[idx], inputs[idx], devices[idx], idx);
        }
        for (int idx = 0; idx < threads.size(); ++idx) { threads[idx].join(); }
    } else {
        worker(modules[0], inputs[0], devices[0], 0);
    }

    std::vector<std::vector<std::shared_ptr<Tensor>>> outputs;
    for (int idx = 0; idx < inputs.size(); ++idx) {
        auto output = results[idx];
        if (!output) {
            LOG(FATAL) << "ParallelApply worker thread " << idx << " did not return a result";
        }
        outputs.push_back(output.value());
    }
    return outputs;
}
} // namespace

ThreadDDP::ThreadDDP(const std::shared_ptr<Module> &module, int dim)
    : dim_(dim), devices_(DeviceManager::Instance()->GetAllAvailableDevices(DeviceType::kCUDA)) {
    CHECK_GT(devices_.size(), 0) << "No available devices found";
    output_device_ = devices_.at(0);
    src_device_ = devices_.at(0);

    module->To(src_device_);

    modules_[kModuleName] = std::move(module);

    // FIXME(dcj): no auto grad
    replicas_ = Replicate(modules_[kModuleName], devices_);

    LOG(ERROR) << "ThreadDDP module created with " << devices_.size() << " devices";
}

std::vector<std::shared_ptr<Tensor>> ThreadDDP::Parameters() const {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto &replica : replicas_) {
        for (auto &param : replica->Parameters()) { params.push_back(param); }
    }
    return params;
}

float ThreadDDP::TrainStep(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                           const std::vector<std::shared_ptr<Tensor>> &targets, const std::shared_ptr<Module> &loss_fn,
                           Optimizer &optimizer) {
    auto &module = modules_.at(kModuleName);
    for (auto tensor : module->Parameters()) { CHECK_EQ(tensor->GetDevice(), src_device_); }

    auto scattered_inputs = Scatter(input_tensors, devices_, dim_);
    auto scattered_targets = Scatter(targets, devices_, dim_);

    optimizer.ZeroGrad();

    std::vector<std::vector<std::shared_ptr<Tensor>>> outputs;
    if (devices_.size() == 1) {
        outputs = {module->Forward(scattered_inputs[0])};
    } else {
        outputs = ParallelApply(replicas_, scattered_inputs, devices_);
    }

    float loss_cpu = 0.0f;
    for (int i = 0; i < replicas_.size(); ++i) {
        auto loss = loss_fn->Forward({outputs[i][0], scattered_targets[i][0]})[0] / replicas_.size();
        loss->Backward();
        auto lossf = static_cast<const float *>(loss->To(DeviceManager::Instance()->GetDefaultDevice()).DataPtr())[0];
        loss_cpu += lossf;
    }

    AllReduceGradients(replicas_, devices_);

    optimizer.Step();

    return loss_cpu;
}
} // namespace infini_train::nn::parallel
