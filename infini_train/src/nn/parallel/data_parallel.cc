#include "infini_train/include/nn/parallel/data_parallel.h"

#include <format>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
namespace {
constexpr char kModuleName[] = "module";

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

DataParallel::DataParallel(const std::shared_ptr<Module> &module, int dim)
    : dim_(dim), devices_(DeviceManager::Instance()->GetAllAvailableDevices(DeviceType::kCUDA)) {
    CHECK_GT(devices_.size(), 0) << "No available devices found";
    output_device_ = devices_.at(0);
    src_device_ = devices_.at(0);

    module->To(src_device_);

    modules_[kModuleName] = std::move(module);

    // TODO(dcj): implement check_balance for cuda devices later.

    LOG(ERROR) << "DataParallel module created with " << devices_.size() << " devices";
}

std::vector<std::shared_ptr<Tensor>> DataParallel::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    auto &module = modules_.at(kModuleName);

    for (auto tensor : module->Parameters()) {
        if (tensor->GetDevice() != src_device_) {
            LOG(FATAL) << std::format("module must have its Parameters on device {} (device_ids[0]) but found "
                                      "one of them on device: {}",
                                      src_device_->ToString(), tensor->GetDevice()->ToString());
        }
    }

    auto scattered_inputs = function::Scatter(input_tensors, devices_, dim_);

    if (devices_.size() == 1) {
        return module->Forward(scattered_inputs[0]);
    }

    auto replicas = function::Replicate(module, devices_);

    auto outputs = ParallelApply(replicas, scattered_inputs, devices_);

    return function::Gather(outputs, output_device_, dim_);
}
} // namespace infini_train::nn::parallel
