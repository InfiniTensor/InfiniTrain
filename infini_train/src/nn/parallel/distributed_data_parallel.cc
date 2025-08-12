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

float ConvertBF16ToFloat(void *ptr) {
    uint16_t *raw_data = reinterpret_cast<uint16_t *>(ptr);
    uint32_t f32_bits = static_cast<uint32_t>(raw_data[0]) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(f));
    return f;
}

float ParallelApply(const std::vector<std::shared_ptr<Module>> &modules,
                    const std::vector<std::vector<std::shared_ptr<Tensor>>> &inputs,
                    const std::vector<std::vector<std::shared_ptr<Tensor>>> &targets,
                    const std::vector<const Device *> &devices, const std::shared_ptr<Module> &loss_fn) {
    CHECK_EQ(modules.size(), inputs.size()) << std::format(
        "The number of modules {} is not equal to the number of inputs {}", modules.size(), inputs.size());
    CHECK_EQ(modules.size(), devices.size());

    // pre-allocate results so we do not need lock in the worker threads
    std::vector<std::optional<std::vector<std::shared_ptr<Tensor>>>> results(modules.size(), std::nullopt);
    std::vector<float> loss_vec(modules.size(), 0.0f);

    auto worker = [&](const std::shared_ptr<Module> &module, const std::vector<std::shared_ptr<Tensor>> &inputs,
                      const std::vector<std::shared_ptr<Tensor>> &targets, const Device *device, int idx, int dp_size) {
        device->SetDevice();
        auto output = module->Forward(inputs);
        results[idx] = output;
        // may conflict with gradient accumulation
        auto loss = loss_fn->Forward({output[0], targets[0]})[0] / dp_size;
        loss->Backward();
        auto loss_cpu = loss->To(DeviceManager::Instance()->GetDefaultDevice());
        if (loss->Dtype() == DataType::kFLOAT32) {
            loss_vec[idx] = static_cast<const float *>(loss_cpu.DataPtr())[0];
        } else if (loss->Dtype() == DataType::kBFLOAT16) {
            loss_vec[idx] = ConvertBF16ToFloat(loss_cpu.DataPtr());
        } else {
            LOG(FATAL) << "Unsupported loss data type " << static_cast<int>(loss->Dtype());
        }
    };

    if (modules.size() > 1) {
        std::vector<std::thread> threads;
        int dp_size = devices.size();
        for (int idx = 0; idx < modules.size(); ++idx) {
            threads.emplace_back(worker, modules[idx], inputs[idx], targets[idx], devices[idx], idx, dp_size);
        }
        for (int idx = 0; idx < threads.size(); ++idx) { threads[idx].join(); }
    } else {
        worker(modules[0], inputs[0], targets[0], devices[0], 0, devices.size());
    }

    float lossf = 0.0f;
    for (int idx = 0; idx < inputs.size(); ++idx) {
        auto output = results[idx];
        if (!output) {
            LOG(FATAL) << "ParallelApply worker thread " << idx << " did not return a result";
        }
        lossf += loss_vec[idx];
    }

    return lossf;
}
} // namespace

void AllReduceGradients(const std::vector<std::shared_ptr<Module>> &replicas) {
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

ThreadDistributedDataParallel::ThreadDistributedDataParallel(const std::shared_ptr<Module> &module, int dim)
    : dim_(dim), devices_(DeviceManager::Instance()->GetAllAvailableDevices(DeviceType::kCUDA)) {
    CHECK_GT(devices_.size(), 0) << "No available devices found";
    output_device_ = devices_.at(0);
    src_device_ = devices_.at(0);

    module->To(src_device_);

    modules_[kModuleName] = std::move(module);

    // FIXME(dcj): no auto grad
    replicas_ = Replicate(modules_[kModuleName], devices_);

    LOG(ERROR) << "ThreadDistributedDataParallel module created with " << devices_.size() << " devices";
}

std::vector<std::shared_ptr<Tensor>> ThreadDistributedDataParallel::Parameters() const {
    std::vector<std::shared_ptr<Tensor>> params;
    for (auto &replica : replicas_) {
        for (auto &param : replica->Parameters()) { params.push_back(param); }
    }
    return params;
}

std::vector<std::shared_ptr<Tensor>> ThreadDistributedDataParallel::Buffers() const {
    std::vector<std::shared_ptr<Tensor>> buffers;
    for (auto &replica : replicas_) {
        for (auto &buffer : replica->Buffers()) { buffers.push_back(buffer); }
    }
    return buffers;
}

void ThreadDistributedDataParallel::To(const Device *device) {
    LOG(FATAL) << "Calling the to device function of ThreadDistributedDataParallel is not allowed!";
}

void ThreadDistributedDataParallel::To(DataType dtype) {
    for (auto &replica : replicas_) { replica->To(dtype); }
}

float ThreadDistributedDataParallel::TrainStep(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                               const std::vector<std::shared_ptr<Tensor>> &targets,
                                               const std::shared_ptr<Module> &loss_fn) {
    auto &module = modules_.at(kModuleName);
    for (auto tensor : module->Parameters()) { CHECK_EQ(tensor->GetDevice(), src_device_); }

    auto scattered_inputs = Scatter(input_tensors, devices_, dim_);
    auto scattered_targets = Scatter(targets, devices_, dim_);

    float lossf = ParallelApply(replicas_, scattered_inputs, scattered_targets, devices_, loss_fn);

    AllReduceGradients(replicas_);

    return lossf;
}
} // namespace infini_train::nn::parallel
