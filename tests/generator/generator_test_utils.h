#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"
#include "tests/common/test_utils.h"

namespace infini_train::test {

class GeneratorDeviceTest : public InfiniTrainTest {};

inline std::shared_ptr<Generator> MakeGenerator(Device device, uint64_t seed) {
    return device.IsCPU() ? MakeCPUGenerator(seed) : MakeCUDAGenerator(device.index(), seed);
}

inline std::vector<float> CopyToCPUData(const std::shared_ptr<Tensor> &tensor) {
    auto cpu_tensor = tensor->To(Device());
    auto *impl = core::GetDeviceGuardImpl(tensor->GetDevice().type());
    impl->SynchronizeDevice(tensor->GetDevice());
    if (cpu_tensor.NumElements() == 0) {
        return {};
    }
    const auto *data = static_cast<const float *>(cpu_tensor.DataPtr());
    return std::vector<float>(data, data + cpu_tensor.NumElements());
}

} // namespace infini_train::test
