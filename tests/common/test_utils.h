#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#if defined(USE_CUDA)
#include <cuda_runtime_api.h>
#endif

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"
#include "gtest/gtest.h"

namespace infini_train {
namespace test {

#if defined(USE_CUDA)
#define REQUIRE_MIN_DEVICES(n)                                                                                         \
    do {                                                                                                               \
        int available_gpus = 0;                                                                                        \
        cudaGetDeviceCount(&available_gpus);                                                                           \
        if (available_gpus < (n)) {                                                                                    \
            GTEST_SKIP() << "requires at least " << (n) << " GPUs (found " << available_gpus << ")";                   \
        }                                                                                                              \
    } while (0)
#else
#define REQUIRE_MIN_DEVICES(n)                                                                                         \
    do { GTEST_SKIP() << "requires at least " << (n) << " GPUs (CUDA disabled)"; } while (0)
#endif

#define SKIP_CPU()                                                                                                     \
    do {                                                                                                               \
        if (GetParam() == infini_train::Device::DeviceType::kCPU) {                                                    \
            GTEST_SKIP() << "skipped on CPU";                                                                          \
        }                                                                                                              \
    } while (0)

#define ONLY_CPU()                                                                                                     \
    do {                                                                                                               \
        if (GetParam() != infini_train::Device::DeviceType::kCPU) {                                                    \
            GTEST_SKIP() << "CPU-only test";                                                                           \
        }                                                                                                              \
    } while (0)

#define ONLY_CUDA()                                                                                                    \
    do {                                                                                                               \
        if (GetParam() != infini_train::Device::DeviceType::kCUDA) {                                                   \
            GTEST_SKIP() << "CUDA-only test";                                                                          \
        }                                                                                                              \
    } while (0)

// Returns tensor bytes on the host for exact comparisons.
// Synchronizes the source device before reading the copied data.
inline std::vector<std::uint8_t> TensorBytes(const std::shared_ptr<Tensor> &tensor) {
    const Tensor host = tensor->To(Device());
    const Device source_device = tensor->GetDevice();
    core::GetDeviceGuardImpl(source_device.type())->SynchronizeDevice(source_device);
    const auto *data = static_cast<const std::uint8_t *>(host.DataPtr());
    return {data, data + host.SizeInBytes()};
}

class InfiniTrainTest : public ::testing::TestWithParam<Device::DeviceType> {
protected:
    Device GetDevice() const { return Device(GetParam(), 0); }
};

} // namespace test
} // namespace infini_train

#if defined(USE_CUDA)
#define INFINI_TRAIN_REGISTER_TEST(TestName)                                                                           \
    INSTANTIATE_TEST_SUITE_P(CPU, TestName, ::testing::Values(infini_train::Device::DeviceType::kCPU));                \
    INSTANTIATE_TEST_SUITE_P(CUDA, TestName, ::testing::Values(infini_train::Device::DeviceType::kCUDA))
#else
#define INFINI_TRAIN_REGISTER_TEST(TestName)                                                                           \
    INSTANTIATE_TEST_SUITE_P(CPU, TestName, ::testing::Values(infini_train::Device::DeviceType::kCPU))
#endif
