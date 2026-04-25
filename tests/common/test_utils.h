#pragma once

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

#if defined(USE_CUDA)
#if defined(__has_include)
#if __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#else
#error "CUDA runtime headers are required when USE_CUDA=ON"
#endif
#else
#include <cuda_runtime_api.h>
#endif
#endif

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
namespace test {

#ifdef USE_CUDA
inline int GetCudaDeviceCount() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return 0;
    }
    return std::max(count, 0);
}
#else
inline int GetCudaDeviceCount() { return 0; }
#endif

inline bool HasCudaRuntime() { return GetCudaDeviceCount() > 0; }

inline void FillSequentialTensor(const std::shared_ptr<Tensor> &tensor, float start = 0.0f) {
    size_t size = 1;
    for (auto dim : tensor->Dims()) { size *= static_cast<size_t>(dim); }

    if (!tensor->GetDevice().IsCPU()) {
        auto cpu_tensor
            = std::make_shared<Tensor>(tensor->Dims(), tensor->Dtype(), Device(Device::DeviceType::kCPU, 0));
        auto *cpu_data = static_cast<float *>(cpu_tensor->DataPtr());
        for (size_t i = 0; i < size; ++i) { cpu_data[i] = start + static_cast<float>(i); }
        tensor->CopyFrom(cpu_tensor);
        return;
    }

    auto *data = static_cast<float *>(tensor->DataPtr());
    for (size_t i = 0; i < size; ++i) { data[i] = start + static_cast<float>(i); }
}

#define REQUIRE_MIN_DEVICES(n)                                                                                         \
    do {                                                                                                               \
        int available_gpus = infini_train::test::GetCudaDeviceCount();                                                 \
        if (available_gpus < (n)) {                                                                                    \
            GTEST_SKIP() << "requires at least " << (n) << " GPUs (found " << available_gpus << ")";                   \
        }                                                                                                              \
    } while (0)

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

class InfiniTrainTest : public ::testing::TestWithParam<Device::DeviceType> {
protected:
    static void SetUpTestSuite() { nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1); }
    Device GetDevice() const { return Device(GetParam(), 0); }
    std::shared_ptr<Tensor> createTensor(const std::vector<int64_t> &shape, DataType dtype = DataType::kFLOAT32,
                                         bool requires_grad = false) {
        auto t = std::make_shared<Tensor>(shape, dtype, GetDevice());
        t->set_requires_grad(requires_grad);
        return t;
    }
};

class AutogradTestBase : public InfiniTrainTest {
protected:
    std::shared_ptr<Tensor> createTensor(const std::vector<int64_t> &shape, float value = 0.0f) {
        auto t = std::make_shared<Tensor>(shape, DataType::kFLOAT32, GetDevice());
        t->set_requires_grad(true);
        FillSequentialTensor(t, value);
        return t;
    }
};

inline std::vector<Device::DeviceType> CudaDeviceTypes() {
    if (HasCudaRuntime()) {
        return {Device::DeviceType::kCUDA};
    }
    LOG(INFO) << "No CUDA runtime found, skipping CUDA tests.";
    return {};
}

} // namespace test
} // namespace infini_train

#define INFINI_TRAIN_REGISTER_TEST(TestName)                                                                           \
    INSTANTIATE_TEST_SUITE_P(CPU, TestName, ::testing::Values(infini_train::Device::DeviceType::kCPU));                \
    INSTANTIATE_TEST_SUITE_P(CUDA, TestName, ::testing::ValuesIn(infini_train::test::CudaDeviceTypes()))
