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

inline bool HasNCCL() {
#ifdef USE_NCCL
    return true;
#else
    return false;
#endif
}

inline bool HasDistributedSupport() { return HasCudaRuntime() && HasNCCL() && GetCudaDeviceCount() >= 2; }

inline void FillSequentialTensor(const std::shared_ptr<Tensor> &tensor, float start = 0.0f) {
    size_t size = 1;
    for (auto dim : tensor->Dims()) { size *= static_cast<size_t>(dim); }

    if (tensor->GetDevice().IsCUDA()) {
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

inline void FillConstantTensor(const std::shared_ptr<Tensor> &tensor, float value) {
    size_t size = 1;
    for (auto dim : tensor->Dims()) { size *= static_cast<size_t>(dim); }

    if (tensor->GetDevice().IsCUDA()) {
        auto cpu_tensor
            = std::make_shared<Tensor>(tensor->Dims(), tensor->Dtype(), Device(Device::DeviceType::kCPU, 0));
        auto *cpu_data = static_cast<float *>(cpu_tensor->DataPtr());
        for (size_t i = 0; i < size; ++i) { cpu_data[i] = value; }
        tensor->CopyFrom(cpu_tensor);
        return;
    }

    auto *data = static_cast<float *>(tensor->DataPtr());
    for (size_t i = 0; i < size; ++i) { data[i] = value; }
}

#define REQUIRE_CUDA()                                                                                                 \
    do {                                                                                                               \
        if (!infini_train::test::HasCudaRuntime()) {                                                                   \
            GTEST_SKIP() << "requires CUDA support (found " << infini_train::test::GetCudaDeviceCount() << " GPUs)";   \
        }                                                                                                              \
    } while (0)

#define REQUIRE_MIN_GPUS(n)                                                                                            \
    do {                                                                                                               \
        int available_gpus = infini_train::test::GetCudaDeviceCount();                                                 \
        if (available_gpus < (n)) {                                                                                    \
            GTEST_SKIP() << "requires at least " << (n) << " GPUs (found " << available_gpus << ")";                   \
        }                                                                                                              \
    } while (0)

#define REQUIRE_NCCL()                                                                                                 \
    do {                                                                                                               \
        if (!infini_train::test::HasNCCL()) {                                                                          \
            GTEST_SKIP() << "NCCL support is disabled (build with USE_NCCL=ON)";                                       \
        }                                                                                                              \
    } while (0)

#define REQUIRE_DISTRIBUTED()                                                                                          \
    do {                                                                                                               \
        REQUIRE_NCCL();                                                                                                \
        REQUIRE_MIN_GPUS(2);                                                                                           \
    } while (0)

class InfiniTrainTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1); }
};

// ---------------------------------------------------------------------------
// Parametrized base classes (TEST_P).
// Use prefix CPU/ or CUDA/ in INSTANTIATE_TEST_SUITE_P so that CMake
// TEST_FILTER can route each instantiation to the right label.
// ---------------------------------------------------------------------------

class InfiniTrainTestP : public ::testing::TestWithParam<Device::DeviceType> {
protected:
    static void SetUpTestSuite() { nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1); }
    void SetUp() override {
        if (GetParam() == Device::DeviceType::kCUDA && !HasCudaRuntime()) {
            GTEST_SKIP() << "requires CUDA (" << GetCudaDeviceCount() << " GPUs found)";
        }
    }
    Device GetDevice() const { return Device(GetParam(), 0); }
};

class TensorTestBaseP : public InfiniTrainTestP {
protected:
    std::shared_ptr<Tensor> createTensor(const std::vector<int64_t> &shape = {2, 3, 4},
                                         DataType dtype = DataType::kFLOAT32, bool requires_grad = false) {
        auto t = std::make_shared<Tensor>(shape, dtype, GetDevice());
        t->set_requires_grad(requires_grad);
        return t;
    }
};

class AutogradTestBaseP : public InfiniTrainTestP {
protected:
    std::shared_ptr<Tensor> createTensor(const std::vector<int64_t> &shape, float value = 0.0f) {
        auto t = std::make_shared<Tensor>(shape, DataType::kFLOAT32, GetDevice());
        t->set_requires_grad(true);
        FillSequentialTensor(t, value);
        return t;
    }
};

// Distributed variant: requires NCCL + >=2 GPUs.
class DistributedInfiniTrainTestP : public ::testing::TestWithParam<Device::DeviceType> {
protected:
    static void SetUpTestSuite() { nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1); }
    void SetUp() override {
        if (!HasNCCL()) {
            GTEST_SKIP() << "NCCL support is disabled (build with USE_NCCL=ON)";
        }
        if (GetCudaDeviceCount() < 2) {
            GTEST_SKIP() << "requires at least 2 GPUs (" << GetCudaDeviceCount() << " found)";
        }
    }
    Device GetDevice() const { return Device(GetParam(), 0); }
};

class TensorTestBase : public InfiniTrainTest {
protected:
    std::vector<int64_t> default_shape_{2, 3, 4};
    DataType default_dtype_{DataType::kFLOAT32};

    std::shared_ptr<Tensor> createTensor(const std::vector<int64_t> &shape = {2, 3, 4},
                                         DataType dtype = DataType::kFLOAT32, bool requires_grad = false,
                                         Device::DeviceType device = Device::DeviceType::kCPU, int device_id = 0) {
        auto tensor = std::make_shared<Tensor>(shape, dtype, Device(device, device_id));
        tensor->set_requires_grad(requires_grad);
        return tensor;
    }

    void fillTensor(std::shared_ptr<Tensor> tensor, float value) { FillSequentialTensor(tensor, value); }
};

class CPUTensorTest : public TensorTestBase {};

#ifdef USE_CUDA
class CUDATensorTest : public TensorTestBase {
protected:
    CUDATensorTest() {
        default_shape_ = {2, 3, 4};
        default_dtype_ = DataType::kFLOAT32;
    }
};
#endif

#ifdef USE_NCCL
class DistributedTensorTest : public TensorTestBase {};
#endif

class AutogradTestBase : public InfiniTrainTest {
protected:
    std::shared_ptr<Tensor> createTensor(const std::vector<int64_t> &shape, float value = 0.0f,
                                         Device::DeviceType device = Device::DeviceType::kCPU, int device_id = 0) {
        auto tensor = std::make_shared<Tensor>(shape, DataType::kFLOAT32, Device(device, device_id));
        tensor->set_requires_grad(true);
        FillSequentialTensor(tensor, value);
        return tensor;
    }
};

class CPUAutogradTest : public AutogradTestBase {};

#ifdef USE_CUDA
class CUDAAutogradTest : public AutogradTestBase {};
#endif

#ifdef USE_NCCL
class DistributedAutogradTest : public AutogradTestBase {};
#endif

} // namespace test
} // namespace infini_train

// ---------------------------------------------------------------------------
// Convenience macros for instantiating parametrized test suites.
// These expand INSTANTIATE_TEST_SUITE_P with the fixed CPU/CUDA/Distributed
// prefixes that CMake TEST_FILTER patterns rely on.
// ---------------------------------------------------------------------------

// Instantiate a test suite on both CPU and CUDA.
#define INFINI_TRAIN_REGISTER_TEST(TestName)                                                                           \
    INSTANTIATE_TEST_SUITE_P(CPU, TestName, ::testing::Values(infini_train::Device::DeviceType::kCPU));                \
    INSTANTIATE_TEST_SUITE_P(CUDA, TestName, ::testing::Values(infini_train::Device::DeviceType::kCUDA))

// Instantiate a test suite on CPU only.
#define INFINI_TRAIN_REGISTER_TEST_CPU(TestName)                                                                       \
    INSTANTIATE_TEST_SUITE_P(CPU, TestName, ::testing::Values(infini_train::Device::DeviceType::kCPU))

// Instantiate a distributed test suite (CUDA device,
// DistributedInfiniTrainTestP provides the NCCL + GPU-count guard).
#define INFINI_TRAIN_REGISTER_TEST_DISTRIBUTED(TestName)                                                               \
    INSTANTIATE_TEST_SUITE_P(Distributed, TestName, ::testing::Values(infini_train::Device::DeviceType::kCUDA))
