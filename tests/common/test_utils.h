#pragma once

#include <memory>
#include <vector>

#if defined(USE_CUDA)
#include <cuda_runtime_api.h>
#endif

#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"
#include "gtest/gtest.h"

namespace infini_train {
namespace test {

// Compares flattened FP32 values with EXPECT_FLOAT_EQ (up to 4 ULPs, not bitwise equality).
// Use when the operation should preserve or deterministically reproduce FP32 values,
// such as fill/copy and data movement or simple arithmetic with the same rounding path.
// This overload checks the element count and flat order, but not the tensor shape.
inline void ExpectTensorFloatEqual(const std::shared_ptr<Tensor> &val1, const std::vector<float> &val2) {
    ASSERT_NE(val1, nullptr);
    ASSERT_EQ(val1->Dtype(), DataType::kFLOAT32);

    auto val1_cpu = val1->To(Device());
    ASSERT_EQ(val1_cpu.NumElements(), val2.size());
    const float *val1_data = static_cast<const float *>(val1_cpu.DataPtr());
    for (size_t i = 0; i < val2.size(); ++i) {
        EXPECT_FLOAT_EQ(val1_data[i], val2[i]) << "Mismatch at flat index " << i;
    }
}

inline void ExpectTensorFloatEqual(const std::shared_ptr<Tensor> &val1, float val2) {
    ASSERT_NE(val1, nullptr);
    ASSERT_EQ(val1->Dtype(), DataType::kFLOAT32);

    auto val1_cpu = val1->To(Device());
    const float *val1_data = static_cast<const float *>(val1_cpu.DataPtr());
    for (size_t i = 0; i < val1_cpu.NumElements(); ++i) {
        EXPECT_FLOAT_EQ(val1_data[i], val2) << "Mismatch at flat index " << i;
    }
}

// Compares flattened FP32 values using the caller-provided absolute error bound.
// Use for numerically computed results whose rounding can vary with operation order or backend,
// such as reductions, normalization, softmax, and loss calculations. abs_error is an absolute, not relative, bound.
// This overload checks the element count and flat order, but not the tensor shape.
inline void ExpectTensorNear(const std::shared_ptr<Tensor> &val1, const std::vector<float> &val2, float abs_error) {
    ASSERT_NE(val1, nullptr);
    ASSERT_EQ(val1->Dtype(), DataType::kFLOAT32);
    ASSERT_GE(abs_error, 0.0f);

    auto val1_cpu = val1->To(Device());
    ASSERT_EQ(val1_cpu.NumElements(), val2.size());
    const float *val1_data = static_cast<const float *>(val1_cpu.DataPtr());
    for (size_t i = 0; i < val2.size(); ++i) {
        EXPECT_NEAR(val1_data[i], val2[i], abs_error) << "Mismatch at flat index " << i;
    }
}

inline void ExpectTensorNear(const std::shared_ptr<Tensor> &val1, float val2, float abs_error) {
    ASSERT_NE(val1, nullptr);
    ASSERT_EQ(val1->Dtype(), DataType::kFLOAT32);
    ASSERT_GE(abs_error, 0.0f);

    auto val1_cpu = val1->To(Device());
    const float *val1_data = static_cast<const float *>(val1_cpu.DataPtr());
    for (size_t i = 0; i < val1_cpu.NumElements(); ++i) {
        EXPECT_NEAR(val1_data[i], val2, abs_error) << "Mismatch at flat index " << i;
    }
}

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
