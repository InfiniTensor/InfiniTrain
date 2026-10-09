#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/core/runtime/device_guard.h"
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

#define REQUIRE_MIN_DEVICES(n)                                                                                         \
    do {                                                                                                               \
        const int required_devices = (n);                                                                              \
        const int available_devices = DeviceCount();                                                                   \
        if (available_devices < required_devices) {                                                                    \
            GTEST_SKIP() << "requires at least " << required_devices << " devices (found " << available_devices        \
                         << ")";                                                                                       \
        }                                                                                                              \
    } while (0)

#define SKIP_CPU()                                                                                                     \
    do {                                                                                                               \
        if (!IsAccelerator()) {                                                                                        \
            GTEST_SKIP() << "skipped on CPU";                                                                          \
        }                                                                                                              \
    } while (0)

#define ONLY_CPU()                                                                                                     \
    do {                                                                                                               \
        if (IsAccelerator()) {                                                                                         \
            GTEST_SKIP() << "CPU-only test";                                                                           \
        }                                                                                                              \
    } while (0)

class InfiniTrainTest : public ::testing::TestWithParam<Device::DeviceType> {
protected:
    Device GetDevice() const {
#if defined(INFINI_TRAIN_TEST_DEVICE_INDEX)
        return Device(GetParam(), INFINI_TRAIN_TEST_DEVICE_INDEX);
#else
        return Device(GetParam(), 0);
#endif
    }

    int DeviceCount() const { return core::GetDeviceGuardImpl(GetDevice().type())->DeviceCount(); }

    bool IsAccelerator() const { return GetDevice().type() != Device::DeviceType::kCPU; }
};

} // namespace test
} // namespace infini_train

#define INFINI_TRAIN_INSTANTIATE_TEST(Prefix, TestName, DeviceType)                                                    \
    INSTANTIATE_TEST_SUITE_P(Prefix, TestName, ::testing::Values(DeviceType))

#if defined(INFINI_TRAIN_TEST_DEVICE_TYPE) != defined(INFINI_TRAIN_TEST_DEVICE_PREFIX)
#error "INFINI_TRAIN_TEST_DEVICE_TYPE and INFINI_TRAIN_TEST_DEVICE_PREFIX must be defined together"
#endif

#define INFINI_TRAIN_REGISTER_TEST(TestName)                                                                           \
    INFINI_TRAIN_INSTANTIATE_TEST(INFINI_TRAIN_TEST_DEVICE_PREFIX, TestName, INFINI_TRAIN_TEST_DEVICE_TYPE)
