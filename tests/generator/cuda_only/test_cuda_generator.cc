#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/generator.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cuda/cuda_generator_impl.h"
#include "tests/common/test_utils.h"

namespace infini_train::test {
namespace {

constexpr uint64_t kSeed = 0x12345678ULL;

std::vector<uint8_t> StateBytes(const Generator &generator) {
    auto state = generator.get_state();
    const auto *data = static_cast<const uint8_t *>(state->DataPtr());
    return std::vector<uint8_t>(data, data + state->SizeInBytes());
}

TEST(CudaGeneratorTest, StateIsAnOpaqueUint8CpuTensor) {
    Generator generator = core::cuda::createCUDAGenerator(0, kSeed);
    auto state = generator.get_state();

    ASSERT_TRUE(state);
    EXPECT_TRUE(state->GetDevice().IsCPU());
    EXPECT_EQ(state->Dtype(), DataType::kUINT8);
    EXPECT_EQ(state->SizeInBytes(), sizeof(uint64_t) * 2);
}

TEST(CudaGeneratorTest, MissingGeneratorUsesMatchingDeviceDefaultGenerator) {
    REQUIRE_MIN_DEVICES(2);
    manual_seed(kSeed);
    const Generator &device_zero = core::cuda::getDefaultCUDAGenerator(0);
    const Generator &device_one = core::cuda::getDefaultCUDAGenerator(1);

    EXPECT_NE(device_zero, device_one);
    EXPECT_EQ(device_zero.device().index(), 0);
    EXPECT_EQ(device_one.device().index(), 1);

    const std::vector<uint8_t> device_zero_before = StateBytes(device_zero);
    const std::vector<uint8_t> device_one_before = StateBytes(device_one);
    auto tensor = std::make_shared<Tensor>(
        std::vector<int64_t>{1024}, DataType::kFLOAT32,
        Device(Device::DeviceType::kCUDA, 1));
    nn::init::Uniform(tensor);

    EXPECT_EQ(device_zero_before, StateBytes(device_zero));
    EXPECT_NE(device_one_before, StateBytes(device_one));
}

} // namespace
} // namespace infini_train::test
