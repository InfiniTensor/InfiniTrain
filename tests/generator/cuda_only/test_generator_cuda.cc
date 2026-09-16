#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_runtime_api.h>

#include "gtest/gtest.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;
using infini_train::test::TensorBytes;

class GeneratorCudaTest : public ::testing::Test {
protected:
    const Device cuda0_{Device::DeviceType::kCUDA, 0};
};

TEST_F(GeneratorCudaTest, StateRejectsCudaAndForeignInput) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    auto generator = CreateGenerator(cuda0_, 4601);
    const auto valid_state = generator.get_state();

    auto wrong_device = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kUINT8, cuda0_);
    EXPECT_DEATH(generator.set_state(*wrong_device), "CPU");

    const auto cpu_state = CreateGenerator(Device(), 4601).get_state();
    EXPECT_DEATH(generator.set_state(*cpu_state), "Check failed");

    auto cpu_generator = CreateGenerator(Device(), 4601);
    EXPECT_DEATH(cpu_generator.set_state(*valid_state), "Check failed");
}

TEST_F(GeneratorCudaTest, CrossBackendGeneratorsAreRejected) {
    const Device cpu;
    const auto cpu_generator = CreateGenerator(cpu, 8001);
    const auto cuda_generator = CreateGenerator(cuda0_, 8002);

    auto cpu_input = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, cpu);
    auto cuda_input = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, cuda0_);
    cpu_input->Fill(1.0f);
    cuda_input->Fill(1.0f);

    EXPECT_THROW(nn::function::Rand({8}, DataType::kFLOAT32, cpu, cuda_generator), std::invalid_argument);
    EXPECT_THROW(nn::function::Rand({8}, DataType::kFLOAT32, cuda0_, cpu_generator), std::invalid_argument);
    EXPECT_THROW(nn::function::Dropout(cpu_input, 0.25, true, cuda_generator), std::invalid_argument);
    EXPECT_THROW(nn::function::Dropout(cuda_input, 0.25, true, cpu_generator), std::invalid_argument);
}

// Regression for the device-index check: a generator bound to cuda:0 must not
// drive a cuda:1 tensor, and the rejected calls must not advance its state.
TEST_F(GeneratorCudaTest, CUDAGeneratorRejectsAnotherCUDADevice) {
    REQUIRE_MIN_DEVICES(2);
    const Device cuda1(Device::DeviceType::kCUDA, 1);
    auto generator = CreateGenerator(cuda0_, 8401);
    const auto state_before = TensorBytes(generator.get_state());

    auto input = std::make_shared<Tensor>(std::vector<int64_t>{257}, DataType::kFLOAT32, cuda1);
    input->Fill(1.0f);

    EXPECT_THROW(nn::function::Rand({257}, DataType::kFLOAT32, cuda1, generator), std::invalid_argument);
    EXPECT_THROW(nn::function::Randn({257}, DataType::kFLOAT32, cuda1, generator), std::invalid_argument);
    EXPECT_THROW(nn::function::Dropout(input, 0.25, true, generator), std::invalid_argument);

    EXPECT_EQ(state_before, TensorBytes(generator.get_state()));
}

TEST_F(GeneratorCudaTest, DefaultCUDAGeneratorsAreIndependentAcrossAllDevices) {
    REQUIRE_MIN_DEVICES(2);
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);

    ManualSeed(8501);
    std::vector<Device> devices;
    devices.reserve(device_count);
    for (int index = 0; index < device_count; ++index) {
        const Device device(Device::DeviceType::kCUDA, index);
        devices.push_back(device);
        EXPECT_EQ(GetDefaultGenerator(device).device(), device);
        EXPECT_EQ(GetDefaultGenerator(device).current_seed(), 8501U);
    }

    for (int active_index = 0; active_index < device_count; ++active_index) {
        std::vector<std::vector<uint8_t>> states_before;
        states_before.reserve(device_count);
        for (const auto &device : devices) {
            states_before.push_back(TensorBytes(GetDefaultGenerator(device).get_state()));
        }

        auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{64}, DataType::kFLOAT32, devices[active_index]);
        nn::init::Uniform(tensor);

        for (int observed_index = 0; observed_index < device_count; ++observed_index) {
            const auto state_after = TensorBytes(GetDefaultGenerator(devices[observed_index]).get_state());
            if (observed_index == active_index) {
                EXPECT_NE(states_before[observed_index], state_after);
            } else {
                EXPECT_EQ(states_before[observed_index], state_after);
            }
        }
    }
}
