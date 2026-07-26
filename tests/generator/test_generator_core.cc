#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {

std::shared_ptr<Tensor> CopyToCPU(const std::shared_ptr<Tensor> &tensor) {
    auto host = std::make_shared<Tensor>(tensor->Dims(), tensor->Dtype(), Device());
    host->CopyFrom(tensor);
    if (tensor->GetDevice().IsCUDA()) {
        core::GetDeviceGuardImpl(tensor->GetDevice().type())->SynchronizeDevice(tensor->GetDevice());
    }
    return host;
}

std::vector<uint8_t> TensorBytes(const std::shared_ptr<Tensor> &tensor) {
    const auto host = CopyToCPU(tensor);
    const auto *data = static_cast<const uint8_t *>(host->DataPtr());
    return {data, data + host->SizeInBytes()};
}

} // namespace

class GeneratorCoreTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorCoreTest, PublicInterfaceCopyCloneAndSeed) {
    const Device device = GetDevice();
    auto generator = CreateGenerator(device);
    EXPECT_TRUE(generator.defined());
    EXPECT_EQ(generator.device(), device);
    EXPECT_EQ(generator.current_seed(), Generator::kDefaultSeed);

    generator.set_current_seed(1234);
    EXPECT_EQ(generator.current_seed(), 1234U);
    const auto first = TensorBytes(nn::function::Rand({128}, DataType::kFLOAT32, device, generator));
    generator.set_current_seed(1234);
    EXPECT_EQ(first, TensorBytes(nn::function::Rand({128}, DataType::kFLOAT32, device, generator)));

    generator.set_current_seed(2345);
    Generator alias = generator;
    EXPECT_EQ(alias, generator);
    const auto state_before = TensorBytes(alias.get_state());
    nn::function::Rand({32}, DataType::kFLOAT32, device, generator);
    EXPECT_NE(state_before, TensorBytes(alias.get_state()));

    auto clone = generator.clone();
    EXPECT_NE(clone, generator);
    EXPECT_EQ(TensorBytes(clone.get_state()), TensorBytes(generator.get_state()));
    clone.set_current_seed(3456);
    EXPECT_EQ(generator.current_seed(), 2345U);
    EXPECT_EQ(clone.current_seed(), 3456U);

    const uint64_t generated_seed = clone.seed();
    EXPECT_EQ(clone.current_seed(), generated_seed);
}

TEST_P(GeneratorCoreTest, HighBitsOfCPUSeedAffectTheSequence) {
    ONLY_CPU();
    const Device device = GetDevice();
    constexpr uint64_t low_seed = 17;
    constexpr uint64_t high_seed = low_seed + (uint64_t{1} << 32);
    EXPECT_NE(TensorBytes(nn::function::Rand({256}, DataType::kFLOAT32, device, CreateGenerator(device, low_seed))),
              TensorBytes(nn::function::Rand({256}, DataType::kFLOAT32, device, CreateGenerator(device, high_seed))));
}

TEST_P(GeneratorCoreTest, StateRestoresUniformAndNormalSequence) {
    const Device device = GetDevice();
    auto generator = CreateGenerator(device, 4567);
    const auto state = generator.get_state();
    EXPECT_EQ(state->Dtype(), DataType::kUINT8);
    EXPECT_TRUE(state->GetDevice().IsCPU());

    const auto uniform = TensorBytes(nn::function::Rand({129}, DataType::kFLOAT32, device, generator));
    generator.set_state(*state);
    EXPECT_EQ(uniform, TensorBytes(nn::function::Rand({129}, DataType::kFLOAT32, device, generator)));

    if (device.IsCPU()) {
        nn::function::Randn({3}, DataType::kFLOAT32, device, generator);
        const auto state_after_odd_normal = generator.get_state();
        const auto normal = TensorBytes(nn::function::Randn({127}, DataType::kFLOAT32, device, generator));
        generator.set_state(*state_after_odd_normal);
        EXPECT_EQ(normal, TensorBytes(nn::function::Randn({127}, DataType::kFLOAT32, device, generator)));
    }
}

TEST_P(GeneratorCoreTest, StateRejectsMalformedOrForeignInput) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const Device device = GetDevice();
    auto generator = CreateGenerator(device, 4601);
    auto wrong_dtype = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kFLOAT32, Device());
    EXPECT_DEATH(generator.set_state(*wrong_dtype), "UINT8");

    const auto valid_state = generator.get_state();
    auto truncated = std::make_shared<Tensor>(
        std::vector<int64_t>{static_cast<int64_t>(valid_state->SizeInBytes() - 1)}, DataType::kUINT8, Device());
    EXPECT_DEATH(generator.set_state(*truncated), "Check failed");

    if (device.IsCUDA()) {
        auto wrong_device = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kUINT8, device);
        EXPECT_DEATH(generator.set_state(*wrong_device), "CPU");
        const auto cpu_state = CreateGenerator(Device(), 4601).get_state();
        EXPECT_DEATH(generator.set_state(*cpu_state), "Check failed");
        auto cpu_generator = CreateGenerator(Device(), 4601);
        EXPECT_DEATH(cpu_generator.set_state(*valid_state), "Check failed");
    }
}

TEST_P(GeneratorCoreTest, DefaultExplicitAndUndefinedPathsHaveExpectedState) {
    const Device device = GetDevice();
    constexpr uint64_t seed = 7890;

    manual_seed(seed);
    const auto *first_default = &GetDefaultGenerator(device);
    const auto *second_default = &GetDefaultGenerator(device);
    EXPECT_EQ(first_default, second_default);
    EXPECT_EQ(first_default->device(), device);

    const auto default_state = TensorBytes(first_default->get_state());
    const auto explicit_result
        = TensorBytes(nn::function::Rand({256}, DataType::kFLOAT32, device, CreateGenerator(device, seed)));
    EXPECT_EQ(default_state, TensorBytes(first_default->get_state()));
    EXPECT_EQ(explicit_result, TensorBytes(nn::function::Rand({256}, DataType::kFLOAT32, device)));

    manual_seed(seed + 1);
    const auto default_result = TensorBytes(nn::function::Rand({256}, DataType::kFLOAT32, device));
    manual_seed(seed + 1);
    Generator undefined;
    EXPECT_EQ(default_result, TensorBytes(nn::function::Rand({256}, DataType::kFLOAT32, device, undefined)));
}

TEST_P(GeneratorCoreTest, CrossBackendGeneratorsAreRejected) {
    ONLY_CPU();
#if defined(USE_CUDA)
    const Device cpu;
    const Device cuda(Device::DeviceType::kCUDA, 0);
    const auto cpu_generator = CreateGenerator(cpu, 8001);
    const auto cuda_generator = CreateGenerator(cuda, 8002);

    auto cpu_input = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, cpu);
    auto cuda_input = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, cuda);
    cpu_input->Fill(1.0f);
    cuda_input->Fill(1.0f);

    EXPECT_THROW(nn::function::Rand({8}, DataType::kFLOAT32, cpu, cuda_generator), std::invalid_argument);
    EXPECT_THROW(nn::function::Rand({8}, DataType::kFLOAT32, cuda, cpu_generator), std::invalid_argument);
    EXPECT_THROW(nn::function::Dropout(cpu_input, 0.25, true, cuda_generator), std::invalid_argument);
    EXPECT_THROW(nn::function::Dropout(cuda_input, 0.25, true, cpu_generator), std::invalid_argument);
#else
    GTEST_SKIP() << "CUDA disabled";
#endif
}

TEST_P(GeneratorCoreTest, CUDAGeneratorCanDriveAnotherCUDADevice) {
    ONLY_CUDA();
#if defined(USE_CUDA)
    REQUIRE_MIN_DEVICES(2);
    const Device cuda0(Device::DeviceType::kCUDA, 0);
    const Device cuda1(Device::DeviceType::kCUDA, 1);
    auto generator = CreateGenerator(cuda0, 8401);
    const auto state_before = TensorBytes(generator.get_state());

    const auto output = nn::function::Rand({257}, DataType::kFLOAT32, cuda1, generator);
    EXPECT_EQ(output->GetDevice(), cuda1);
    EXPECT_NE(state_before, TensorBytes(generator.get_state()));
    EXPECT_EQ(TensorBytes(output),
              TensorBytes(nn::function::Rand({257}, DataType::kFLOAT32, cuda1, CreateGenerator(cuda0, 8401))));
#else
    GTEST_SKIP() << "CUDA disabled";
#endif
}

TEST_P(GeneratorCoreTest, InitializerEntryPointsUseExplicitGenerator) {
    const Device device = GetDevice();
    auto make_tensor = [&] {
        return std::make_shared<Tensor>(std::vector<int64_t>{16, 16}, DataType::kFLOAT32, device);
    };

    auto normal_a = make_tensor();
    auto normal_b = make_tensor();
    nn::init::Normal(normal_a, 1.5f, 0.5f, CreateGenerator(device, 8101));
    nn::init::Normal(normal_b, 1.5f, 0.5f, CreateGenerator(device, 8101));
    EXPECT_EQ(TensorBytes(normal_a), TensorBytes(normal_b));

    auto uniform_a = make_tensor();
    auto uniform_b = make_tensor();
    nn::init::Uniform(uniform_a, -2.0f, 3.0f, CreateGenerator(device, 8102));
    uniform_b->Uniform(-2.0f, 3.0f, CreateGenerator(device, 8102));
    EXPECT_EQ(TensorBytes(uniform_a), TensorBytes(uniform_b));

    auto kaiming_a = make_tensor();
    auto kaiming_b = make_tensor();
    nn::init::KaimingUniform(kaiming_a, 0.1f, nn::init::KaimingMode::kFanIn, nn::init::NonLinearityType::kLeakyReLU,
                             CreateGenerator(device, 8103));
    nn::init::KaimingUniform(kaiming_b, 0.1f, nn::init::KaimingMode::kFanIn, nn::init::NonLinearityType::kLeakyReLU,
                             CreateGenerator(device, 8103));
    EXPECT_EQ(TensorBytes(kaiming_a), TensorBytes(kaiming_b));
}

TEST_P(GeneratorCoreTest, DefaultCUDAGeneratorsAreIndependentAcrossAllDevices) {
    ONLY_CUDA();
#if defined(USE_CUDA)
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count < 2) {
        GTEST_SKIP() << "requires at least 2 GPUs (found " << device_count << ")";
    }

    manual_seed(8501);
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
#else
    GTEST_SKIP() << "CUDA disabled";
#endif
}

INFINI_TRAIN_REGISTER_TEST(GeneratorCoreTest);
