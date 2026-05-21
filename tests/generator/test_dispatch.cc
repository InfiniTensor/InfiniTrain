#include <cstring>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class GeneratorDispatchTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorDispatchTest, NullGeneratorFallsBackToDefault) {
    if (GetParam() != Device::DeviceType::kCPU) {
        GTEST_SKIP() << "CUDA random kernels are Phase 2";
    }
    manual_seed(13);
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, GetDevice());
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, GetDevice());

    auto resolved = ResolveGenerator(std::nullopt, GetDevice());
    Dispatcher::Instance().Call<void>({GetDevice().type(), "UniformRandom"}, a, 0.0f, 1.0f, resolved.get());

    manual_seed(13);
    auto resolved2 = ResolveGenerator(std::nullopt, GetDevice());
    Dispatcher::Instance().Call<void>({GetDevice().type(), "UniformRandom"}, b, 0.0f, 1.0f, resolved2.get());

    Tensor a_cpu = a->To(Device(Device::DeviceType::kCPU, 0));
    Tensor b_cpu = b->To(Device(Device::DeviceType::kCPU, 0));
    EXPECT_EQ(0, std::memcmp(a_cpu.DataPtr(), b_cpu.DataPtr(), 8 * sizeof(float)));
}

INFINI_TRAIN_REGISTER_TEST(GeneratorDispatchTest);
