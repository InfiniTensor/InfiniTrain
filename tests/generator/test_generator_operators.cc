// Generator end-to-end operator integration tests.
//
// These tests verify that random number generation operators (Uniform, Normal,
// KaimingUniform, and Dropout) correctly interact with the Generator mechanism
// on both CPU and CUDA backends, leveraging GTest parameterization.

#include "tests/common/test_utils.h"
#include "infini_train/include/core/cpu_generator.h"
#include "infini_train/include/core/generator.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"
#include <cstdint>
#include <memory>
#include <vector>

#ifdef USE_CUDA
#include "infini_train/include/core/cuda_generator.h"
#endif

using namespace infini_train;

namespace {
Device CPUDevice() {
    return Device(Device::DeviceType::kCPU, 0);
}
} // namespace

class GeneratorOperatorsTest : public infini_train::test::InfiniTrainTest {
protected:
    std::shared_ptr<Tensor> MakeFloatTensor(int64_t n) {
        return std::make_shared<Tensor>(std::vector<int64_t>{n}, DataType::kFLOAT32, GetDevice());
    }

    std::shared_ptr<Tensor> MakeOnesTensor(int64_t n) {
        auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{n}, DataType::kFLOAT32, GetDevice());
        nn::init::Ones(tensor);
        return tensor;
    }

    std::vector<float> ReadFloats(const std::shared_ptr<Tensor> &tensor) {
        const int64_t n = tensor->NumElements();
        std::vector<float> host_data(n);
        auto impl = core::GetDeviceGuardImpl(GetDevice().type());
        core::DeviceGuard guard(GetDevice());
        impl->MemcpyAsync(host_data.data(), tensor->DataPtr(), n * sizeof(float),
                          GetDevice().IsCPU() ? core::MemcpyKind::kD2D : core::MemcpyKind::kD2H,
                          impl->GetStream(GetDevice()));
        impl->SynchronizeStream(impl->GetStream(GetDevice()));
        return host_data;
    }

    core::Generator MakeDeviceGenerator(uint64_t seed) {
        if (GetDevice().IsCPU()) {
            return core::Generator(std::make_shared<core::CPUGeneratorImpl>(seed));
        } else {
#ifdef USE_CUDA
            return core::Generator(std::make_shared<core::CUDAGeneratorImpl>(GetDevice().index(), seed));
#else
            LOG(FATAL) << "CUDA generator requested but USE_CUDA is not defined";
#endif
        }
    }
};

// ============================================================================
// Initializer Tests (Uniform, Normal, KaimingUniform)
// ============================================================================

TEST_P(GeneratorOperatorsTest, UniformReproducibleWithSameSeed) {
    auto t1 = MakeFloatTensor(64);
    auto t2 = MakeFloatTensor(64);

    auto g1 = MakeDeviceGenerator(2024);
    auto g2 = MakeDeviceGenerator(2024);

    nn::init::Uniform(t1, -1.0f, 1.0f, g1);
    nn::init::Uniform(t2, -1.0f, 1.0f, g2);

    EXPECT_EQ(ReadFloats(t1), ReadFloats(t2));
}

TEST_P(GeneratorOperatorsTest, UniformDiffersWithDifferentSeed) {
    auto t1 = MakeFloatTensor(64);
    auto t2 = MakeFloatTensor(64);

    auto g1 = MakeDeviceGenerator(1);
    auto g2 = MakeDeviceGenerator(2);

    nn::init::Uniform(t1, 0.0f, 1.0f, g1);
    nn::init::Uniform(t2, 0.0f, 1.0f, g2);

    EXPECT_NE(ReadFloats(t1), ReadFloats(t2));
}

TEST_P(GeneratorOperatorsTest, RepeatedCallsAdvanceGeneratorState) {
    auto t1 = MakeFloatTensor(32);
    auto t2 = MakeFloatTensor(32);

    auto gen = MakeDeviceGenerator(7);
    nn::init::Uniform(t1, 0.0f, 1.0f, gen);
    nn::init::Uniform(t2, 0.0f, 1.0f, gen);

    EXPECT_NE(ReadFloats(t1), ReadFloats(t2));
}

TEST_P(GeneratorOperatorsTest, NormalReproducibleWithSameSeed) {
    auto t1 = MakeFloatTensor(128);
    auto t2 = MakeFloatTensor(128);

    auto g1 = MakeDeviceGenerator(99);
    auto g2 = MakeDeviceGenerator(99);

    nn::init::Normal(t1, 0.0f, 1.0f, g1);
    nn::init::Normal(t2, 0.0f, 1.0f, g2);

    EXPECT_EQ(ReadFloats(t1), ReadFloats(t2));
}

TEST_P(GeneratorOperatorsTest, KaimingUniformReproducibleWithSameSeed) {
    auto t1 = std::make_shared<Tensor>(std::vector<int64_t>{8, 16}, DataType::kFLOAT32, GetDevice());
    auto t2 = std::make_shared<Tensor>(std::vector<int64_t>{8, 16}, DataType::kFLOAT32, GetDevice());

    auto g1 = MakeDeviceGenerator(31337);
    auto g2 = MakeDeviceGenerator(31337);

    nn::init::KaimingUniform(t1, 0.0f, nn::init::KaimingMode::kFanIn, nn::init::NonLinearityType::kReLU, g1);
    nn::init::KaimingUniform(t2, 0.0f, nn::init::KaimingMode::kFanIn, nn::init::NonLinearityType::kReLU, g2);

    EXPECT_EQ(ReadFloats(t1), ReadFloats(t2));
}

TEST_P(GeneratorOperatorsTest, TensorUniformReproducibleWithSameSeed) {
    auto t1 = MakeFloatTensor(48);
    auto t2 = MakeFloatTensor(48);

    auto g1 = MakeDeviceGenerator(555);
    auto g2 = MakeDeviceGenerator(555);

    t1->Uniform(-2.0f, 2.0f, g1);
    t2->Uniform(-2.0f, 2.0f, g2);

    EXPECT_EQ(ReadFloats(t1), ReadFloats(t2));
}

TEST_P(GeneratorOperatorsTest, DefaultGeneratorReproducibleViaManualSeed) {
    auto t1 = MakeFloatTensor(64);
    auto t2 = MakeFloatTensor(64);

    core::ManualSeed(20240601);
    nn::init::Uniform(t1, 0.0f, 1.0f);

    core::ManualSeed(20240601);
    nn::init::Uniform(t2, 0.0f, 1.0f);

    EXPECT_EQ(ReadFloats(t1), ReadFloats(t2));
}

TEST_P(GeneratorOperatorsTest, ExplicitGeneratorDoesNotConsumeDefault) {
    auto baseline = MakeFloatTensor(32);
    core::ManualSeed(2024);
    nn::init::Uniform(baseline, 0.0f, 1.0f);

    auto with_explicit = MakeFloatTensor(32);
    core::ManualSeed(2024);
    auto explicit_gen = MakeDeviceGenerator(123456);
    auto scratch = MakeFloatTensor(32);
    nn::init::Uniform(scratch, 0.0f, 1.0f, explicit_gen);
    nn::init::Uniform(with_explicit, 0.0f, 1.0f);

    EXPECT_EQ(ReadFloats(baseline), ReadFloats(with_explicit));
}

TEST_P(GeneratorOperatorsTest, KaimingUniformDefaultGeneratorReproducible) {
    auto t1 = std::make_shared<Tensor>(std::vector<int64_t>{16, 32}, DataType::kFLOAT32, GetDevice());
    auto t2 = std::make_shared<Tensor>(std::vector<int64_t>{16, 32}, DataType::kFLOAT32, GetDevice());

    core::ManualSeed(9999);
    nn::init::KaimingUniform(t1);

    core::ManualSeed(9999);
    nn::init::KaimingUniform(t2);

    EXPECT_EQ(ReadFloats(t1), ReadFloats(t2));
}

TEST_P(GeneratorOperatorsTest, NormalDefaultGeneratorReproducible) {
    auto t1 = MakeFloatTensor(128);
    auto t2 = MakeFloatTensor(128);

    core::ManualSeed(7777);
    nn::init::Normal(t1, 0.0f, 0.02f);

    core::ManualSeed(7777);
    nn::init::Normal(t2, 0.0f, 0.02f);

    EXPECT_EQ(ReadFloats(t1), ReadFloats(t2));
}

TEST_P(GeneratorOperatorsTest, RejectsNonFloat32Tensor) {
    auto int_tensor = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kINT32, GetDevice());
    auto gen = MakeDeviceGenerator(1);
    EXPECT_DEATH(nn::init::Uniform(int_tensor, 0.0f, 1.0f, gen), "only supports FLOAT32");
}

TEST_P(GeneratorOperatorsTest, RejectsGeneratorDeviceMismatch) {
#ifdef USE_CUDA
    if (GetDevice().IsCPU()) {
        auto cpu_tensor = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, CPUDevice());
        core::Generator cuda_gen(std::make_shared<core::CUDAGeneratorImpl>(0, 1));
        EXPECT_DEATH(nn::init::Uniform(cpu_tensor, 0.0f, 1.0f, cuda_gen), "device type mismatch");
    } else {
        auto cuda_tensor = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, GetDevice());
        core::Generator cpu_gen(std::make_shared<core::CPUGeneratorImpl>(1));
        EXPECT_DEATH(nn::init::Uniform(cuda_tensor, 0.0f, 1.0f, cpu_gen), "device type mismatch");
    }
#endif
}

// ============================================================================
// Dropout Tests
// ============================================================================

TEST_P(GeneratorOperatorsTest, DropoutReproducibleWithSameSeed) {
    auto t1 = MakeOnesTensor(256);
    auto t2 = MakeOnesTensor(256);

    auto g1 = MakeDeviceGenerator(2024);
    auto g2 = MakeDeviceGenerator(2024);

    nn::init::Dropout(t1, 0.5f, g1);
    nn::init::Dropout(t2, 0.5f, g2);

    EXPECT_EQ(ReadFloats(t1), ReadFloats(t2));
}

TEST_P(GeneratorOperatorsTest, DropoutZerosAndScalesSurvivors) {
    const float p = 0.3f;
    const float scale = 1.0f / (1.0f - p);

    auto tensor = MakeOnesTensor(1024);
    auto gen = MakeDeviceGenerator(7);
    nn::init::Dropout(tensor, p, gen);

    int zeros = 0;
    int survivors = 0;
    for (float v : ReadFloats(tensor)) {
        if (v == 0.0f) {
            ++zeros;
        } else {
            EXPECT_FLOAT_EQ(v, scale);
            ++survivors;
        }
    }
    EXPECT_GT(zeros, 0);
    EXPECT_GT(survivors, 0);
}

TEST_P(GeneratorOperatorsTest, DropoutDifferentSeedProducesDifferentMask) {
    auto t1 = MakeOnesTensor(256);
    auto t2 = MakeOnesTensor(256);

    auto g1 = MakeDeviceGenerator(1);
    auto g2 = MakeDeviceGenerator(2);

    nn::init::Dropout(t1, 0.5f, g1);
    nn::init::Dropout(t2, 0.5f, g2);

    EXPECT_NE(ReadFloats(t1), ReadFloats(t2));
}

TEST_P(GeneratorOperatorsTest, DropoutRepeatedCallsAdvanceGeneratorState) {
    auto t1 = MakeOnesTensor(256);
    auto t2 = MakeOnesTensor(256);

    auto gen = MakeDeviceGenerator(11);
    nn::init::Dropout(t1, 0.5f, gen);
    nn::init::Dropout(t2, 0.5f, gen);

    EXPECT_NE(ReadFloats(t1), ReadFloats(t2));
}

TEST_P(GeneratorOperatorsTest, DropoutUsesDefaultGeneratorReproducibly) {
    auto t1 = MakeOnesTensor(256);
    auto t2 = MakeOnesTensor(256);

    core::ManualSeed(20240601);
    nn::init::Dropout(t1, 0.5f);

    core::ManualSeed(20240601);
    nn::init::Dropout(t2, 0.5f);

    EXPECT_EQ(ReadFloats(t1), ReadFloats(t2));
}

TEST_P(GeneratorOperatorsTest, DropoutBoundaryProbabilityZeroKeepsAllElements) {
    auto tensor = MakeOnesTensor(512);
    auto gen = MakeDeviceGenerator(3);
    nn::init::Dropout(tensor, 0.0f, gen);

    for (float v : ReadFloats(tensor)) {
        EXPECT_FLOAT_EQ(v, 1.0f);
    }
}

TEST_P(GeneratorOperatorsTest, DropoutBoundaryProbabilityNearOneDropsMost) {
    const float p = 0.99f;
    auto tensor = MakeOnesTensor(4096);
    auto gen = MakeDeviceGenerator(123);
    nn::init::Dropout(tensor, p, gen);

    int zeros = 0;
    for (float v : ReadFloats(tensor)) {
        if (v == 0.0f) {
            ++zeros;
        }
    }
    EXPECT_GT(zeros, 4096 * 9 / 10);
}

TEST_P(GeneratorOperatorsTest, DropoutRejectsIllegalProbability) {
    auto gen = MakeDeviceGenerator(1);
    {
        auto tensor = MakeOnesTensor(8);
        EXPECT_DEATH(nn::init::Dropout(tensor, 1.0f, gen), "probability must be in");
    }
    {
        auto tensor = MakeOnesTensor(8);
        EXPECT_DEATH(nn::init::Dropout(tensor, -0.1f, gen), "probability must be in");
    }
}

INFINI_TRAIN_REGISTER_TEST(GeneratorOperatorsTest);
