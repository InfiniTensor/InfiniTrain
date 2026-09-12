#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/dropout.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

#include "infini_train/include/device.h"
#include "tests/common/test_utils.h"

using namespace infini_train;
using infini_train::test::TensorBytes;

namespace {

template <typename T> void ExpectUniformRangeAndFinite(const Tensor &tensor) {
    const auto *data = static_cast<const T *>(tensor.DataPtr());
    for (int64_t index = 0; index < tensor.NumElements(); ++index) {
        const double value = static_cast<double>(data[index]);
        EXPECT_TRUE(std::isfinite(value));
        EXPECT_GE(value, 0.0);
        EXPECT_LT(value, 1.0);
    }
}

template <typename T> void ExpectFinite(const Tensor &tensor) {
    const auto *data = static_cast<const T *>(tensor.DataPtr());
    for (int64_t index = 0; index < tensor.NumElements(); ++index) {
        EXPECT_TRUE(std::isfinite(static_cast<double>(data[index])));
    }
}

void CheckUniformRangeAndFinite(const std::shared_ptr<Tensor> &tensor) {
    const Tensor host = tensor->To(Device());
    const Device source_device = tensor->GetDevice();
    core::GetDeviceGuardImpl(source_device.type())->SynchronizeDevice(source_device);
    switch (host.Dtype()) {
    case DataType::kFLOAT16:
        ExpectUniformRangeAndFinite<FP16>(host);
        break;
    case DataType::kBFLOAT16:
        ExpectUniformRangeAndFinite<BF16>(host);
        break;
    case DataType::kFLOAT32:
        ExpectUniformRangeAndFinite<float>(host);
        break;
    case DataType::kFLOAT64:
        ExpectUniformRangeAndFinite<double>(host);
        break;
    default:
        FAIL() << "Unexpected dtype";
    }
}

void CheckFinite(const std::shared_ptr<Tensor> &tensor) {
    const Tensor host = tensor->To(Device());
    const Device source_device = tensor->GetDevice();
    core::GetDeviceGuardImpl(source_device.type())->SynchronizeDevice(source_device);
    switch (host.Dtype()) {
    case DataType::kFLOAT16:
        ExpectFinite<FP16>(host);
        break;
    case DataType::kBFLOAT16:
        ExpectFinite<BF16>(host);
        break;
    case DataType::kFLOAT32:
        ExpectFinite<float>(host);
        break;
    case DataType::kFLOAT64:
        ExpectFinite<double>(host);
        break;
    default:
        FAIL() << "Unexpected dtype";
    }
}

std::vector<std::vector<uint8_t>> RunRandomScript(const Device &device, std::optional<Generator> generator) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{257}, DataType::kFLOAT32, device);
    input->Fill(1.0f);
    return {
        TensorBytes(nn::function::Rand({257}, DataType::kFLOAT32, device, generator)),
        TensorBytes(nn::function::Randn({257}, DataType::kFLOAT32, device, generator)),
        TensorBytes(nn::function::Dropout(input, 0.25, true, generator)),
        TensorBytes(nn::function::Rand({257}, DataType::kFLOAT32, device, generator)),
    };
}

void ExpectMaskValues(const std::shared_ptr<Tensor> &mask) {
    EXPECT_EQ(mask->Dtype(), DataType::kUINT8);
    const Tensor host = mask->To(Device());
    const Device source_device = mask->GetDevice();
    core::GetDeviceGuardImpl(source_device.type())->SynchronizeDevice(source_device);
    const auto *data = static_cast<const uint8_t *>(host.DataPtr());
    bool saw_zero = false;
    bool saw_one = false;
    for (int64_t index = 0; index < host.NumElements(); ++index) {
        EXPECT_TRUE(data[index] == 0 || data[index] == 1);
        saw_zero |= data[index] == 0;
        saw_one |= data[index] == 1;
    }
    EXPECT_TRUE(saw_zero);
    EXPECT_TRUE(saw_one);
}

template <typename T> void ExpectUnitInputDropoutValues(const Tensor &output, const Tensor &mask, double p) {
    const auto *output_data = static_cast<const T *>(output.DataPtr());
    const auto *mask_data = static_cast<const uint8_t *>(mask.DataPtr());
    const double kept_value = 1.0 / (1.0 - p);
    for (int64_t index = 0; index < output.NumElements(); ++index) {
        const double expected = mask_data[index] ? kept_value : 0.0;
        EXPECT_NEAR(static_cast<double>(output_data[index]), expected, 0.02);
    }
}

void ExpectUnitInputDropoutValues(const std::shared_ptr<Tensor> &output, const std::shared_ptr<Tensor> &mask,
                                  double p) {
    const Tensor host_output = output->To(Device());
    const Tensor host_mask = mask->To(Device());
    // Each async D2H copy must complete before its host buffer is read.
    const Device output_device = output->GetDevice();
    core::GetDeviceGuardImpl(output_device.type())->SynchronizeDevice(output_device);
    const Device mask_source_device = mask->GetDevice();
    core::GetDeviceGuardImpl(mask_source_device.type())->SynchronizeDevice(mask_source_device);
    switch (host_output.Dtype()) {
    case DataType::kFLOAT16:
        ExpectUnitInputDropoutValues<FP16>(host_output, host_mask, p);
        break;
    case DataType::kBFLOAT16:
        ExpectUnitInputDropoutValues<BF16>(host_output, host_mask, p);
        break;
    case DataType::kFLOAT32:
        ExpectUnitInputDropoutValues<float>(host_output, host_mask, p);
        break;
    case DataType::kFLOAT64:
        ExpectUnitInputDropoutValues<double>(host_output, host_mask, p);
        break;
    default:
        FAIL() << "Unexpected dtype";
    }
}

} // namespace

class GeneratorRandomOpsTest : public infini_train::test::InfiniTrainTest {};

TEST_P(GeneratorRandomOpsTest, RandAndRandnSupportAllFloatingDtypes) {
    const std::vector<DataType> dtypes
        = {DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64};
    const Device device = GetDevice();

    for (const auto dtype : dtypes) {
        auto uniform_first = CreateGenerator(device, 1001);
        auto uniform_replay = CreateGenerator(device, 1001);
        auto uniform = nn::function::Rand({4097}, dtype, device, uniform_first);
        EXPECT_EQ(TensorBytes(uniform), TensorBytes(nn::function::Rand({4097}, dtype, device, uniform_replay)));
        CheckUniformRangeAndFinite(uniform);

        auto normal_first = CreateGenerator(device, 2002);
        auto normal_replay = CreateGenerator(device, 2002);
        auto normal = nn::function::Randn({4097}, dtype, device, normal_first);
        EXPECT_EQ(TensorBytes(normal), TensorBytes(nn::function::Randn({4097}, dtype, device, normal_replay)));
        CheckFinite(normal);
    }
}

TEST_P(GeneratorRandomOpsTest, UniformConstantDoesNotAdvanceGenerator) {
    const Device device = GetDevice();
    for (const auto dtype : {DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        auto generator = CreateGenerator(device, 2101);
        nn::function::Rand({11}, dtype, device, generator);
        const auto state_before = TensorBytes(generator.get_state());
        auto replay = generator.Clone();
        auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{257}, dtype, device);
        auto expected = std::make_shared<Tensor>(tensor->Dims(), dtype, Device());
        expected->Fill(0.1f);

        nn::init::Uniform(tensor, 0.1f, 0.1f, generator);

        EXPECT_EQ(TensorBytes(tensor), TensorBytes(expected));
        EXPECT_EQ(TensorBytes(generator.get_state()), state_before);
        EXPECT_EQ(TensorBytes(nn::function::Rand({129}, dtype, device, generator)),
                  TensorBytes(nn::function::Rand({129}, dtype, device, replay)));
    }
}

TEST_P(GeneratorRandomOpsTest, UniformEmptyTensorDoesNotAdvanceGenerator) {
    const Device device = GetDevice();
    for (const auto dtype : {DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        auto generator = CreateGenerator(device, 2102);
        nn::function::Rand({11}, dtype, device, generator);
        const auto state_before = TensorBytes(generator.get_state());
        auto replay = generator.Clone();
        auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{2, 0, 3}, dtype, device);

        nn::init::Uniform(tensor, -2.0f, 3.0f, generator);

        EXPECT_EQ(tensor->NumElements(), 0);
        EXPECT_EQ(TensorBytes(generator.get_state()), state_before);
        EXPECT_EQ(TensorBytes(nn::function::Rand({129}, dtype, device, generator)),
                  TensorBytes(nn::function::Rand({129}, dtype, device, replay)));
    }
}

TEST_P(GeneratorRandomOpsTest, DropoutSupportsAllFloatingDtypesWithExplicitGenerator) {
    const std::vector<DataType> dtypes
        = {DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64};
    const Device device = GetDevice();

    for (const auto dtype : dtypes) {
        auto input = nn::function::Rand({4097}, dtype, device, CreateGenerator(device, 3003));
        ManualSeed(4004);
        const auto default_state_before = TensorBytes(GetDefaultGenerator(device).get_state());

        auto dropout_first = CreateGenerator(device, 5005);
        auto dropout_replay = CreateGenerator(device, 5005);
        auto output = nn::function::Dropout(input, 0.25, true, dropout_first);
        EXPECT_EQ(TensorBytes(output), TensorBytes(nn::function::Dropout(input, 0.25, true, dropout_replay)));
        CheckFinite(output);
        EXPECT_EQ(default_state_before, TensorBytes(GetDefaultGenerator(device).get_state()));
    }
}

TEST_P(GeneratorRandomOpsTest, SameSeedReplaysRandRandnAndDropoutScript) {
    const Device device = GetDevice();
    EXPECT_EQ(RunRandomScript(device, CreateGenerator(device, 5678)),
              RunRandomScript(device, CreateGenerator(device, 5678)));
    EXPECT_NE(RunRandomScript(device, CreateGenerator(device, 5678)),
              RunRandomScript(device, CreateGenerator(device, 5679)));

    auto generator = CreateGenerator(device, 6789);
    const auto first = TensorBytes(nn::function::Rand({256}, DataType::kFLOAT32, device, generator));
    const auto second = TensorBytes(nn::function::Rand({256}, DataType::kFLOAT32, device, generator));
    EXPECT_NE(first, second);
}

TEST_P(GeneratorRandomOpsTest, DefaultGeneratorReplaysRandRandnAndDropoutScript) {
    const Device device = GetDevice();

    ManualSeed(7001);
    const auto expected = RunRandomScript(device, std::nullopt);
    ManualSeed(7001);
    EXPECT_EQ(expected, RunRandomScript(device, std::nullopt));

    ManualSeed(7001);
    std::optional<Generator> undefined = std::nullopt;
    EXPECT_EQ(expected, RunRandomScript(device, undefined));
}

TEST_P(GeneratorRandomOpsTest, ExplicitRandomGeneratorsDoNotAdvanceDefaultGenerator) {
    const Device device = GetDevice();
    ManualSeed(7101);
    const auto default_state_before = TensorBytes(GetDefaultGenerator(device).get_state());

    RunRandomScript(device, CreateGenerator(device, 7102));

    EXPECT_EQ(default_state_before, TensorBytes(GetDefaultGenerator(device).get_state()));
}

TEST_P(GeneratorRandomOpsTest, OffsetViewsUseTheirOwnDataPointers) {
    const Device device = GetDevice();
    constexpr int64_t kStorageElements = 12;
    constexpr int64_t kOffsetElements = 4;
    constexpr int64_t kViewElements = 4;
    constexpr size_t kOffsetBytes = kOffsetElements * sizeof(float);

    auto storage = std::make_shared<Tensor>(std::vector<int64_t>{kStorageElements}, DataType::kFLOAT32, device);
    storage->Fill(-1.0f);
    auto uniform_view = std::make_shared<Tensor>(*storage, kOffsetBytes, std::vector<int64_t>{kViewElements});
    nn::init::Uniform(uniform_view, 0.0f, 1.0f, CreateGenerator(device, 7201));

    const Tensor uniform_storage = storage->To(Device());
    const Device storage_device = storage->GetDevice();
    core::GetDeviceGuardImpl(storage_device.type())->SynchronizeDevice(storage_device);
    const auto *uniform_data = static_cast<const float *>(uniform_storage.DataPtr());
    for (int64_t index = 0; index < kOffsetElements; ++index) { EXPECT_FLOAT_EQ(uniform_data[index], -1.0f); }
    for (int64_t index = kOffsetElements; index < kOffsetElements + kViewElements; ++index) {
        EXPECT_GE(uniform_data[index], 0.0f);
        EXPECT_LT(uniform_data[index], 1.0f);
    }
    for (int64_t index = kOffsetElements + kViewElements; index < kStorageElements; ++index) {
        EXPECT_FLOAT_EQ(uniform_data[index], -1.0f);
    }

    auto input_storage = std::make_shared<Tensor>(std::vector<int64_t>{kStorageElements}, DataType::kFLOAT32, device);
    input_storage->Fill(-3.0f);
    auto input_view = std::make_shared<Tensor>(*input_storage, kOffsetBytes, std::vector<int64_t>{kViewElements});
    input_view->Fill(1.0f);
    const auto output = nn::function::Dropout(input_view, 0.5, true, CreateGenerator(device, 7202));
    const Tensor host_output = output->To(Device());
    const Device output_device = output->GetDevice();
    core::GetDeviceGuardImpl(output_device.type())->SynchronizeDevice(output_device);
    const auto *output_data = static_cast<const float *>(host_output.DataPtr());
    for (int64_t index = 0; index < kViewElements; ++index) {
        EXPECT_TRUE(output_data[index] == 0.0f || std::abs(output_data[index] - 2.0f) < 1e-6f);
    }

    const Tensor input_storage_host = input_storage->To(Device());
    const Device input_storage_device = input_storage->GetDevice();
    core::GetDeviceGuardImpl(input_storage_device.type())->SynchronizeDevice(input_storage_device);
    const auto *input_storage_data = static_cast<const float *>(input_storage_host.DataPtr());
    for (int64_t index = 0; index < kOffsetElements; ++index) { EXPECT_FLOAT_EQ(input_storage_data[index], -3.0f); }
    for (int64_t index = kOffsetElements + kViewElements; index < kStorageElements; ++index) {
        EXPECT_FLOAT_EQ(input_storage_data[index], -3.0f);
    }
}

TEST_P(GeneratorRandomOpsTest, DropoutBackwardSupportsAllFloatingDtypes) {
    const Device device = GetDevice();
    const std::vector<DataType> dtypes
        = {DataType::kFLOAT16, DataType::kBFLOAT16, DataType::kFLOAT32, DataType::kFLOAT64};

    for (const auto dtype : dtypes) {
        auto input = std::make_shared<Tensor>(std::vector<int64_t>{4097}, dtype, device, true);
        input->Fill(1.0f);
        auto dropout = std::make_shared<autograd::Dropout>(0.25, CreateGenerator(device, 8201));
        const auto outputs = dropout->Apply({input});
        ASSERT_EQ(outputs.size(), 2U);
        ExpectMaskValues(outputs[1]);
        ExpectUnitInputDropoutValues(outputs[0], outputs[1], 0.25);

        auto grad_output = std::make_shared<Tensor>(input->Dims(), dtype, device);
        grad_output->Fill(1.0f);
        const auto grad_inputs = dropout->Backward({grad_output, nullptr});
        ASSERT_EQ(grad_inputs.size(), 1U);
        EXPECT_EQ(TensorBytes(outputs[0]), TensorBytes(grad_inputs[0]));
    }
}

TEST_P(GeneratorRandomOpsTest, DropoutIdentityBoundariesDoNotAdvanceGenerator) {
    const Device device = GetDevice();
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{64}, DataType::kFLOAT32, device, true);
    input->Fill(1.0f);
    auto generator = CreateGenerator(device, 8301);

    auto expect_unchanged = [&](const auto &operation) {
        const auto before = TensorBytes(generator.get_state());
        operation();
        EXPECT_EQ(before, TensorBytes(generator.get_state()));
    };

    expect_unchanged([&] { EXPECT_EQ(nn::function::Dropout(input, 0.5, false, generator), input); });
    expect_unchanged([&] { EXPECT_EQ(nn::function::Dropout(input, 0.0, true, generator), input); });
}

TEST_P(GeneratorRandomOpsTest, DropoutP1ProducesZeroForwardAndBackwardWithoutAdvancing) {
    const Device device = GetDevice();
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{64}, DataType::kFLOAT32, device, true);
    input->Fill(1.0f);
    auto generator = CreateGenerator(device, 8301);
    const auto before = TensorBytes(generator.get_state());

    auto all_dropped = nn::function::Dropout(input, 1.0, true, generator);
    EXPECT_EQ(before, TensorBytes(generator.get_state()));
    auto zeros = std::make_shared<Tensor>(input->Dims(), input->Dtype(), device);
    zeros->Fill(0.0f);
    EXPECT_EQ(TensorBytes(all_dropped), TensorBytes(zeros));

    auto dropout = std::make_shared<autograd::Dropout>(1.0, generator);
    const auto outputs = dropout->Apply({input});
    auto grad_output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), device);
    grad_output->Fill(1.0f);
    const auto grad_inputs = dropout->Backward({grad_output, nullptr});
    EXPECT_EQ(TensorBytes(grad_inputs[0]), TensorBytes(zeros));
}

TEST_P(GeneratorRandomOpsTest, InvalidDistributionParametersAreRejected) {
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    const Device device = GetDevice();
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kFLOAT32, device);
    auto integer_tensor = std::make_shared<Tensor>(std::vector<int64_t>{8}, DataType::kINT32, device);
    auto generator = CreateGenerator(device, 9001);

    EXPECT_DEATH(nn::init::Uniform(tensor, 2.0f, 1.0f, generator), "uniform expects a");
    EXPECT_DEATH(nn::init::Normal(tensor, 0.0f, -1.0f, generator), "std >= 0.0");
    EXPECT_DEATH(nn::function::Dropout(tensor, -0.1, true, generator), "between 0 and 1");
    EXPECT_DEATH(nn::function::Dropout(tensor, 1.1, true, generator), "between 0 and 1");
    EXPECT_DEATH(nn::function::Rand({8}, DataType::kINT32, device, generator), "floating-point");
    EXPECT_DEATH(nn::init::Uniform(integer_tensor, 0.0f, 1.0f, generator), "floating-point");
    EXPECT_DEATH(nn::init::Normal(integer_tensor, 0.0f, 1.0f, generator), "floating-point");
}

INFINI_TRAIN_REGISTER_TEST(GeneratorRandomOpsTest);
