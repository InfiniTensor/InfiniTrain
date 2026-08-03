#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class TensorCopyCpuTest : public ::testing::Test {};

TEST_F(TensorCopyCpuTest, CopiesCPUToCPU) {
    auto source
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    auto target
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    source->Fill(1.0f);
    target->CopyFrom(source);
    test::ExpectTensorEqual(target, 1.0f);
}
