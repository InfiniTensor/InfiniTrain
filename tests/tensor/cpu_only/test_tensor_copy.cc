#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/tensor.h"

using namespace infini_train;

class TensorCopyCpuTest : public ::testing::Test {};

TEST_F(TensorCopyCpuTest, CopiesCPUToCPU) {
    auto source
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    auto target
        = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
    source->Fill(1.0f);
    target->CopyFrom(source);
    auto *target_data = static_cast<float *>(target->DataPtr());
    for (int i = 0; i < 6; ++i) { EXPECT_FLOAT_EQ(target_data[i], 1.0f); }
}
