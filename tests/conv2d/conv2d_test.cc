#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "infini_train/include/tensor.h"
#include "infini_train/include/dispatcher.h"

using namespace infini_train;

TEST(Conv2dTest, ForwardBasic) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 5, 5}, DataType::kFLOAT32);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, 3, 3}, DataType::kFLOAT32);

    float *in_ptr = static_cast<float *>(input->DataPtr());
    for (int i = 0; i < 25; ++i) in_ptr[i] = static_cast<float>(i);

    float *w_ptr = static_cast<float *>(weight->DataPtr());
    for (int i = 0; i < 9; ++i) w_ptr[i] = 1.0f;

    auto output = Dispatcher::Instance().Call<std::shared_ptr<Tensor>>(
        {Device::DeviceType::kCPU, "Conv2dForward"}, input, weight, nullptr, 1, 0);

    float *out_ptr = static_cast<float *>(output->DataPtr());
    EXPECT_NEAR(out_ptr[0], 54.0f, 1e-5);
    EXPECT_NEAR(out_ptr[1], 63.0f, 1e-5);
    EXPECT_NEAR(out_ptr[4], 108.0f, 1e-5);
}