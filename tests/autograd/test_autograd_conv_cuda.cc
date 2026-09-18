// CUDA Conv2d forward parity vs CPU direct kernel (im2col + GEMM path).
// Grid: B{1,2} x Cin{1,3} x Cout{2,3} x H!=W x K{1,3} x s{1,2} x p{0,1},
// with and without bias. Threshold: abs err <= 1e-5.
// Plus weight-grad parity (im2col + batched GEMM + N-reduce) over a smaller
// grid: B{1,2} x Cin{1,3} x Cout{2,3} x K{1,3} x s{1,2} x p{0,1}. Threshold: abs err <= 1e-5.
// Plus input-grad parity (W^T * dY GEMM + gather col2im) over the same
// smaller grid. Threshold: abs err <= 1e-5.
// Plus bias-grad parity (sum over N*Hout*Wout) over B{1,2} x Cin{1,3} x Cout{2,3}.
// Threshold: abs err <= 1e-5.
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/conv.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class AutogradConvCudaTest : public infini_train::test::InfiniTrainTest {};

namespace {

std::vector<float> DeterministicVals(int64_t n, int seed) {
    std::vector<float> vals(static_cast<size_t>(n));
    uint32_t state = static_cast<uint32_t>(seed);
    for (int64_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        vals[static_cast<size_t>(i)] = static_cast<float>(state % 1000) / 1000.0f - 0.5f;
    }
    return vals;
}

void CheckParity(const Device &cuda_device, int64_t batch, int64_t cin, int64_t cout, int64_t h, int64_t w, int64_t k,
                 int64_t stride, int64_t padding, bool with_bias) {
    const Device cpu_device = Device();
    const std::vector<int64_t> in_dims = {batch, cin, h, w};
    const std::vector<int64_t> w_dims = {cout, cin, k, k};
    const std::vector<int64_t> b_dims = {cout};

    int64_t in_n = batch * cin * h * w;
    int64_t w_n = cout * cin * k * k;
    auto in_vals = DeterministicVals(in_n, 11);
    auto w_vals = DeterministicVals(w_n, 23);
    auto b_vals = DeterministicVals(cout, 37);

    auto cpu_in = std::make_shared<Tensor>(in_vals.data(), in_dims, DataType::kFLOAT32, cpu_device);
    auto cpu_w = std::make_shared<Tensor>(w_vals.data(), w_dims, DataType::kFLOAT32, cpu_device);
    auto cpu_b = with_bias ? std::make_shared<Tensor>(b_vals.data(), b_dims, DataType::kFLOAT32, cpu_device) : nullptr;

    auto cuda_in = std::make_shared<Tensor>(cpu_in->To(cuda_device));
    auto cuda_w = std::make_shared<Tensor>(cpu_w->To(cuda_device));
    std::shared_ptr<Tensor> cuda_b = nullptr;
    if (with_bias) {
        cuda_b = std::make_shared<Tensor>(cpu_b->To(cuda_device));
    }

    auto cpu_fn = std::make_shared<autograd::Conv2d>(stride, padding);
    auto cuda_fn = std::make_shared<autograd::Conv2d>(stride, padding);
    std::vector<std::shared_ptr<Tensor>> cpu_args = with_bias
                                                      ? std::vector<std::shared_ptr<Tensor>>{cpu_in, cpu_w, cpu_b}
                                                      : std::vector<std::shared_ptr<Tensor>>{cpu_in, cpu_w};
    std::vector<std::shared_ptr<Tensor>> cuda_args = with_bias
                                                       ? std::vector<std::shared_ptr<Tensor>>{cuda_in, cuda_w, cuda_b}
                                                       : std::vector<std::shared_ptr<Tensor>>{cuda_in, cuda_w};
    auto cpu_out = cpu_fn->Apply(cpu_args);
    auto cuda_out = cuda_fn->Apply(cuda_args);
    ASSERT_EQ(cpu_out.size(), 1);
    ASSERT_EQ(cuda_out.size(), 1);
    EXPECT_EQ(cuda_out[0]->Dims(), cpu_out[0]->Dims());

    auto expected_cpu = cpu_out[0]->To(Device());
    const float *expected = static_cast<const float *>(expected_cpu.DataPtr());
    test::ExpectTensorNear(cuda_out[0], std::vector<float>(expected, expected + expected_cpu.NumElements()), 1e-5f);
}

} // namespace

void CheckWeightGradParity(const Device &cuda_device, int64_t batch, int64_t cin, int64_t cout, int64_t h, int64_t w,
                           int64_t k, int64_t stride, int64_t padding) {
    const Device cpu_device = Device();
    const std::vector<int64_t> in_dims = {batch, cin, h, w};
    const std::vector<int64_t> w_dims = {cout, cin, k, k};
    const int64_t oh = (h + 2 * padding - k) / stride + 1;
    const int64_t ow = (w + 2 * padding - k) / stride + 1;
    const std::vector<int64_t> go_dims = {batch, cout, oh, ow};

    auto in_vals = DeterministicVals(batch * cin * h * w, 11);
    auto w_vals = DeterministicVals(cout * cin * k * k, 23);
    auto go_vals = DeterministicVals(batch * cout * oh * ow, 41);

    auto cpu_in = std::make_shared<Tensor>(in_vals.data(), in_dims, DataType::kFLOAT32, cpu_device);
    auto cpu_w = std::make_shared<Tensor>(w_vals.data(), w_dims, DataType::kFLOAT32, cpu_device)->RequiresGrad();
    auto cpu_go = std::make_shared<Tensor>(go_vals.data(), go_dims, DataType::kFLOAT32, cpu_device);

    // NOTE: input deliberately does NOT require grad (weight-grad path only;
    // CUDA Conv2dBackwardInput is a different task), so grads[0] is null.
    auto cuda_in = std::make_shared<Tensor>(cpu_in->To(cuda_device));
    auto cuda_w = std::make_shared<Tensor>(cpu_w->To(cuda_device))->RequiresGrad();
    auto cuda_go = std::make_shared<Tensor>(cpu_go->To(cuda_device));

    auto cpu_fn = std::make_shared<autograd::Conv2d>(stride, padding);
    auto cuda_fn = std::make_shared<autograd::Conv2d>(stride, padding);
    cpu_fn->Apply({cpu_in, cpu_w});
    cuda_fn->Apply({cuda_in, cuda_w});

    auto cpu_grads = cpu_fn->Backward({cpu_go});
    auto cuda_grads = cuda_fn->Backward({cuda_go});
    ASSERT_EQ(cpu_grads.size(), 2);
    ASSERT_EQ(cuda_grads.size(), 2);
    EXPECT_EQ(cuda_grads[0], nullptr);
    ASSERT_NE(cuda_grads[1], nullptr);
    EXPECT_EQ(cuda_grads[1]->Dims(), cpu_grads[1]->Dims());

    auto expected_cpu = cpu_grads[1]->To(Device());
    const float *expected = static_cast<const float *>(expected_cpu.DataPtr());
    test::ExpectTensorNear(cuda_grads[1], std::vector<float>(expected, expected + expected_cpu.NumElements()), 1e-5f);
}

void CheckBiasGradParity(const Device &cuda_device, int64_t batch, int64_t cin, int64_t cout) {
    const Device cpu_device = Device();
    const int64_t h = 5;
    const int64_t w = 4;
    const int64_t k = 3;
    const int64_t stride = 1;
    const int64_t padding = 1;
    const std::vector<int64_t> in_dims = {batch, cin, h, w};
    const std::vector<int64_t> w_dims = {cout, cin, k, k};
    const std::vector<int64_t> b_dims = {cout};
    const int64_t oh = (h + 2 * padding - k) / stride + 1;
    const int64_t ow = (w + 2 * padding - k) / stride + 1;
    const std::vector<int64_t> go_dims = {batch, cout, oh, ow};

    auto in_vals = DeterministicVals(batch * cin * h * w, 11);
    auto w_vals = DeterministicVals(cout * cin * k * k, 23);
    auto b_vals = DeterministicVals(cout, 37);
    auto go_vals = DeterministicVals(batch * cout * oh * ow, 41);

    auto cpu_in = std::make_shared<Tensor>(in_vals.data(), in_dims, DataType::kFLOAT32, cpu_device);
    auto cpu_w = std::make_shared<Tensor>(w_vals.data(), w_dims, DataType::kFLOAT32, cpu_device);
    auto cpu_b = std::make_shared<Tensor>(b_vals.data(), b_dims, DataType::kFLOAT32, cpu_device)->RequiresGrad();
    auto cpu_go = std::make_shared<Tensor>(go_vals.data(), go_dims, DataType::kFLOAT32, cpu_device);

    // NOTE: input/weight deliberately do NOT require grad (bias-grad path
    // only), so grads[0] and grads[1] are null.
    auto cuda_in = std::make_shared<Tensor>(cpu_in->To(cuda_device));
    auto cuda_w = std::make_shared<Tensor>(cpu_w->To(cuda_device));
    auto cuda_b = std::make_shared<Tensor>(cpu_b->To(cuda_device))->RequiresGrad();
    auto cuda_go = std::make_shared<Tensor>(cpu_go->To(cuda_device));

    auto cpu_fn = std::make_shared<autograd::Conv2d>(stride, padding);
    auto cuda_fn = std::make_shared<autograd::Conv2d>(stride, padding);
    cpu_fn->Apply({cpu_in, cpu_w, cpu_b});
    cuda_fn->Apply({cuda_in, cuda_w, cuda_b});

    auto cpu_grads = cpu_fn->Backward({cpu_go});
    auto cuda_grads = cuda_fn->Backward({cuda_go});
    ASSERT_EQ(cpu_grads.size(), 3);
    ASSERT_EQ(cuda_grads.size(), 3);
    EXPECT_EQ(cuda_grads[0], nullptr);
    EXPECT_EQ(cuda_grads[1], nullptr);
    ASSERT_NE(cuda_grads[2], nullptr);
    EXPECT_EQ(cuda_grads[2]->Dims(), cpu_grads[2]->Dims());

    auto expected_cpu = cpu_grads[2]->To(Device());
    const float *expected = static_cast<const float *>(expected_cpu.DataPtr());
    test::ExpectTensorNear(cuda_grads[2], std::vector<float>(expected, expected + expected_cpu.NumElements()), 1e-5f);
}

void CheckInputGradParity(const Device &cuda_device, int64_t batch, int64_t cin, int64_t cout, int64_t h, int64_t w,
                          int64_t k, int64_t stride, int64_t padding) {
    const Device cpu_device = Device();
    const std::vector<int64_t> in_dims = {batch, cin, h, w};
    const std::vector<int64_t> w_dims = {cout, cin, k, k};
    const int64_t oh = (h + 2 * padding - k) / stride + 1;
    const int64_t ow = (w + 2 * padding - k) / stride + 1;
    const std::vector<int64_t> go_dims = {batch, cout, oh, ow};

    auto in_vals = DeterministicVals(batch * cin * h * w, 11);
    auto w_vals = DeterministicVals(cout * cin * k * k, 23);
    auto go_vals = DeterministicVals(batch * cout * oh * ow, 41);

    auto cpu_in = std::make_shared<Tensor>(in_vals.data(), in_dims, DataType::kFLOAT32, cpu_device)->RequiresGrad();
    auto cpu_w = std::make_shared<Tensor>(w_vals.data(), w_dims, DataType::kFLOAT32, cpu_device);
    auto cpu_go = std::make_shared<Tensor>(go_vals.data(), go_dims, DataType::kFLOAT32, cpu_device);

    // NOTE: weight deliberately does NOT require grad (input-grad path only),
    // so grads[1] is null.
    auto cuda_in = std::make_shared<Tensor>(cpu_in->To(cuda_device))->RequiresGrad();
    auto cuda_w = std::make_shared<Tensor>(cpu_w->To(cuda_device));
    auto cuda_go = std::make_shared<Tensor>(cpu_go->To(cuda_device));

    auto cpu_fn = std::make_shared<autograd::Conv2d>(stride, padding);
    auto cuda_fn = std::make_shared<autograd::Conv2d>(stride, padding);
    cpu_fn->Apply({cpu_in, cpu_w});
    cuda_fn->Apply({cuda_in, cuda_w});

    auto cpu_grads = cpu_fn->Backward({cpu_go});
    auto cuda_grads = cuda_fn->Backward({cuda_go});
    ASSERT_EQ(cpu_grads.size(), 2);
    ASSERT_EQ(cuda_grads.size(), 2);
    ASSERT_NE(cuda_grads[0], nullptr);
    EXPECT_EQ(cuda_grads[1], nullptr);
    EXPECT_EQ(cuda_grads[0]->Dims(), cpu_grads[0]->Dims());

    auto expected_cpu = cpu_grads[0]->To(Device());
    const float *expected = static_cast<const float *>(expected_cpu.DataPtr());
    test::ExpectTensorNear(cuda_grads[0], std::vector<float>(expected, expected + expected_cpu.NumElements()), 1e-5f);
}

TEST_P(AutogradConvCudaTest, Conv2dBackwardBiasParityGrid) {
    ONLY_CUDA();
    const Device cuda_device = GetDevice();
    for (int64_t batch : {1, 2}) {
        for (int64_t cin : {1, 3}) {
            for (int64_t cout : {2, 3}) { CheckBiasGradParity(cuda_device, batch, cin, cout); }
        }
    }
}

TEST_P(AutogradConvCudaTest, Conv2dBackwardInputParityGrid) {
    ONLY_CUDA();
    const Device cuda_device = GetDevice();
    for (int64_t batch : {1, 2}) {
        for (int64_t cin : {1, 3}) {
            for (int64_t cout : {2, 3}) {
                for (int64_t k : {1, 3}) {
                    for (int64_t stride : {1, 2}) {
                        for (int64_t padding : {0, 1}) {
                            // H != W; H + 2p >= K and W + 2p >= K always hold here.
                            CheckInputGradParity(cuda_device, batch, cin, cout, /*h=*/5, /*w=*/4, k, stride, padding);
                        }
                    }
                }
            }
        }
    }
}

TEST_P(AutogradConvCudaTest, Conv2dBackwardWeightParityGrid) {
    ONLY_CUDA();
    const Device cuda_device = GetDevice();
    for (int64_t batch : {1, 2}) {
        for (int64_t cin : {1, 3}) {
            for (int64_t cout : {2, 3}) {
                for (int64_t k : {1, 3}) {
                    for (int64_t stride : {1, 2}) {
                        for (int64_t padding : {0, 1}) {
                            // H != W; H + 2p >= K and W + 2p >= K always hold here.
                            CheckWeightGradParity(cuda_device, batch, cin, cout, /*h=*/5, /*w=*/4, k, stride, padding);
                        }
                    }
                }
            }
        }
    }
}

TEST_P(AutogradConvCudaTest, Conv2dForwardParityGrid) {
    ONLY_CUDA();
    const Device cuda_device = GetDevice();
    for (int64_t batch : {1, 2}) {
        for (int64_t cin : {1, 3}) {
            for (int64_t cout : {2, 3}) {
                for (int64_t k : {1, 3}) {
                    for (int64_t stride : {1, 2}) {
                        for (int64_t padding : {0, 1}) {
                            for (bool with_bias : {false, true}) {
                                // H != W; H + 2p >= K and W + 2p >= K always hold here.
                                CheckParity(cuda_device, batch, cin, cout, /*h=*/5, /*w=*/4, k, stride, padding,
                                            with_bias);
                            }
                        }
                    }
                }
            }
        }
    }
}

INFINI_TRAIN_REGISTER_TEST(AutogradConvCudaTest);
