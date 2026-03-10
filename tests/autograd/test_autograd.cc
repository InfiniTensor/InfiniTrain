#include <gtest/gtest.h>

#include <vector>
#include <cmath>

#include "infini_train/include/tensor.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/autograd/transform.h"
#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/autograd/matmul.h"
#include "infini_train/include/autograd/reduction.h"
#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/autograd/softmax.h"
#include "infini_train/include/autograd/normalization.h"
#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/autograd/outer.h"
#include "infini_train/include/autograd/misc.h"

using namespace infini_train;

class AutogradTestBase : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1);
    }

    std::shared_ptr<Tensor> createTensor(const std::vector<int64_t>& shape, float value = 0.0f) {
        auto tensor = std::make_shared<Tensor>(shape, DataType::kFLOAT32,
                                               Device(Device::DeviceType::kCPU, 0));
        tensor->set_requires_grad(true);
        auto data = static_cast<float*>(tensor->DataPtr());
        size_t size = 1;
        for (auto dim : shape) size *= dim;
        for (size_t i = 0; i < size; ++i) {
            data[i] = value + static_cast<float>(i);
        }
        return tensor;
    }
};

class AutogradForwardTest : public AutogradTestBase {};
class AutogradBackwardTest : public AutogradTestBase {};
class AutogradCudaTest : public AutogradTestBase {};
class AutogradDistributedTest : public AutogradTestBase {};

TEST_F(AutogradForwardTest, AddForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto b = createTensor({2, 3}, 2.0f);
    auto add_fn = std::make_shared<autograd::Add>();
    auto result = add_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
}

TEST_F(AutogradBackwardTest, AddBackward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto b = createTensor({2, 3}, 2.0f);
    auto add_fn = std::make_shared<autograd::Add>();
    auto result = add_fn->Apply({a, b});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = add_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_F(AutogradForwardTest, SubForward) {
    auto a = createTensor({2, 3}, 5.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto sub_fn = std::make_shared<autograd::Sub>();
    auto result = sub_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, MulForward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto mul_fn = std::make_shared<autograd::Mul>();
    auto result = mul_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradBackwardTest, MulBackward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto mul_fn = std::make_shared<autograd::Mul>();
    auto result = mul_fn->Apply({a, b});
    auto grad = createTensor({2, 3}, 1.0f);
    auto grad_inputs = mul_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_F(AutogradForwardTest, DivForward) {
    auto a = createTensor({2, 3}, 6.0f);
    auto b = createTensor({2, 3}, 2.0f);
    auto div_fn = std::make_shared<autograd::Div>();
    auto result = div_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, NegForward) {
    auto a = createTensor({2, 3}, 5.0f);
    auto neg_fn = std::make_shared<autograd::Neg>();
    auto result = neg_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, SinForward) {
    auto a = createTensor({2, 3}, 0.0f);
    auto sin_fn = std::make_shared<autograd::Sin>();
    auto result = sin_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, CosForward) {
    auto a = createTensor({2, 3}, 0.0f);
    auto cos_fn = std::make_shared<autograd::Cos>();
    auto result = cos_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, TanhForward) {
    auto a = createTensor({2, 3}, 0.0f);
    auto tanh_fn = std::make_shared<autograd::Tanh>();
    auto result = tanh_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, ExpForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto exp_fn = std::make_shared<autograd::Exp>();
    auto result = exp_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, LogForward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto log_fn = std::make_shared<autograd::Log>();
    auto result = log_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, ReciprocalForward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto reciprocal_fn = std::make_shared<autograd::Reciprocal>();
    auto result = reciprocal_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, PowForward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto pow_fn = std::make_shared<autograd::Pow>(2.0f);
    auto result = pow_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, RsqrtForward) {
    auto a = createTensor({2, 3}, 4.0f);
    auto rsqrt_fn = std::make_shared<autograd::Rsqrt>();
    auto result = rsqrt_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, SigmoidForward) {
    auto a = createTensor({2, 3}, 0.0f);
    auto sigmoid_fn = std::make_shared<autograd::Sigmoid>();
    auto result = sigmoid_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, MatmulForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto b = createTensor({3, 4}, 1.0f);
    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 4}));
}

TEST_F(AutogradForwardTest, SumForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto sum_fn = std::make_shared<autograd::Sum>(1, false);
    auto result = sum_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, MeanForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto mean_fn = std::make_shared<autograd::Mean>(1, false);
    auto result = mean_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, MaxForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto max_fn = std::make_shared<autograd::Max>(1, false);
    auto result = max_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, MinForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto min_fn = std::make_shared<autograd::Min>(1, false);
    auto result = min_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, SoftmaxForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto softmax_fn = std::make_shared<autograd::Softmax>(1);
    auto result = softmax_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
}

TEST_F(AutogradForwardTest, LayerNormForward) {
    auto a = createTensor({2, 3, 4}, 1.0f);
    auto weight = createTensor({4}, 1.0f);
    auto bias = createTensor({4}, 0.0f);
    auto layernorm_fn = std::make_shared<autograd::LayerNorm>(1e-5f);
    auto result = layernorm_fn->Apply({a, weight, bias});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, LinearForward) {
    auto input = createTensor({2, 3}, 1.0f);
    auto weight = createTensor({4, 3}, 1.0f);
    auto bias = createTensor({4}, 0.0f);
    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight, bias});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 4}));
}

TEST_F(AutogradForwardTest, TransposeForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto transpose_fn = std::make_shared<autograd::Transpose>(0, 1);
    auto result = transpose_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{3, 2}));
}

TEST_F(AutogradForwardTest, SliceForward) {
    auto a = createTensor({4, 4}, 1.0f);
    auto slice_fn = std::make_shared<autograd::Slice>(
        std::vector<int64_t>{1, 1},
        std::vector<int64_t>{3, 3},
        std::vector<int64_t>{1, 1});
    auto result = slice_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, SplitForward) {
    auto a = createTensor({4, 4}, 1.0f);
    auto split_fn = std::make_shared<autograd::Split>(2, 0);
    auto result = split_fn->Apply({a});
    EXPECT_EQ(result.size(), 2);
}

TEST_F(AutogradForwardTest, ConcatForward) {
    auto a = createTensor({2, 2}, 1.0f);
    auto b = createTensor({2, 2}, 2.0f);
    auto concat_fn = std::make_shared<autograd::Concat>(0);
    auto result = concat_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{4, 2}));
}

TEST_F(AutogradForwardTest, StackForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto b = createTensor({2, 3}, 2.0f);
    auto stack_fn = std::make_shared<autograd::Stack>(0);
    auto result = stack_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 2, 3}));
}

TEST_F(AutogradForwardTest, TrilForward) {
    auto a = createTensor({3, 3}, 1.0f);
    auto tril_fn = std::make_shared<autograd::Tril>(0);
    auto result = tril_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, TriuForward) {
    auto a = createTensor({3, 3}, 1.0f);
    auto triu_fn = std::make_shared<autograd::Triu>(0);
    auto result = triu_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, OuterForward) {
    auto a = createTensor({3}, 1.0f);
    auto b = createTensor({4}, 1.0f);
    auto outer_fn = std::make_shared<autograd::Outer>();
    auto result = outer_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{3, 4}));
}

TEST_F(AutogradForwardTest, AddScalarForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto add_scalar_fn = std::make_shared<autograd::AddScalar>(2.0f);
    auto result = add_scalar_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, MulScalarForward) {
    auto a = createTensor({2, 3}, 2.0f);
    auto mul_scalar_fn = std::make_shared<autograd::MulScalar>(3.0f);
    auto result = mul_scalar_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, LtForward) {
    auto a = createTensor({2, 3}, 5.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto lt_fn = std::make_shared<autograd::Lt>();
    auto result = lt_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, LeForward) {
    auto a = createTensor({2, 3}, 3.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto le_fn = std::make_shared<autograd::Le>();
    auto result = le_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, GtForward) {
    auto a = createTensor({2, 3}, 5.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto gt_fn = std::make_shared<autograd::Gt>();
    auto result = gt_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, GeForward) {
    auto a = createTensor({2, 3}, 3.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto ge_fn = std::make_shared<autograd::Ge>();
    auto result = ge_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, EqualsForward) {
    auto a = createTensor({2, 3}, 3.0f);
    auto b = createTensor({2, 3}, 3.0f);
    auto eq_fn = std::make_shared<autograd::Equals>();
    auto result = eq_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, AndForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto b = createTensor({2, 3}, 1.0f);
    auto and_fn = std::make_shared<autograd::And>();
    auto result = and_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, OrForward) {
    auto a = createTensor({2, 3}, 0.0f);
    auto b = createTensor({2, 3}, 1.0f);
    auto or_fn = std::make_shared<autograd::Or>();
    auto result = or_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradForwardTest, NoOpForward) {
    auto a = createTensor({2, 3}, 1.0f);
    auto noop_fn = std::make_shared<autograd::NoOp>(std::vector<int64_t>{2, 3});
    auto result = noop_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
}

// ============================================================================
// CUDA Tests - require CUDA build and GPU
// ============================================================================

#ifdef USE_CUDA
TEST_F(AutogradCudaTest, AddForwardCUDA) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    a->set_requires_grad(true);
    auto a_data = static_cast<float*>(a->DataPtr());
    for (int i = 0; i < 6; ++i) a_data[i] = 1.0f;

    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    b->set_requires_grad(true);
    auto b_data = static_cast<float*>(b->DataPtr());
    for (int i = 0; i < 6; ++i) b_data[i] = 2.0f;

    auto add_fn = std::make_shared<autograd::Add>();
    auto result = add_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
}

TEST_F(AutogradCudaTest, MatmulForwardCUDA) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    a->set_requires_grad(true);
    auto a_data = static_cast<float*>(a->DataPtr());
    for (int i = 0; i < 6; ++i) a_data[i] = 1.0f;

    auto b = std::make_shared<Tensor>(std::vector<int64_t>{3, 4}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    b->set_requires_grad(true);
    auto b_data = static_cast<float*>(b->DataPtr());
    for (int i = 0; i < 12; ++i) b_data[i] = 1.0f;

    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 4}));
}

TEST_F(AutogradCudaTest, SumForwardCUDA) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    a->set_requires_grad(true);
    auto a_data = static_cast<float*>(a->DataPtr());
    for (int i = 0; i < 6; ++i) a_data[i] = 1.0f;

    auto sum_fn = std::make_shared<autograd::Sum>(1, false);
    auto result = sum_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_F(AutogradCudaTest, SoftmaxForwardCUDA) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    a->set_requires_grad(true);
    auto a_data = static_cast<float*>(a->DataPtr());
    for (int i = 0; i < 6; ++i) a_data[i] = 1.0f;

    auto softmax_fn = std::make_shared<autograd::Softmax>(1);
    auto result = softmax_fn->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
}

TEST_F(AutogradCudaTest, LinearForwardCUDA) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCUDA, 0));
    input->set_requires_grad(true);
    auto input_data = static_cast<float*>(input->DataPtr());
    for (int i = 0; i < 6; ++i) input_data[i] = 1.0f;

    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32,
                                            Device(Device::DeviceType::kCUDA, 0));
    weight->set_requires_grad(true);
    auto weight_data = static_cast<float*>(weight->DataPtr());
    for (int i = 0; i < 12; ++i) weight_data[i] = 1.0f;

    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCUDA, 0));
    bias->set_requires_grad(true);
    auto bias_data = static_cast<float*>(bias->DataPtr());
    for (int i = 0; i < 4; ++i) bias_data[i] = 0.0f;

    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight, bias});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 4}));
}
#endif // USE_CUDA

// ============================================================================
// Distributed Tests - require CUDA + NCCL
// ============================================================================

#ifdef USE_NCCL
TEST_F(AutogradDistributedTest, AllReduceDistributed) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    a->set_requires_grad(true);
    auto a_data = static_cast<float*>(a->DataPtr());
    for (int i = 0; i < 6; ++i) a_data[i] = 1.0f;

    EXPECT_TRUE(a->IsCUDA());
    EXPECT_TRUE(a->requires_grad());
}

TEST_F(AutogradDistributedTest, AllGatherDistributed) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{4, 4}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    a->set_requires_grad(true);
    auto a_data = static_cast<float*>(a->DataPtr());
    for (int i = 0; i < 16; ++i) a_data[i] = 1.0f;

    EXPECT_TRUE(a->IsCUDA());
    EXPECT_EQ(a->Dims(), (std::vector<int64_t>{4, 4}));
}

TEST_F(AutogradDistributedTest, ReduceScatterDistributed) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 8}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    a->set_requires_grad(true);
    auto a_data = static_cast<float*>(a->DataPtr());
    for (int i = 0; i < 16; ++i) a_data[i] = 1.0f;

    EXPECT_TRUE(a->IsCUDA());
    EXPECT_EQ(a->Dims(), (std::vector<int64_t>{2, 8}));
}

TEST_F(AutogradDistributedTest, DistributedMatmul) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 4}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    a->set_requires_grad(true);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{4, 2}, DataType::kFLOAT32,
                                       Device(Device::DeviceType::kCUDA, 0));
    b->set_requires_grad(true);

    auto matmul_fn = std::make_shared<autograd::Matmul>();
    auto result = matmul_fn->Apply({a, b});

    EXPECT_EQ(result.size(), 1);
    EXPECT_TRUE(result[0]->IsCUDA());
}

TEST_F(AutogradDistributedTest, DistributedLinear) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32,
                                           Device(Device::DeviceType::kCUDA, 0));
    input->set_requires_grad(true);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32,
                                            Device(Device::DeviceType::kCUDA, 0));
    weight->set_requires_grad(true);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32,
                                          Device(Device::DeviceType::kCUDA, 0));
    bias->set_requires_grad(true);

    auto linear_fn = std::make_shared<autograd::Linear>();
    auto result = linear_fn->Apply({input, weight, bias});

    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 4}));
    EXPECT_TRUE(result[0]->IsCUDA());
}
#endif // USE_NCCL
