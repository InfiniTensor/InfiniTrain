#include <cmath>
#include <limits>
#include <numeric>

#include "infini_train/include/autograd/convolution.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/convolution.h"
#include "infini_train/include/nn/modules/flatten.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/optimizer.h"
#include "tests/common/test_utils.h"

using namespace infini_train;

class CnnTest : public test::InfiniTrainTest {
protected:
    void SetUp() override {
        if (GetDevice().IsCUDA()) {
            REQUIRE_MIN_DEVICES(1);
        }
    }

    std::shared_ptr<Tensor> Make(const std::vector<float> &v, const std::vector<int64_t> &shape, bool grad = true) {
        auto t = std::make_shared<Tensor>(v.data(), shape, DataType::kFLOAT32, GetDevice());
        t->set_requires_grad(grad);
        return t;
    }

    std::vector<float> Values(const std::shared_ptr<Tensor> &t) {
        auto cpu = t->To(Device());
        core::GetDeviceGuardImpl(GetDevice().type())->SynchronizeDevice(GetDevice());
        const auto p = static_cast<const float *>(cpu.DataPtr());
        return {p, p + cpu.NumElements()};
    }

    void Near(const std::shared_ptr<Tensor> &t, const std::vector<float> &expected, float tol = 1e-5f) {
        ASSERT_NE(t, nullptr);
        const auto actual = Values(t);
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i) { EXPECT_NEAR(actual[i], expected[i], tol) << "index " << i; }
    }
};

TEST_P(CnnTest, ConvKnownForwardAndBackward) {
    auto x = Make({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 1, 3, 3});
    auto w = Make({1, 0, 0, -1}, {1, 1, 2, 2});
    auto b = Make({2}, {1});
    auto y = nn::function::Conv2d(x, w, b);
    EXPECT_EQ(y->Dims(), (std::vector<int64_t>{1, 1, 2, 2}));
    Near(y, {-2, -2, -2, -2});
    y->Backward(Make({1, 2, 3, 4}, y->Dims(), false));
    Near(x->grad(), {1, 2, 0, 3, 3, -2, 0, -3, -4});
    Near(w->grad(), {37, 47, 67, 77});
    Near(b->grad(), {10});
}

// Independent double-precision reference, including virtual zero padding.
// Finite differences below validate all three analytical derivatives.
namespace {
std::vector<double> Reference(const std::vector<double> &x, const std::vector<double> &w, const std::vector<double> &b,
                              int stride, int pad) {
    constexpr int n = 2, ci = 2, h = 4, width = 5, co = 2, kh = 2, kw = 3;
    const int oh = (h + 2 * pad - kh) / stride + 1;
    const int ow = (width + 2 * pad - kw) / stride + 1;
    std::vector<double> y(n * co * oh * ow);
    for (int batch = 0; batch < n; ++batch) {
        for (int oc = 0; oc < co; ++oc) {
            for (int row = 0; row < oh; ++row) {
                for (int col = 0; col < ow; ++col) {
                    double sum = b.empty() ? 0.0 : b[oc];
                    for (int ic = 0; ic < ci; ++ic) {
                        for (int r = 0; r < kh; ++r) {
                            for (int c = 0; c < kw; ++c) {
                                int ir = row * stride + r - pad, jc = col * stride + c - pad;
                                if (ir >= 0 && ir < h && jc >= 0 && jc < width) {
                                    sum += x[((batch * ci + ic) * h + ir) * width + jc]
                                         * w[((oc * ci + ic) * kh + r) * kw + c];
                                }
                            }
                        }
                    }
                    y[((batch * co + oc) * oh + row) * ow + col] = sum;
                }
            }
        }
    }
    return y;
}
} // namespace

TEST_P(CnnTest, ConvMultiChannelFiniteDifferences) {
    for (const auto [stride, pad] : std::vector<std::pair<int, int>>{{1, 0}, {2, 1}}) {
        for (bool bias : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "stride=" << stride << " padding=" << pad << " bias=" << bias);
            std::vector<double> xv(80), wv(24), bv(bias ? 2 : 0);
            for (int i = 0; i < 80; ++i) { xv[i] = (i % 13 - 6) * 0.125; }
            for (int i = 0; i < 24; ++i) { wv[i] = (i % 7 - 3) * 0.0625; }
            if (bias) {
                bv = {0.125, -0.25};
            }
            auto x = Make(std::vector<float>(xv.begin(), xv.end()), {2, 2, 4, 5});
            auto w = Make(std::vector<float>(wv.begin(), wv.end()), {2, 2, 2, 3});
            auto b = bias ? Make(std::vector<float>(bv.begin(), bv.end()), {2}) : nullptr;
            auto y = nn::function::Conv2d(x, w, b, stride, pad);
            auto expected = Reference(xv, wv, bv, stride, pad);
            Near(y, std::vector<float>(expected.begin(), expected.end()));
            std::vector<float> upstream(expected.size());
            for (size_t i = 0; i < upstream.size(); ++i) { upstream[i] = (static_cast<int>(i % 9) - 4) * 0.125f; }
            y->Backward(Make(upstream, y->Dims(), false));
            auto objective = [&]() {
                const auto out = Reference(xv, wv, bv, stride, pad);
                return std::inner_product(out.begin(), out.end(), upstream.begin(), 0.0);
            };
            auto check = [&](std::vector<double> &v, const std::shared_ptr<Tensor> &grad) {
                const auto actual = Values(grad);
                ASSERT_EQ(v.size(), actual.size());
                for (size_t i = 0; i < v.size(); ++i) {
                    const double old = v[i], eps = 1e-4;
                    v[i] = old + eps;
                    const double plus = objective();
                    v[i] = old - eps;
                    const double minus = objective();
                    v[i] = old;
                    EXPECT_NEAR(actual[i], (plus - minus) / (2 * eps), 2e-5) << "index " << i;
                }
            };
            check(xv, x->grad());
            check(wv, w->grad());
            if (bias) {
                check(bv, b->grad());
            }
        }
    }
}

TEST_P(CnnTest, ConvSelectiveGradientsAndNoBias) {
    for (int mask = 1; mask < 8; ++mask) {
        auto x = Make(std::vector<float>(9, 1), {1, 1, 3, 3}, mask & 1);
        auto w = Make(std::vector<float>(4, 1), {1, 1, 2, 2}, mask & 2);
        auto b = Make({0}, {1}, mask & 4);
        nn::function::Conv2d(x, w, b)->Backward(Make(std::vector<float>(4, 1), {1, 1, 2, 2}, false));
        EXPECT_EQ(x->grad() != nullptr, static_cast<bool>(mask & 1));
        EXPECT_EQ(w->grad() != nullptr, static_cast<bool>(mask & 2));
        EXPECT_EQ(b->grad() != nullptr, static_cast<bool>(mask & 4));
    }
    auto conv = std::make_shared<nn::Conv2d>(1, 2, 1, 1, 0, false, GetDevice());
    EXPECT_EQ(conv->Parameters().size(), 1);
    EXPECT_FALSE(conv->has_bias());
    conv->parameter("weight")->Fill(1.0f);
    auto y = (*conv)({Make({1, 2, 3, 4}, {1, 1, 2, 2})})[0];
    Near(y, {1, 2, 3, 4, 1, 2, 3, 4});
    y->Backward(Make(std::vector<float>(8, 1), y->Dims(), false));
    Near(conv->parameter("weight")->grad(), {10, 10});
}

TEST_P(CnnTest, ReLUForwardBackwardAndZero) {
    auto x = Make({-2, -1, 0, 1, 2, 3}, {2, 3});
    nn::ReLU relu;
    auto y = relu({x})[0];
    Near(y, {0, 0, 0, 1, 2, 3});
    Near(x, {-2, -1, 0, 1, 2, 3});
    y->Backward(Make({1, 2, 3, 4, 5, 6}, {2, 3}, false));
    Near(x->grad(), {0, 0, 0, 4, 5, 6});
    auto special
        = nn::function::ReLU(Make({-std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                                   std::numeric_limits<float>::quiet_NaN()},
                                  {3}, false));
    auto values = Values(special);
    EXPECT_EQ(values[0], 0);
    EXPECT_TRUE(std::isinf(values[1]));
    EXPECT_TRUE(std::isnan(values[2]));
}

TEST_P(CnnTest, FlattenShapeValuesAndGradient) {
    std::vector<float> values(24);
    std::iota(values.begin(), values.end(), 0.0f);
    for (const auto [start, end] : std::vector<std::pair<int, int>>{{1, -1}, {-3, -1}, {1, 2}, {0, -1}}) {
        auto x = Make(values, {2, 2, 2, 3});
        nn::Flatten flatten(start, end);
        auto y = flatten({x})[0];
        const std::vector<int64_t> expected = start == 0 ? std::vector<int64_t>{24}
                                            : end == 2   ? std::vector<int64_t>{2, 4, 3}
                                                         : std::vector<int64_t>{2, 12};
        EXPECT_EQ(y->Dims(), expected);
        Near(y, values);
        y->Backward(Make(values, y->Dims(), false));
        ASSERT_NE(x->grad(), nullptr);
        EXPECT_EQ(x->grad()->Dims(), x->Dims());
        Near(x->grad(), values);
    }
}

TEST_P(CnnTest, SequentialLossBackwardAndOptimizer) {
    auto conv = std::make_shared<nn::Conv2d>(1, 2, 2);
    auto conv2 = std::make_shared<nn::Conv2d>(2, 2, 2);
    auto linear = std::make_shared<nn::Linear>(8, 2);
    auto net = std::make_shared<nn::Sequential>(std::vector<std::shared_ptr<nn::Module>>{
        conv, std::make_shared<nn::ReLU>(), conv2, std::make_shared<nn::ReLU>(), std::make_shared<nn::Flatten>(),
        linear});
    net->To(GetDevice());
    conv->parameter("weight")->Fill(0.1f);
    conv->parameter("bias")->Fill(0.1f);
    conv2->parameter("weight")->Fill(0.1f);
    conv2->parameter("bias")->Fill(0.1f);
    std::vector<float> weights(16);
    for (int i = 0; i < 16; ++i) { weights[i] = (i < 8 ? 1 : -1) * 0.05f; }
    linear->parameter("weight")->CopyFrom(Make(weights, {2, 8}, false));
    linear->parameter("bias")->Fill(0.0f);
    EXPECT_EQ(net->NamedParameters().size(), 6);
    auto x = Make(std::vector<float>(32, 0.5f), {2, 1, 4, 4}, false);
    auto labels = Make({1, 1}, {2}, false)->To(DataType::kINT64);
    auto target = std::make_shared<Tensor>(labels);
    nn::CrossEntropyLoss loss;
    optimizers::SGD optimizer(net->Parameters(), 0.02f);
    float initial = 0, final = 0;
    for (int step = 0; step < 4; ++step) {
        optimizer.ZeroGrad();
        auto l = loss({(*net)({x})[0], target})[0];
        final = Values(l)[0];
        if (!step) {
            initial = final;
        }
        l->Backward();
        std::vector<std::vector<float>> expected;
        for (const auto &p : net->Parameters()) {
            ASSERT_NE(p->grad(), nullptr);
            auto before = Values(p), grad = Values(p->grad());
            bool nonzero = false;
            for (size_t i = 0; i < before.size(); ++i) {
                EXPECT_TRUE(std::isfinite(grad[i]));
                nonzero |= grad[i] != 0;
                before[i] -= 0.02f * grad[i];
            }
            EXPECT_TRUE(nonzero);
            expected.push_back(before);
        }
        optimizer.Step();
        const auto params = net->Parameters();
        for (size_t i = 0; i < params.size(); ++i) { Near(params[i], expected[i]); }
    }
    EXPECT_LT(final, initial);
    optimizer.ZeroGrad();
    for (const auto &p : net->Parameters()) { EXPECT_EQ(p->grad(), nullptr); }
}

TEST_P(CnnTest, NoGradInferenceAndAccumulation) {
    auto x = Make(std::vector<float>(9, 1), {1, 1, 3, 3});
    auto w = Make(std::vector<float>(4, 1), {1, 1, 2, 2});
    {
        autograd::NoGradGuard guard;
        auto y = nn::function::ReLU(nn::function::Conv2d(x, w));
        EXPECT_FALSE(y->requires_grad());
        EXPECT_EQ(y->grad_fn(), nullptr);
    }
    for (int i = 0; i < 2; ++i) {
        nn::function::Conv2d(x, w)->Backward(Make(std::vector<float>(4, 1), {1, 1, 2, 2}, false));
    }
    Near(w->grad(), {8, 8, 8, 8});
}

TEST_P(CnnTest, InvalidArgumentsFailClearly) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    // CUDA death tests must not fork after a CUDA context has been initialized.
    if (GetDevice().IsCUDA()) {
        GTEST_SKIP() << "Argument validation is shared with CPU";
    }
    auto x = Make(std::vector<float>(9, 1), {1, 1, 3, 3});
    auto w = Make(std::vector<float>(4, 1), {1, 1, 2, 2});
    EXPECT_DEATH(nn::function::Conv2d(x, w, nullptr, 0), "stride_");
    EXPECT_DEATH(nn::function::Conv2d(x, w, nullptr, 1, -1), "padding_");
    EXPECT_DEATH(nn::function::Conv2d(x->View({1, 9}), w), "NCHW");
    EXPECT_DEATH(nn::function::Conv2d(x, Make(std::vector<float>(8, 1), {1, 2, 2, 2})), "Check failed");
    EXPECT_DEATH(nn::function::Conv2d(x, Make(std::vector<float>(16, 1), {1, 1, 4, 4})), "Check failed");
    EXPECT_DEATH(nn::function::Conv2d(x, w, Make({1, 2}, {2})), "Check failed");
    EXPECT_DEATH(nn::function::Conv2d(std::make_shared<Tensor>(x->To(DataType::kFLOAT64)), w), "FP32");
    EXPECT_DEATH(nn::Conv2d(1, 1, 0), "kernel_size");
    EXPECT_DEATH(nn::Flatten(1, 4)({x}), "Check failed");
    EXPECT_DEATH(nn::Flatten(-5)({x}), "Check failed");
}

INFINI_TRAIN_REGISTER_TEST(CnnTest);
