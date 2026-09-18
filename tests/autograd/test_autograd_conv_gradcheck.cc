// Finite-difference gradcheck for Conv2d autograd (CPU).
//
// Config: N=1, Cin=2, Cout=2, H=4, W=5, K=3, stride=2, padding=1.
// Compares analytic Backward() grads (dInput/dWeight/dBias) against central
// differences of loss = dot(forward, grad_output). Self-contained: no torch
// dependency. Threshold: relative err < 1e-3 per element.
#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/conv.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {
// N=1, Cin=2, Cout=2, H=4, W=5, K=3, stride=2, padding=1 -> out (1,2,2,3)
constexpr int64_t kStride = 2, kPadding = 1;
const std::vector<int64_t> kInputDims = {1, 2, 4, 5};
const std::vector<int64_t> kWeightDims = {2, 2, 3, 3};
const std::vector<int64_t> kBiasDims = {2};
const std::vector<int64_t> kOutDims = {1, 2, 2, 3};
constexpr float kEps = 1e-2f;
constexpr float kRelTol = 1e-3f;

// Deterministic pseudo-random values in roughly [-0.5, 0.5]; distinct stream per salt.
float Pattern(size_t idx, int salt) {
    int v = static_cast<int>((idx * 37u + static_cast<unsigned>(salt) * 17u) % 11u) - 5; // [-5, 5]
    return static_cast<float>(v) * 0.1f;
}
} // namespace

class AutogradConvGradcheckTest : public infini_train::test::InfiniTrainTest {};

TEST_P(AutogradConvGradcheckTest, Conv2dCentralDifference) {
    ONLY_CPU();
    const Device device = GetDevice();

    const size_t nx = 40, nw = 36, nb = 2, ny = 12;
    ASSERT_EQ(nx, static_cast<size_t>(1 * 2 * 4 * 5));
    ASSERT_EQ(nw, static_cast<size_t>(2 * 2 * 3 * 3));
    ASSERT_EQ(ny, static_cast<size_t>(1 * 2 * 2 * 3));

    std::vector<float> x0(nx), w0(nw), b0(nb), go(ny);
    for (size_t i = 0; i < nx; ++i) { x0[i] = Pattern(i, 1); }
    for (size_t i = 0; i < nw; ++i) { w0[i] = Pattern(i, 2); }
    for (size_t i = 0; i < nb; ++i) { b0[i] = Pattern(i, 3); }
    for (size_t i = 0; i < ny; ++i) { go[i] = 0.5f + Pattern(i, 4); }

    // Scalar loss for a given parameter setting: dot(forward, go).
    // NOTE: the loss is exactly linear in each single perturbed parameter
    // (conv is multilinear), so central differences have no truncation error
    // and a larger step (1e-2) minimises fp32 cancellation in the difference.
    auto loss_fn = [&](const std::vector<float> &x, const std::vector<float> &w,
                       const std::vector<float> &b) {
        auto input = std::make_shared<Tensor>(x.data(), kInputDims, DataType::kFLOAT32, device);
        auto weight = std::make_shared<Tensor>(w.data(), kWeightDims, DataType::kFLOAT32, device);
        auto bias = std::make_shared<Tensor>(b.data(), kBiasDims, DataType::kFLOAT32, device);
        auto fn = std::make_shared<autograd::Conv2d>(kStride, kPadding);
        auto out = fn->Apply({input, weight, bias});
        EXPECT_EQ(out.size(), 1);
        auto out_cpu = out[0]->To(Device());
        const auto *p = static_cast<const float *>(out_cpu.DataPtr());
        double acc = 0.0;
        for (size_t i = 0; i < ny; ++i) { acc += static_cast<double>(p[i]) * static_cast<double>(go[i]); }
        return acc;
    };

    // Analytic grads via autograd Backward.
    auto input = std::make_shared<Tensor>(x0.data(), kInputDims, DataType::kFLOAT32, device)->RequiresGrad();
    auto weight = std::make_shared<Tensor>(w0.data(), kWeightDims, DataType::kFLOAT32, device)->RequiresGrad();
    auto bias = std::make_shared<Tensor>(b0.data(), kBiasDims, DataType::kFLOAT32, device)->RequiresGrad();
    auto fn = std::make_shared<autograd::Conv2d>(kStride, kPadding);
    auto out = fn->Apply({input, weight, bias});
    ASSERT_EQ(out.size(), 1);
    EXPECT_EQ(out[0]->Dims(), kOutDims);
    auto grad_out = std::make_shared<Tensor>(go.data(), kOutDims, DataType::kFLOAT32, device);
    auto grads = fn->Backward({grad_out});
    ASSERT_EQ(grads.size(), 3);
    ASSERT_NE(grads[0], nullptr);
    ASSERT_NE(grads[1], nullptr);
    ASSERT_NE(grads[2], nullptr);

    const std::vector<const std::shared_ptr<Tensor> *> analytic = {&grads[0], &grads[1], &grads[2]};
    const std::vector<size_t> sizes = {nx, nw, nb};
    std::vector<std::vector<float>> bases = {x0, w0, b0};

    for (int p = 0; p < 3; ++p) {
        auto g_cpu = (*analytic[p])->To(Device());
        const auto *g = static_cast<const float *>(g_cpu.DataPtr());
        ASSERT_EQ(g_cpu.NumElements(), sizes[p]);
        double max_rel = 0.0;
        for (size_t i = 0; i < sizes[p]; ++i) {
            auto plus = bases[p], minus = bases[p];
            plus[i] += kEps;
            minus[i] -= kEps;
            // Select which parameter vector is perturbed.
            const std::vector<float> &xp = (p == 0 ? plus : bases[0]);
            const std::vector<float> &xm = (p == 0 ? minus : bases[0]);
            const std::vector<float> &wp = (p == 1 ? plus : bases[1]);
            const std::vector<float> &wm = (p == 1 ? minus : bases[1]);
            const std::vector<float> &bp = (p == 2 ? plus : bases[2]);
            const std::vector<float> &bm = (p == 2 ? minus : bases[2]);
            const double numeric = (loss_fn(xp, wp, bp) - loss_fn(xm, wm, bm)) / (2.0 * kEps);
            const double a = g[i];
            const double denom = std::max({std::fabs(a), std::fabs(numeric), 1e-8});
            double rel;
            if (std::max(std::fabs(a), std::fabs(numeric)) < 1e-5) {
                // Both ~0: fall back to absolute error so tiny noise cannot blow up the ratio.
                rel = std::fabs(a - numeric) / 1e-5;
            } else {
                rel = std::fabs(a - numeric) / denom;
            }
            max_rel = std::max(max_rel, rel);
            EXPECT_LT(rel, kRelTol) << "param " << p << " index " << i << " analytic=" << a
                                    << " numeric=" << numeric;
        }
        EXPECT_LT(max_rel, kRelTol) << "param " << p << " max relative error too large";
    }
}

INFINI_TRAIN_REGISTER_TEST(AutogradConvGradcheckTest);
