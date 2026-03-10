// Test flash attention integration.
//
// Tests:
//   1. Forward correctness: compare with naive scaled dot-product attention.
//   2. Backward correctness: verify gradients are non-null and have correct shapes.
//   3. Causal mask: verify causal attention produces different results than non-causal.
//
// Build (example):
//   cmake -DUSE_CUDA=ON ... && cmake --build . --target test_flash_attention
// Run:
//   ./test_flash_attention

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/flash_attention.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

// ============================================================================
// Helper utilities
// ============================================================================

// Fill a CPU float32 tensor with random values in [lo, hi].
static void FillRandom(Tensor &t, float lo = -0.5f, float hi = 0.5f, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    float *ptr = static_cast<float *>(t.DataPtr());
    for (size_t i = 0; i < t.NumElements(); i++) {
        ptr[i] = dist(rng);
    }
}

// Copy float values from a CPU tensor to a std::vector<float>.
static std::vector<float> ToVector(const Tensor &t) {
    const float *ptr = static_cast<const float *>(t.DataPtr());
    return std::vector<float>(ptr, ptr + t.NumElements());
}

// Create a CUDA bfloat16 tensor from a CPU float32 tensor.
static std::shared_ptr<Tensor> ToCudaBf16(const std::shared_ptr<Tensor> &cpu_f32) {
    auto cpu_bf16 = std::make_shared<Tensor>(cpu_f32->To(DataType::kBFLOAT16));
    auto cuda_bf16 = std::make_shared<Tensor>(cpu_bf16->To(Device(Device::DeviceType::kCUDA, 0)));
    return cuda_bf16;
}

// Convert CUDA bfloat16 tensor to CPU float32.
static std::shared_ptr<Tensor> ToCpuF32(const std::shared_ptr<Tensor> &cuda_bf16) {
    auto cpu_bf16 = std::make_shared<Tensor>(cuda_bf16->To(Device(Device::DeviceType::kCPU, 0)));
    auto cpu_f32 = std::make_shared<Tensor>(cpu_bf16->To(DataType::kFLOAT32));
    return cpu_f32;
}

// Naive scaled dot-product attention (CPU, float32).
// Q, K, V: [bs, num_heads, seq_len, head_dim] flattened as row-major arrays.
// Output:  O [bs, num_heads, q_len, head_dim]
static std::vector<float> NaiveAttention(const std::vector<float> &Q_data,
                                          const std::vector<float> &K_data,
                                          const std::vector<float> &V_data,
                                          int bs, int q_head, int kv_head,
                                          int q_len, int kv_len, int head_dim,
                                          bool is_causal) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int q_kv_ratio = q_head / kv_head;
    std::vector<float> O(bs * q_head * q_len * head_dim, 0.0f);

    for (int b = 0; b < bs; b++) {
        for (int h = 0; h < q_head; h++) {
            int kv_h = h / q_kv_ratio;
            for (int qi = 0; qi < q_len; qi++) {
                // Compute attention scores: S[qi, ki] = sum_d Q[qi,d] * K[ki,d] * scale
                std::vector<float> scores(kv_len);
                for (int ki = 0; ki < kv_len; ki++) {
                    float s = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        float q = Q_data[((b * q_head + h) * q_len + qi) * head_dim + d];
                        float k = K_data[((b * kv_head + kv_h) * kv_len + ki) * head_dim + d];
                        s += q * k;
                    }
                    s *= scale;
                    // Causal mask: if ki > qi, mask to -inf
                    if (is_causal && ki > qi) {
                        s = -1e9f;
                    }
                    scores[ki] = s;
                }

                // Softmax
                float max_s = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (int ki = 0; ki < kv_len; ki++) {
                    scores[ki] = std::exp(scores[ki] - max_s);
                    sum_exp += scores[ki];
                }
                for (int ki = 0; ki < kv_len; ki++) {
                    scores[ki] /= sum_exp;
                }

                // Output: O[qi, d] = sum_ki scores[ki] * V[ki, d]
                for (int d = 0; d < head_dim; d++) {
                    float out = 0.0f;
                    for (int ki = 0; ki < kv_len; ki++) {
                        float v = V_data[((b * kv_head + kv_h) * kv_len + ki) * head_dim + d];
                        out += scores[ki] * v;
                    }
                    O[((b * q_head + h) * q_len + qi) * head_dim + d] = out;
                }
            }
        }
    }
    return O;
}

// Compute max absolute error between two vectors.
static float MaxAbsError(const std::vector<float> &a, const std::vector<float> &b) {
    assert(a.size() == b.size());
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        err = std::max(err, std::abs(a[i] - b[i]));
    }
    return err;
}

// ============================================================================
// Test 1: Forward correctness (non-causal)
// ============================================================================
void TestForwardNonCausal() {
    std::cout << "\n=== Test 1: Forward correctness (non-causal) ===" << std::endl;

    const int bs = 1, q_head = 2, kv_head = 2;
    const int q_len = 64, kv_len = 64, head_dim = 64;

    // Create CPU float32 tensors
    auto Q_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, q_head, q_len, head_dim},
                                          DataType::kFLOAT32);
    auto K_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, kv_head, kv_len, head_dim},
                                          DataType::kFLOAT32);
    auto V_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, kv_head, kv_len, head_dim},
                                          DataType::kFLOAT32);
    FillRandom(*Q_cpu, -0.3f, 0.3f, 1);
    FillRandom(*K_cpu, -0.3f, 0.3f, 2);
    FillRandom(*V_cpu, -0.3f, 0.3f, 3);

    // Reference output (float32)
    auto Q_vec = ToVector(*Q_cpu);
    auto K_vec = ToVector(*K_cpu);
    auto V_vec = ToVector(*V_cpu);
    auto ref_O = NaiveAttention(Q_vec, K_vec, V_vec, bs, q_head, kv_head, q_len, kv_len, head_dim, false);

    // Flash attention on CUDA bfloat16
    auto Q_cuda = ToCudaBf16(Q_cpu);
    auto K_cuda = ToCudaBf16(K_cpu);
    auto V_cuda = ToCudaBf16(V_cpu);

    auto flash_outputs = std::make_shared<autograd::FlashAttention>(/*is_causal=*/false)
                             ->Apply({Q_cuda, K_cuda, V_cuda});
    auto O_cuda = flash_outputs[0];

    // Convert output to CPU float32 for comparison
    auto O_cpu = ToCpuF32(O_cuda);
    auto flash_O = ToVector(*O_cpu);

    float err = MaxAbsError(ref_O, flash_O);
    // bfloat16 has ~7 bits of mantissa, so relative error ~1%, absolute error
    // should be well within 0.1 for values in [-1, 1].
    const float kTol = 0.1f;
    std::printf("  Max abs error (flash vs naive): %.6f  (tolerance %.3f)  %s\n",
                err, kTol, err < kTol ? "PASS" : "FAIL");
    assert(err < kTol);
}

// ============================================================================
// Test 2: Forward correctness (causal)
// ============================================================================
void TestForwardCausal() {
    std::cout << "\n=== Test 2: Forward correctness (causal) ===" << std::endl;

    const int bs = 1, q_head = 2, kv_head = 2;
    const int q_len = 64, kv_len = 64, head_dim = 64;

    auto Q_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, q_head, q_len, head_dim},
                                          DataType::kFLOAT32);
    auto K_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, kv_head, kv_len, head_dim},
                                          DataType::kFLOAT32);
    auto V_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, kv_head, kv_len, head_dim},
                                          DataType::kFLOAT32);
    FillRandom(*Q_cpu, -0.3f, 0.3f, 10);
    FillRandom(*K_cpu, -0.3f, 0.3f, 20);
    FillRandom(*V_cpu, -0.3f, 0.3f, 30);

    auto Q_vec = ToVector(*Q_cpu);
    auto K_vec = ToVector(*K_cpu);
    auto V_vec = ToVector(*V_cpu);
    auto ref_O = NaiveAttention(Q_vec, K_vec, V_vec, bs, q_head, kv_head, q_len, kv_len, head_dim, true);

    auto Q_cuda = ToCudaBf16(Q_cpu);
    auto K_cuda = ToCudaBf16(K_cpu);
    auto V_cuda = ToCudaBf16(V_cpu);

    auto flash_outputs = std::make_shared<autograd::FlashAttention>(/*is_causal=*/true)
                             ->Apply({Q_cuda, K_cuda, V_cuda});
    auto O_cuda = flash_outputs[0];

    auto O_cpu = ToCpuF32(O_cuda);
    auto flash_O = ToVector(*O_cpu);

    float err = MaxAbsError(ref_O, flash_O);
    const float kTol = 0.1f;
    std::printf("  Max abs error (flash causal vs naive causal): %.6f  (tolerance %.3f)  %s\n",
                err, kTol, err < kTol ? "PASS" : "FAIL");
    assert(err < kTol);
}

// ============================================================================
// Test 3: Backward (gradient shapes and no crashes)
// ============================================================================
void TestBackward() {
    std::cout << "\n=== Test 3: Backward (gradient shapes) ===" << std::endl;

    const int bs = 1, q_head = 2, kv_head = 2;
    const int q_len = 64, kv_len = 64, head_dim = 64;

    auto Q_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, q_head, q_len, head_dim},
                                          DataType::kFLOAT32);
    auto K_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, kv_head, kv_len, head_dim},
                                          DataType::kFLOAT32);
    auto V_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, kv_head, kv_len, head_dim},
                                          DataType::kFLOAT32);
    FillRandom(*Q_cpu, -0.3f, 0.3f, 100);
    FillRandom(*K_cpu, -0.3f, 0.3f, 200);
    FillRandom(*V_cpu, -0.3f, 0.3f, 300);

    auto Q_cuda = ToCudaBf16(Q_cpu);
    auto K_cuda = ToCudaBf16(K_cpu);
    auto V_cuda = ToCudaBf16(V_cpu);
    Q_cuda->set_requires_grad(true);
    K_cuda->set_requires_grad(true);
    V_cuda->set_requires_grad(true);

    auto flash_outputs = std::make_shared<autograd::FlashAttention>(/*is_causal=*/false)
                             ->Apply({Q_cuda, K_cuda, V_cuda});
    auto O = flash_outputs[0];

    // Backward with all-ones gradient (match O's shape and dtype)
    const size_t n = O->NumElements();
    std::vector<float> ones_data(n, 1.0f);
    auto grad_cpu = std::make_shared<Tensor>(ones_data.data(), O->Dims(), DataType::kFLOAT32);
    auto grad_O = ToCudaBf16(grad_cpu);
    O->Backward(grad_O);

    assert(Q_cuda->grad() != nullptr);
    assert(K_cuda->grad() != nullptr);
    assert(V_cuda->grad() != nullptr);

    const auto &dQ_dims = Q_cuda->grad()->Dims();
    const auto &dK_dims = K_cuda->grad()->Dims();
    const auto &dV_dims = V_cuda->grad()->Dims();

    assert(dQ_dims == Q_cuda->Dims());
    assert(dK_dims == K_cuda->Dims());
    assert(dV_dims == V_cuda->Dims());

    std::printf("  dQ shape: [%lld, %lld, %lld, %lld]  PASS\n",
                dQ_dims[0], dQ_dims[1], dQ_dims[2], dQ_dims[3]);
    std::printf("  dK shape: [%lld, %lld, %lld, %lld]  PASS\n",
                dK_dims[0], dK_dims[1], dK_dims[2], dK_dims[3]);
    std::printf("  dV shape: [%lld, %lld, %lld, %lld]  PASS\n",
                dV_dims[0], dV_dims[1], dV_dims[2], dV_dims[3]);
}

// ============================================================================
// Test 4: Backward gradient accuracy (numerical differentiation)
// ============================================================================
void TestBackwardAccuracy() {
    std::cout << "\n=== Test 4: Backward gradient accuracy (numerical diff) ===" << std::endl;

    const int bs = 1, q_head = 2, kv_head = 2;
    const int q_len = 64, kv_len = 64, head_dim = 64;
    const float eps = 1e-2f;  // finite difference step (bf16 limited precision)

    // We only check a small subset of gradient entries to keep the test fast.
    const int num_check = 8;

    auto Q_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, q_head, q_len, head_dim},
                                          DataType::kFLOAT32);
    auto K_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, kv_head, kv_len, head_dim},
                                          DataType::kFLOAT32);
    auto V_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, kv_head, kv_len, head_dim},
                                          DataType::kFLOAT32);
    FillRandom(*Q_cpu, -0.2f, 0.2f, 42);
    FillRandom(*K_cpu, -0.2f, 0.2f, 43);
    FillRandom(*V_cpu, -0.2f, 0.2f, 44);

    // Compute analytical gradient via backward
    auto Q_cuda = ToCudaBf16(Q_cpu);
    auto K_cuda = ToCudaBf16(K_cpu);
    auto V_cuda = ToCudaBf16(V_cpu);
    Q_cuda->set_requires_grad(true);
    K_cuda->set_requires_grad(true);
    V_cuda->set_requires_grad(true);

    auto O = std::make_shared<autograd::FlashAttention>(/*is_causal=*/false)
                 ->Apply({Q_cuda, K_cuda, V_cuda})[0];
    // Use all-ones output gradient
    std::vector<float> ones(O->NumElements(), 1.0f);
    auto grad_cpu_f32 = std::make_shared<Tensor>(ones.data(), O->Dims(), DataType::kFLOAT32);
    O->Backward(ToCudaBf16(grad_cpu_f32));

    auto dQ_cpu = ToCpuF32(Q_cuda->grad());
    auto dV_cpu = ToCpuF32(V_cuda->grad());

    auto dQ_analytic = ToVector(*dQ_cpu);
    auto dV_analytic = ToVector(*dV_cpu);

    // Numerical gradient for Q (check first num_check elements)
    float Q_base_data[bs * q_head * q_len * head_dim];
    std::copy(static_cast<float *>(Q_cpu->DataPtr()),
              static_cast<float *>(Q_cpu->DataPtr()) + Q_cpu->NumElements(),
              Q_base_data);

    std::printf("  Checking %d dQ entries (numerical vs analytical):\n", num_check);
    float max_dQ_err = 0.0f;
    for (int idx = 0; idx < num_check; idx++) {
        // f(Q + eps*e_i)
        float saved = Q_base_data[idx];
        Q_base_data[idx] = saved + eps;
        auto Q_plus = std::make_shared<Tensor>(Q_base_data,
                                               std::vector<int64_t>{bs, q_head, q_len, head_dim},
                                               DataType::kFLOAT32);
        auto Q_plus_cuda = ToCudaBf16(Q_plus);
        auto K2_cuda = ToCudaBf16(K_cpu);
        auto V2_cuda = ToCudaBf16(V_cpu);
        auto O_plus = std::make_shared<autograd::FlashAttention>(false)->Apply({Q_plus_cuda, K2_cuda, V2_cuda})[0];
        auto O_plus_cpu = ToCpuF32(O_plus);
        float sum_plus = 0.0f;
        for (float v : ToVector(*O_plus_cpu)) sum_plus += v;

        // f(Q - eps*e_i)
        Q_base_data[idx] = saved - eps;
        auto Q_minus = std::make_shared<Tensor>(Q_base_data,
                                                std::vector<int64_t>{bs, q_head, q_len, head_dim},
                                                DataType::kFLOAT32);
        auto Q_minus_cuda = ToCudaBf16(Q_minus);
        auto K3_cuda = ToCudaBf16(K_cpu);
        auto V3_cuda = ToCudaBf16(V_cpu);
        auto O_minus = std::make_shared<autograd::FlashAttention>(false)->Apply({Q_minus_cuda, K3_cuda, V3_cuda})[0];
        auto O_minus_cpu = ToCpuF32(O_minus);
        float sum_minus = 0.0f;
        for (float v : ToVector(*O_minus_cpu)) sum_minus += v;

        Q_base_data[idx] = saved;

        float num_grad = (sum_plus - sum_minus) / (2.0f * eps);
        float ana_grad = dQ_analytic[idx];
        float err = std::abs(num_grad - ana_grad);
        max_dQ_err = std::max(max_dQ_err, err);
    }

    // bfloat16 precision limits gradient accuracy; use a generous tolerance.
    const float kGradTol = 0.15f;
    std::printf("  Max dQ error (numerical vs analytical): %.6f  (tolerance %.3f)  %s\n",
                max_dQ_err, kGradTol, max_dQ_err < kGradTol ? "PASS" : "FAIL");
    assert(max_dQ_err < kGradTol);
}

// ============================================================================
// Test 5: Causal backward gradient accuracy (longer sequence)
// ============================================================================
void TestCausalBackwardAccuracy() {
    std::cout << "\n=== Test 5: Causal backward gradient accuracy (q_len=128) ===" << std::endl;

    const int bs = 2, q_head = 4, kv_head = 4;
    const int q_len = 128, kv_len = 128, head_dim = 64;
    const float eps = 1e-2f;
    const int num_check = 8;

    auto Q_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, q_head, q_len, head_dim},
                                          DataType::kFLOAT32);
    auto K_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, kv_head, kv_len, head_dim},
                                          DataType::kFLOAT32);
    auto V_cpu = std::make_shared<Tensor>(std::vector<int64_t>{bs, kv_head, kv_len, head_dim},
                                          DataType::kFLOAT32);
    FillRandom(*Q_cpu, -0.2f, 0.2f, 51);
    FillRandom(*K_cpu, -0.2f, 0.2f, 52);
    FillRandom(*V_cpu, -0.2f, 0.2f, 53);

    auto Q_cuda = ToCudaBf16(Q_cpu);
    auto K_cuda = ToCudaBf16(K_cpu);
    auto V_cuda = ToCudaBf16(V_cpu);
    Q_cuda->set_requires_grad(true);
    K_cuda->set_requires_grad(true);
    V_cuda->set_requires_grad(true);

    auto O = std::make_shared<autograd::FlashAttention>(/*is_causal=*/true)
                 ->Apply({Q_cuda, K_cuda, V_cuda})[0];
    std::vector<float> ones(O->NumElements(), 1.0f);
    auto grad_cpu_f32 = std::make_shared<Tensor>(ones.data(), O->Dims(), DataType::kFLOAT32);
    O->Backward(ToCudaBf16(grad_cpu_f32));

    // Check for NaN in gradients
    auto dQ_cpu = ToCpuF32(Q_cuda->grad());
    auto dK_cpu = ToCpuF32(K_cuda->grad());
    auto dV_cpu = ToCpuF32(V_cuda->grad());
    auto dQ_analytic = ToVector(*dQ_cpu);
    auto dV_analytic = ToVector(*dV_cpu);

    bool has_nan = false;
    for (float v : dQ_analytic) { if (std::isnan(v) || std::isinf(v)) { has_nan = true; break; } }
    if (!has_nan) for (float v : ToVector(*dK_cpu)) { if (std::isnan(v) || std::isinf(v)) { has_nan = true; break; } }
    if (!has_nan) for (float v : dV_analytic) { if (std::isnan(v) || std::isinf(v)) { has_nan = true; break; } }
    std::printf("  NaN/Inf check: %s\n", has_nan ? "FAIL (NaN/Inf found!)" : "PASS");
    assert(!has_nan);

    // Numerical gradient check for Q (causal)
    float Q_base_data[bs * q_head * q_len * head_dim];
    std::copy(static_cast<float *>(Q_cpu->DataPtr()),
              static_cast<float *>(Q_cpu->DataPtr()) + Q_cpu->NumElements(),
              Q_base_data);

    std::printf("  Checking %d dQ entries (numerical vs analytical, causal):\n", num_check);
    float max_dQ_err = 0.0f;
    for (int idx = 0; idx < num_check; idx++) {
        float saved = Q_base_data[idx];

        Q_base_data[idx] = saved + eps;
        auto Q_plus = std::make_shared<Tensor>(Q_base_data,
                                               std::vector<int64_t>{bs, q_head, q_len, head_dim},
                                               DataType::kFLOAT32);
        auto O_plus = std::make_shared<autograd::FlashAttention>(true)
                          ->Apply({ToCudaBf16(Q_plus), ToCudaBf16(K_cpu), ToCudaBf16(V_cpu)})[0];
        float sum_plus = 0.0f;
        for (float v : ToVector(*ToCpuF32(O_plus))) sum_plus += v;

        Q_base_data[idx] = saved - eps;
        auto Q_minus = std::make_shared<Tensor>(Q_base_data,
                                                std::vector<int64_t>{bs, q_head, q_len, head_dim},
                                                DataType::kFLOAT32);
        auto O_minus = std::make_shared<autograd::FlashAttention>(true)
                           ->Apply({ToCudaBf16(Q_minus), ToCudaBf16(K_cpu), ToCudaBf16(V_cpu)})[0];
        float sum_minus = 0.0f;
        for (float v : ToVector(*ToCpuF32(O_minus))) sum_minus += v;

        Q_base_data[idx] = saved;

        float num_grad = (sum_plus - sum_minus) / (2.0f * eps);
        float ana_grad = dQ_analytic[idx];
        float err = std::abs(num_grad - ana_grad);
        max_dQ_err = std::max(max_dQ_err, err);
    }

    const float kGradTol = 0.15f;
    std::printf("  Max dQ error (causal, q_len=128): %.6f  (tolerance %.3f)  %s\n",
                max_dQ_err, kGradTol, max_dQ_err < kGradTol ? "PASS" : "FAIL");
    assert(max_dQ_err < kGradTol);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    // Initialize the parallel environment (single process, single GPU)
    nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1);

    std::cout << "=== Flash Attention Integration Tests ===" << std::endl;

    TestForwardNonCausal();
    TestForwardCausal();
    TestBackward();
    TestBackwardAccuracy();
    TestCausalBackwardAccuracy();

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
