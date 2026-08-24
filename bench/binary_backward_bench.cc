// Microbenchmark for BinaryBackward (Mul/Add backward) kernels: BF16 vs FP32.
// Mirrors bench/torch_binary_backward_bench.py for PyTorch comparison.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

namespace {
// Pin the CUDA async mempool so it never releases cached blocks back to the OS. With the default
// release threshold (0), every profiler stream-sync (PROFILE_MODE build) unmaps all cached memory
// and the next iteration re-maps ~200MB of physical pages — the benchmark would measure WSL2 page
// mapping speed (~7ms/iter) instead of the kernels. PyTorch's caching allocator never unmaps, so
// this also keeps the comparison fair.
void PinMemPool() {
    cudaMemPool_t pool = nullptr;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) != cudaSuccess || pool == nullptr) {
        return;
    }
    uint64_t threshold = ~0ull;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
}

// Check that every element of t equals expect (relative error within tol). Returns max rel error.
double MaxRelError(const std::shared_ptr<Tensor> &t, double expect) {
    auto host = t->To(DataType::kFLOAT32).To(Device());
    const float *data = static_cast<const float *>(host.DataPtr());
    double max_err = 0.0;
    for (size_t i = 0; i < host.NumElements(); ++i) {
        const double err = std::abs(static_cast<double>(data[i]) - expect) / std::max(std::abs(expect), 1e-12);
        max_err = std::max(max_err, err);
    }
    return max_err;
}
} // namespace

static float TimeBackward(std::function<void()> fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) { fn(); }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) { fn(); }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms * 1000.0f / iters; // us per iter
}

template <typename Op>
static void RunCase(const char *op_name, std::vector<int64_t> a_dims, std::vector<int64_t> b_dims, DataType dtype) {
    constexpr bool kIsMul = std::is_same_v<Op, autograd::Mul>;
    auto dev = Device(Device::DeviceType::kCUDA, 0);
    auto a = std::make_shared<Tensor>(a_dims, dtype, dev, true);
    a->Fill(2.0f);
    auto b = std::make_shared<Tensor>(b_dims, dtype, dev, true);
    b->Fill(3.0f);
    auto op = std::make_shared<Op>();
    auto out = op->Apply({a, b});
    auto grad = std::make_shared<Tensor>(a_dims, dtype, dev, true);
    grad->Fill(1.0f);

    // One-shot correctness check. a=2, b=3, grad=1:
    //   mul: ga = g*b = 3;        gb = g*a = 2 per use (row-bcast: rows*2, col-bcast: cols*2)
    //   add: ga = g = 1;          gb = g = 1 per use   (row-bcast: rows,   col-bcast: cols)
    {
        auto grads = op->Backward({grad});
        const double ga_expect = kIsMul ? 3.0 : 1.0;
        const double unit = kIsMul ? 2.0 : 1.0;
        double gb_expect = unit;
        if (a_dims != b_dims) {
            gb_expect = unit * (b_dims.size() == 1 ? a_dims[0] : a_dims[1]);
        }
        const double tol = dtype == DataType::kFLOAT32 ? 1e-5 : 1e-2;
        const double err_a = MaxRelError(grads[0], ga_expect);
        const double err_b = MaxRelError(grads[1], gb_expect);
        const bool pass = err_a <= tol && err_b <= tol;
        printf("  correctness: %s (ga max rel err %.2e, gb max rel err %.2e, tol %.0e)\n", pass ? "PASS" : "FAIL",
               err_a, err_b, tol);
    }

    const float us = TimeBackward([&]() { auto g = op->Backward({grad}); }, 20, 200);
    const double bytes = 5.0 * a->NumElements() * (dtype == DataType::kFLOAT32 ? 4 : 2); // g,a,b in; ga,gb out (approx)
    printf("[%-5s] a=[%ld,%ld] b_bcast=%-5s %-8s %8.1f us  (~%.0f GB/s)\n", op_name, a_dims[0], a_dims[1],
           a_dims == b_dims ? "false" : "true", dtype == DataType::kFLOAT32 ? "float32" : "bfloat16", us,
           bytes / (us * 1e-6) / 1e9);
}

int main() {
    google::InitGoogleLogging("binary_backward_bench");
    PinMemPool();
    const std::vector<std::pair<int64_t, int64_t>> shapes = {{65536, 768}, {8192, 3072}};
    for (auto [r, c] : shapes) {
        for (DataType dt : {DataType::kBFLOAT16, DataType::kFLOAT32}) {
            RunCase<autograd::Mul>("mul", {r, c}, {r, c}, dt);
            RunCase<autograd::Mul>("mul", {r, c}, {c}, dt);    // row-broadcast (bias style)
            RunCase<autograd::Mul>("mul", {r, c}, {r, 1}, dt); // col-broadcast
            RunCase<autograd::Add>("add", {r, c}, {r, c}, dt);
            RunCase<autograd::Add>("add", {r, c}, {c}, dt);
        }
    }
    return 0;
}
