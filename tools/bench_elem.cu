// 微基准：标量 vs 128-bit 向量化 的 elementwise 访问模式（与 InfiniTrain kernel 同构）
// 用法: ./bench_elem [元素数]
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::printf("CUDA error %s @ %d\n", cudaGetErrorString(e_), __LINE__);                                     \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

template <typename T, int N> struct __align__(sizeof(T) * N) Avec {
    T val[N];
};

// ---------------- 一元：out = in * scalar ----------------
template <typename T> __global__ void unary_scalar(T *__restrict__ o, const T *__restrict__ i, size_t n, float s) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        o[idx] = (T)((float)i[idx] * s);
    }
}

template <typename T, int V> __global__ void unary_vec(T *__restrict__ o, const T *__restrict__ i, size_t n, float s) {
    using VecT = Avec<T, V>;
    const size_t nv = n / V;
    for (size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x; v < nv; v += (size_t)gridDim.x * blockDim.x) {
        const size_t base = v * V;
        const VecT in = *reinterpret_cast<const VecT *>(i + base);
        VecT out;
#pragma unroll
        for (int k = 0; k < V; ++k) { out.val[k] = (T)((float)in.val[k] * s); }
        *reinterpret_cast<VecT *>(o + base) = out;
    }
}

// ---------------- 二元（同形状）：out = a + b ----------------
template <typename T>
__global__ void binary_scalar(T *__restrict__ o, const T *__restrict__ a, const T *__restrict__ b, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        o[idx] = (T)((float)a[idx] + (float)b[idx]);
    }
}

template <typename T, int V>
__global__ void binary_vec(T *__restrict__ o, const T *__restrict__ a, const T *__restrict__ b, size_t n) {
    using VecT = Avec<T, V>;
    const size_t nv = n / V;
    for (size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x; v < nv; v += (size_t)gridDim.x * blockDim.x) {
        const size_t base = v * V;
        const VecT av = *reinterpret_cast<const VecT *>(a + base);
        const VecT bv = *reinterpret_cast<const VecT *>(b + base);
        VecT out;
#pragma unroll
        for (int k = 0; k < V; ++k) { out.val[k] = (T)((float)av.val[k] + (float)bv.val[k]); }
        *reinterpret_cast<VecT *>(o + base) = out;
    }
}

// ---------------- 二元后向（同形状）：outA = g*b, outB = g*a ----------------
template <typename T>
__global__ void bwd_scalar(T *__restrict__ oa, T *__restrict__ ob, const T *__restrict__ g, const T *__restrict__ a,
                           const T *__restrict__ b, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        oa[idx] = (T)((float)g[idx] * (float)b[idx]);
        ob[idx] = (T)((float)g[idx] * (float)a[idx]);
    }
}

template <typename T, int V>
__global__ void bwd_vec(T *__restrict__ oa, T *__restrict__ ob, const T *__restrict__ g, const T *__restrict__ a,
                        const T *__restrict__ b, size_t n) {
    using VecT = Avec<T, V>;
    const size_t nv = n / V;
    for (size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x; v < nv; v += (size_t)gridDim.x * blockDim.x) {
        const size_t base = v * V;
        const VecT gv = *reinterpret_cast<const VecT *>(g + base);
        const VecT av = *reinterpret_cast<const VecT *>(a + base);
        const VecT bv = *reinterpret_cast<const VecT *>(b + base);
        VecT x, y;
#pragma unroll
        for (int k = 0; k < V; ++k) {
            x.val[k] = (T)((float)gv.val[k] * (float)bv.val[k]);
            y.val[k] = (T)((float)gv.val[k] * (float)av.val[k]);
        }
        *reinterpret_cast<VecT *>(oa + base) = x;
        *reinterpret_cast<VecT *>(ob + base) = y;
    }
}

// ---------------- 类型转换：bf16 <-> fp32 ----------------
template <typename Tdst, typename Tsrc>
__global__ void cast_scalar(Tdst *__restrict__ d, const Tsrc *__restrict__ s, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d[idx] = (Tdst)s[idx];
    }
}

// fp32 -> bf16：每线程 8 个元素（2×16B 读 + 1×16B 写）
__global__ void cast_f32_to_bf16_vec(__nv_bfloat16 *__restrict__ d, const float *__restrict__ s, size_t n) {
    const size_t nv = n / 8;
    for (size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x; v < nv; v += (size_t)gridDim.x * blockDim.x) {
        const size_t base = v * 8;
        const float4 lo = *reinterpret_cast<const float4 *>(s + base);
        const float4 hi = *reinterpret_cast<const float4 *>(s + base + 4);
        const float tmp[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        Avec<__nv_bfloat16, 8> out;
#pragma unroll
        for (int k = 0; k < 8; ++k) { out.val[k] = __float2bfloat16(tmp[k]); }
        *reinterpret_cast<Avec<__nv_bfloat16, 8> *>(d + base) = out;
    }
}

// bf16 -> fp32：每线程 8 个元素（1×16B 读 + 2×16B 写）
__global__ void cast_bf16_to_f32_vec(float *__restrict__ d, const __nv_bfloat16 *__restrict__ s, size_t n) {
    const size_t nv = n / 8;
    for (size_t v = (size_t)blockIdx.x * blockDim.x + threadIdx.x; v < nv; v += (size_t)gridDim.x * blockDim.x) {
        const size_t base = v * 8;
        const Avec<__nv_bfloat16, 8> in = *reinterpret_cast<const Avec<__nv_bfloat16, 8> *>(s + base);
        float tmp[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) { tmp[k] = __bfloat162float(in.val[k]); }
        *reinterpret_cast<float4 *>(d + base) = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
        *reinterpret_cast<float4 *>(d + base + 4) = make_float4(tmp[4], tmp[5], tmp[6], tmp[7]);
    }
}

// ---------------- 计时 ----------------
struct Timer {
    cudaEvent_t s, e;
    Timer() {
        cudaEventCreate(&s);
        cudaEventCreate(&e);
    }
    void start() { cudaEventRecord(s); }
    float stop_ms() {
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0;
        cudaEventElapsedTime(&ms, s, e);
        return ms;
    }
};

template <typename Launch>
static void bench(const char *name, size_t elems, size_t bytes_per_elem_moved, Launch launch, int iters = 200) {
    for (int i = 0; i < 20; ++i) { launch(); }
    CK(cudaDeviceSynchronize());
    Timer t;
    t.start();
    for (int i = 0; i < iters; ++i) { launch(); }
    const float ms = t.stop_ms();
    const double us = ms * 1000.0 / iters;
    const double gbps = (bytes_per_elem_moved * elems) / (us * 1e-6) / 1e9;
    std::printf("  %-24s %9.1f us   %8.0f GB/s\n", name, us, gbps);
}

int main(int argc, char **argv) {
    const size_t n = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (size_t)80 * 64 * 768;
    std::printf("元素数 = %zu  (bf16 = %.1f MB, fp32 = %.1f MB)\n", n, n * 2.0 / 1e6, n * 4.0 / 1e6);

    const int blk = 256;
    const int grid = (int)((n + blk - 1) / blk);

    __nv_bfloat16 *bi, *bo, *bb;
    float *fi, *fo, *fb;
    CK(cudaMalloc(&bi, n * 2));
    CK(cudaMalloc(&bo, n * 2));
    CK(cudaMalloc(&bb, n * 2));
    CK(cudaMalloc(&fi, n * 4));
    CK(cudaMalloc(&fo, n * 4));
    CK(cudaMalloc(&fb, n * 4));

    std::printf("\n[一元 out = in * 0.5]  访存 = 读+写\n");
    bench("bf16 标量", n, 4, [&] { unary_scalar<<<grid, blk>>>(bo, bi, n, 0.5f); });
    bench("bf16 向量(8)", n, 4, [&] { unary_vec<__nv_bfloat16, 8><<<grid / 8, blk>>>(bo, bi, n, 0.5f); });
    bench("fp32 标量", n, 8, [&] { unary_scalar<<<grid, blk>>>(fo, fi, n, 0.5f); });
    bench("fp32 向量(4)", n, 8, [&] { unary_vec<float, 4><<<grid / 4, blk>>>(fo, fi, n, 0.5f); });

    std::printf("\n[二元 out = a + b]  访存 = 2读+1写\n");
    bench("bf16 标量", n, 6, [&] { binary_scalar<<<grid, blk>>>(bo, bi, bb, n); });
    bench("bf16 向量(8)", n, 6, [&] { binary_vec<__nv_bfloat16, 8><<<grid / 8, blk>>>(bo, bi, bb, n); });
    bench("fp32 标量", n, 12, [&] { binary_scalar<<<grid, blk>>>(fo, fi, fb, n); });
    bench("fp32 向量(4)", n, 12, [&] { binary_vec<float, 4><<<grid / 4, blk>>>(fo, fi, fb, n); });

    std::printf("\n[二元后向 oa=g*b, ob=g*a]  访存 = 3读+2写\n");
    bench("bf16 标量", n, 10, [&] { bwd_scalar<<<grid, blk>>>(bo, bb, bi, bi, bb, n); });
    bench("bf16 向量(8)", n, 10, [&] { bwd_vec<__nv_bfloat16, 8><<<grid / 8, blk>>>(bo, bb, bi, bi, bb, n); });
    bench("fp32 标量", n, 20, [&] { bwd_scalar<<<grid, blk>>>(fo, fb, fi, fi, fb, n); });
    bench("fp32 向量(4)", n, 20, [&] { bwd_vec<float, 4><<<grid / 4, blk>>>(fo, fb, fi, fi, fb, n); });

    std::printf("\n[类型转换]  访存 = 读+写\n");
    bench("bf16->fp32 标量", n, 6, [&] { cast_scalar<<<grid, blk>>>(fo, bi, n); });
    bench("bf16->fp32 向量(8)", n, 6, [&] { cast_bf16_to_f32_vec<<<grid / 8, blk>>>(fo, bi, n); });
    bench("fp32->bf16 标量", n, 6, [&] { cast_scalar<<<grid, blk>>>(bo, fi, n); });
    bench("fp32->bf16 向量(8)", n, 6, [&] { cast_f32_to_bf16_vec<<<grid / 8, blk>>>(bo, fi, n); });

    std::printf("\n[空 kernel：纯启动开销]  grid=%d\n", grid);
    bench("空 kernel", 1, 0, [&] { unary_scalar<<<grid, blk>>>(bo, bi, 0, 1.0f); });

    return 0;
}
