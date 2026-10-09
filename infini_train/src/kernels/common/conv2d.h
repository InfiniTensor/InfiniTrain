#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define INFINI_CONV_HD __host__ __device__
#else
#define INFINI_CONV_HD
#endif

namespace infini_train::kernels {
struct Conv2dShape {
    int64_t n, ci, h, w, co, kh, kw, oh, ow, stride, padding;
};

// Each output/gradient element has a single writer, including on CUDA.
// This direct implementation favors a small, deterministic baseline over GEMM tuning.
INFINI_CONV_HD inline float Conv2dForwardAt(int64_t i, const float *x, const float *weight, const float *bias,
                                            Conv2dShape s) {
    const int64_t ow = i % s.ow, oh = i / s.ow % s.oh, oc = i / (s.ow * s.oh) % s.co;
    const int64_t n = i / (s.ow * s.oh * s.co);
    float sum = bias ? bias[oc] : 0.0f;
    for (int64_t ic = 0; ic < s.ci; ++ic) {
        for (int64_t kh = 0; kh < s.kh; ++kh) {
            const int64_t ih = oh * s.stride - s.padding + kh;
            if (ih < 0 || ih >= s.h) {
                continue;
            }
            for (int64_t kw = 0; kw < s.kw; ++kw) {
                const int64_t iw = ow * s.stride - s.padding + kw;
                if (iw >= 0 && iw < s.w) {
                    sum += x[((n * s.ci + ic) * s.h + ih) * s.w + iw]
                         * weight[((oc * s.ci + ic) * s.kh + kh) * s.kw + kw];
                }
            }
        }
    }
    return sum;
}

INFINI_CONV_HD inline float Conv2dInputGradAt(int64_t i, const float *weight, const float *dy, Conv2dShape s) {
    const int64_t iw = i % s.w, ih = i / s.w % s.h, ic = i / (s.w * s.h) % s.ci;
    const int64_t n = i / (s.w * s.h * s.ci);
    float sum = 0.0f;
    for (int64_t oc = 0; oc < s.co; ++oc) {
        for (int64_t kh = 0; kh < s.kh; ++kh) {
            const int64_t ph = ih + s.padding - kh;
            if (ph < 0 || ph % s.stride != 0 || ph / s.stride >= s.oh) {
                continue;
            }
            for (int64_t kw = 0; kw < s.kw; ++kw) {
                const int64_t pw = iw + s.padding - kw;
                if (pw >= 0 && pw % s.stride == 0 && pw / s.stride < s.ow) {
                    sum += dy[((n * s.co + oc) * s.oh + ph / s.stride) * s.ow + pw / s.stride]
                         * weight[((oc * s.ci + ic) * s.kh + kh) * s.kw + kw];
                }
            }
        }
    }
    return sum;
}

INFINI_CONV_HD inline float Conv2dWeightGradAt(int64_t i, const float *x, const float *dy, Conv2dShape s) {
    const int64_t kw = i % s.kw, kh = i / s.kw % s.kh, ic = i / (s.kw * s.kh) % s.ci;
    const int64_t oc = i / (s.kw * s.kh * s.ci);
    float sum = 0.0f;
    for (int64_t n = 0; n < s.n; ++n) {
        for (int64_t oh = 0; oh < s.oh; ++oh) {
            const int64_t ih = oh * s.stride - s.padding + kh;
            if (ih < 0 || ih >= s.h) {
                continue;
            }
            for (int64_t ow = 0; ow < s.ow; ++ow) {
                const int64_t iw = ow * s.stride - s.padding + kw;
                if (iw >= 0 && iw < s.w) {
                    sum += x[((n * s.ci + ic) * s.h + ih) * s.w + iw] * dy[((n * s.co + oc) * s.oh + oh) * s.ow + ow];
                }
            }
        }
    }
    return sum;
}

INFINI_CONV_HD inline float Conv2dBiasGradAt(int64_t oc, const float *dy, Conv2dShape s) {
    float sum = 0.0f;
    for (int64_t n = 0; n < s.n; ++n) {
        for (int64_t p = 0; p < s.oh * s.ow; ++p) { sum += dy[(n * s.co + oc) * s.oh * s.ow + p]; }
    }
    return sum;
}
} // namespace infini_train::kernels

#undef INFINI_CONV_HD
