#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels {

// Shared shape arithmetic for Conv2d (CPU/CUDA must not redefine it).
// Layout: input NCHW, weight OIHW. Kh/Kw are read independently from
// weight dims [2]/[3]; no Kh == Kw assumption.
struct Conv2dMeta {
    int64_t batch = 0;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    int64_t input_h = 0;
    int64_t input_w = 0;
    int64_t kernel_h = 0;
    int64_t kernel_w = 0;
    int64_t stride = 1;
    int64_t padding = 0;
    int64_t output_h = 0;
    int64_t output_w = 0;
    int64_t patches = 0;      // Hout * Wout
    int64_t kernel_elems = 0; // Cin * Kh * Kw
};

inline Conv2dMeta MakeConv2dMeta(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                 int64_t stride, int64_t padding) {
    CHECK(input != nullptr) << "Conv2d input must not be null";
    CHECK(weight != nullptr) << "Conv2d weight must not be null";
    CHECK_GT(stride, 0) << "Conv2d stride must be positive";
    CHECK_GE(padding, 0) << "Conv2d padding must be non-negative";

    const auto &in_dims = input->Dims();
    const auto &w_dims = weight->Dims();
    CHECK_EQ(in_dims.size(), 4) << "Conv2d input must be NCHW";
    CHECK_EQ(w_dims.size(), 4) << "Conv2d weight must be OIHW";
    CHECK_EQ(in_dims[1], w_dims[1]) << "Conv2d input channels must match weight in-channels";

    Conv2dMeta meta;
    meta.batch = in_dims[0];
    meta.in_channels = in_dims[1];
    meta.input_h = in_dims[2];
    meta.input_w = in_dims[3];
    meta.out_channels = w_dims[0];
    meta.kernel_h = w_dims[2];
    meta.kernel_w = w_dims[3];
    meta.stride = stride;
    meta.padding = padding;

    CHECK_GE(meta.input_h + 2 * padding, meta.kernel_h) << "Conv2d kernel taller than padded input";
    CHECK_GE(meta.input_w + 2 * padding, meta.kernel_w) << "Conv2d kernel wider than padded input";
    meta.output_h = (meta.input_h + 2 * padding - meta.kernel_h) / stride + 1;
    meta.output_w = (meta.input_w + 2 * padding - meta.kernel_w) / stride + 1;
    meta.patches = meta.output_h * meta.output_w;
    meta.kernel_elems = meta.in_channels * meta.kernel_h * meta.kernel_w;
    return meta;
}

inline std::vector<int64_t> Conv2dOutputDims(const Conv2dMeta &meta) {
    return {meta.batch, meta.out_channels, meta.output_h, meta.output_w};
}

} // namespace infini_train::kernels
