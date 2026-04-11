#pragma once

#include <cstdint>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"

/**
 * General Utility Macros
 */
#define EXPAND(X) X
// This macro lets you pass an arbitrary expression that may contain internal
// commas to another macro without having the commas causing the expression
// to be interpreted as being multiple arguments
// Basically an alternative for __VA_OPTS__ before C++20
// ref: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Dispatch_v2.h
#define WRAP(...) __VA_ARGS__
#define CAT(a, b) CAT_(a, b)
#define CAT_(a, b) a##b

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))
#define LOG_LOC(LEVEL, MSG) LOG(LEVEL) << MSG << " at " << __FILE__ << ":" << __LINE__

inline std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int i = dims.size() - 2; i >= 0; --i) { strides[i] = strides[i + 1] * dims[i + 1]; }
    return strides;
}
