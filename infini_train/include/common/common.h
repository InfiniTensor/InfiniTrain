#pragma once

#include <cstdint>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))
#define LOG_LOC(LEVEL, MSG) LOG(LEVEL) << MSG << " at " << __FILE__ << ":" << __LINE__
#define LOG_UNSUPPORTED_DTYPE(DTYPE, CONTEXT_IDENTIFIER)                                                               \
    LOG_LOC(FATAL, WRAP(CONTEXT_IDENTIFIER << ": Unsupported data type: "                                              \
                                                  + kDataTypeToDesc.at(static_cast<infini_train::DataType>(dtype))))

inline std::vector<int64_t> ComputeStrides(const std::vector<int64_t> &dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (int i = dims.size() - 2; i >= 0; --i) { strides[i] = strides[i + 1] * dims[i + 1]; }
    return strides;
}
