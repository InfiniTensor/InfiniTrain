#pragma once

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipcub/hipcub.hpp>
#ifdef USE_RCCL
#include "infini_train/include/common/dcu/rccl_compat.h"
#endif

#include "glog/logging.h"

#include "infini_train/include/common/dcu/cub_compat.cuh"
#include "infini_train/include/common/dcu/kernel_helper.cuh"

namespace infini_train::common::dcu {

// Common HIP Macros
#define HIP_CHECK(call)                                                                                                \
    do {                                                                                                               \
        hipError_t status = call;                                                                                      \
        if (status != hipSuccess) {                                                                                    \
            LOG(FATAL) << "HIP Error: " << hipGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;         \
        }                                                                                                              \
    } while (0)

#define HIPBLAS_CHECK(call)                                                                                            \
    do {                                                                                                               \
        hipblasStatus_t status = call;                                                                                 \
        if (status != HIPBLAS_STATUS_SUCCESS) {                                                                        \
            LOG(FATAL) << "HIPBLAS Error: status=" << static_cast<int>(status) << " at " << __FILE__ << ":"            \
                       << __LINE__;                                                                                    \
        }                                                                                                              \
    } while (0)

#ifdef USE_RCCL
#define NCCL_CHECK(expr)                                                                                               \
    do {                                                                                                               \
        ncclResult_t _status = (expr);                                                                                 \
        if (_status != ncclSuccess) {                                                                                  \
            LOG(FATAL) << "NCCL error: " << ncclGetErrorString(_status) << " at " << __FILE__ << ":" << __LINE__       \
                       << " (" << #expr << ")";                                                                        \
        }                                                                                                              \
    } while (0)
#endif

} // namespace infini_train::common::dcu
