#pragma once

#include <mcr/mc_runtime.h>
#include <mcr/mc_runtime_api.h>
#include <mcblas/mcblas.h>

#ifdef USE_MCCL
#include <mccl.h>
#endif

#include "glog/logging.h"

namespace infini_train::common::maca {

// Common MACA Macros
#define MACA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        mcError_t status = call;                                                                                     \
        if (status != mcSuccess) {                                                                                   \
            LOG(FATAL) << "MACA Error: " << mcGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

#define MCBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        mcblasStatus_t status = call;                                                                                  \
        if (status != MCBLAS_STATUS_SUCCESS) {                                                                         \
            LOG(FATAL) << "MCBLAS Error: " << mcblasGetStatusString(status) << " at " << __FILE__ << ":" << __LINE__;  \
        }                                                                                                              \
    } while (0)

#ifdef USE_MCCL
#define MCCL_CHECK(expr)                                                                                               \
    do {                                                                                                               \
        mcclResult_t _status = (expr);                                                                                 \
        if (_status != mcclSuccess) {                                                                                  \
            LOG(FATAL) << "MCCL error: " << mcclGetErrorString(_status) << " at " << __FILE__ << ":" << __LINE__       \
                       << " (" << #expr << ")";                                                                        \
        }                                                                                                              \
    } while (0)
#endif

} // namespace infini_train::common::maca
