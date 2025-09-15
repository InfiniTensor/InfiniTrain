#pragma once

#include "cublas_v2.h"
#include "cuda.h"
#include "cuda_runtime.h"
#ifdef USE_NCCL
#include "nccl.h"
#endif
#include "glog/logging.h"

namespace infini_train::common::cuda {

// Common CUDA Macros
#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            LOG(FATAL) << "CUBLAS Error: " << cublasGetStatusString(status) << " at " << __FILE__ << ":" << __LINE__;  \
        }                                                                                                              \
    } while (0)

#define CUDA_DRIVER_CHECK(call)                                                                                        \
    do {                                                                                                               \
        CUresult status = call;                                                                                        \
        if (status != CUresult::CUDA_SUCCESS) {                                                                        \
            const char *err_str = nullptr;                                                                             \
            cuGetErrorString(status, &err_str);                                                                        \
            LOG(FATAL) << "CUDA Driver API error: " << #call << " failed with error (" << status                       \
                       << "): " << (err_str ? err_str : "Unknown error");                                              \
        }                                                                                                              \
    } while (0)

#ifdef USE_NCCL
#define NCCL_CHECK(expr)                                                                                               \
    do {                                                                                                               \
        ncclResult_t _status = (expr);                                                                                 \
        if (_status != ncclSuccess) {                                                                                  \
            LOG(FATAL) << "NCCL error: " << ncclGetErrorString(_status) << " at " << __FILE__ << ":" << __LINE__       \
                       << " (" << #expr << ")";                                                                        \
        }                                                                                                              \
    } while (0)
#endif

} // namespace infini_train::common::cuda
