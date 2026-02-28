#pragma once

#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "infini_train/include/core/runtime/runtime_common.h"

namespace infini_train::core {
class Stream;
}

namespace infini_train::core::cuda {

class CudaEvent final : public Event {
public:
    explicit CudaEvent(EventFlag flags = EventFlag::kDefault);
    ~CudaEvent() override;

    cudaEvent_t cuda_event() const;

private:
    cudaEvent_t event_ = nullptr;
};

class CudaStream : public Stream {
public:
    CudaStream();
    explicit CudaStream(int priority);

    // NOTE(dcj):
    // The CudaStream are "leaked": they are created but never destroyed because the
    // destruction of global variables could happen after the CUDA runtime has
    // already been destroyed and thus invoking cudaStreamDestroy could lead to a
    // crash. It's likely an issue in CUDA, but to be safe - let's just "forget"
    // the destruction.
    ~CudaStream() override;

    cudaStream_t cuda_stream() const;

private:
    cudaStream_t stream_ = nullptr;
};

class CudaBlasHandle : public BlasHandle {
public:
    explicit CudaBlasHandle(Stream *stream);
    ~CudaBlasHandle() override;

    cublasHandle_t cublas_handle() const;

private:
    cublasHandle_t cublas_handle_;
};

} // namespace infini_train::core::cuda
