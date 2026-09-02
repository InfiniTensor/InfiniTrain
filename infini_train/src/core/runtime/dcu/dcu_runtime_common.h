#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include "infini_train/include/core/runtime/runtime_common.h"

namespace infini_train::core {
class Stream;
}

namespace infini_train::core::dcu {

class DcuEvent final : public Event {
public:
    explicit DcuEvent(EventFlag flags = EventFlag::kDefault);
    ~DcuEvent() override;

    hipEvent_t hip_event() const;

private:
    hipEvent_t event_ = nullptr;
};

class DcuStream : public Stream {
public:
    DcuStream();
    explicit DcuStream(int priority);

    // NOTE(dcj):
    // The DcuStream are "leaked": they are created but never destroyed because the
    // destruction of global variables could happen after the HIP runtime has
    // already been destroyed and thus invoking hipStreamDestroy could lead to a
    // crash. It's likely an issue in HIP, but to be safe - let's just "forget"
    // the destruction.
    ~DcuStream() override;

    hipStream_t hip_stream() const;

private:
    hipStream_t stream_ = nullptr;
};

class DcuBlasHandle : public BlasHandle {
public:
    explicit DcuBlasHandle(Stream *stream);

    // NOTE(dcj):
    // The DcuBlasHandle are "leaked": they are created but never destroyed because the
    // destruction of global variables could happen after the HIP runtime has
    // already been destroyed and thus invoking chipblasDestroy could lead to a
    // crash. It's likely an issue in HIP, but to be safe - let's just "forget"
    // the destruction.
    ~DcuBlasHandle() override;

    hipblasHandle_t hipblas_handle() const;

private:
    hipblasHandle_t hipblas_handle_;
};

} // namespace infini_train::core::dcu
