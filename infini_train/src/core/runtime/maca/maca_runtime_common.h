#pragma once

#include <cstdint>

#include <mcblas/mcblas.h>
#include <mcr/mc_runtime.h>
#include <mcr/mc_runtime_api.h>

#include "infini_train/include/core/runtime/runtime_common.h"

namespace infini_train::core {
class Stream;
}

namespace infini_train::core::maca {

class MacaEvent final : public Event {
public:
    explicit MacaEvent(EventFlag flags = EventFlag::kDefault);
    ~MacaEvent() override;

    mcEvent_t maca_event() const;

private:
    mcEvent_t event_ = nullptr;
};

class MacaStream : public Stream {
public:
    MacaStream();
    explicit MacaStream(int priority);

    // NOTE(dcj):
    // Mirror CudaStream: destruction of global variables may outlive the MACA
    // runtime, so we intentionally leak the underlying mcStream_t rather than
    // risk calling mcStreamDestroy after runtime teardown.
    ~MacaStream() override;

    mcStream_t maca_stream() const;

private:
    mcStream_t stream_ = nullptr;
};

class MacaBlasHandle : public BlasHandle {
public:
    explicit MacaBlasHandle(Stream *stream);

    // NOTE(dcj):
    // Mirror CudaBlasHandle: leaked intentionally; see MacaStream note.
    ~MacaBlasHandle() override;

    mcblasHandle_t mcblas_handle() const;

private:
    mcblasHandle_t mcblas_handle_;
};

} // namespace infini_train::core::maca
