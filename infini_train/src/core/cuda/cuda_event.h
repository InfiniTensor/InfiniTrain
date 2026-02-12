#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "infini_train/include/core/event.h"

namespace infini_train::core::cuda {

class CudaEvent final : public Event {
public:
    explicit CudaEvent(uint32_t flags = cudaEventDefault);
    ~CudaEvent() override;

    cudaEvent_t cuda_event() const;

private:
    cudaEvent_t event_ = nullptr;
};

} // namespace infini_train::core::cuda
