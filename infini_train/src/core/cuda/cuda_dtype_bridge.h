#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "infini_train/include/core/dtype_bridge.h"

namespace infini_train::core {

// Framework FP16 -> CUDA __half
template <> struct NativeScalar<Device::DeviceType::kCUDA, infini_train::FP16> {
    using type = __half;
};

// Framework BF16 -> CUDA __nv_bfloat16
template <> struct NativeScalar<Device::DeviceType::kCUDA, infini_train::BF16> {
    using type = __nv_bfloat16;
};

} // namespace infini_train::core
