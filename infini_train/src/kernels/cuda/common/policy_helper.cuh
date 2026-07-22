#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace infini_train::kernels::cuda::common {

struct LaunchConfig {
    dim3 grid;
    dim3 block;
};

LaunchConfig MakeElementwiseLaunch(int64_t numel, int block_size = 256);

uint64_t CalcPhiloxCounterOffset(int64_t numel, int64_t threads, int unroll = 4);

} // namespace infini_train::kernels::cuda::common