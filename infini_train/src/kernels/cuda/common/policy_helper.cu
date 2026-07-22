#include "infini_train/src/kernels/cuda/common/policy_helper.cuh"

#include <algorithm>

namespace infini_train::kernels::cuda::common {

LaunchConfig MakeElementwiseLaunch(int64_t numel, int block_size) {

    int device = 0;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int blocks_per_sm = prop.maxThreadsPerMultiProcessor / block_size;

    int max_blocks = prop.multiProcessorCount * blocks_per_sm;

    int grid = static_cast<int>((numel + block_size - 1) / block_size);

    grid = std::min(grid, max_blocks);

    return {dim3(grid), dim3(block_size)};
}
uint64_t CalcPhiloxCounterOffset(int64_t numel, int64_t threads, int unroll) {

    uint64_t iters_per_thread = (numel + threads * unroll - 1) / (threads * unroll);

    return iters_per_thread * unroll;
}

} // namespace infini_train::kernels::cuda::common
