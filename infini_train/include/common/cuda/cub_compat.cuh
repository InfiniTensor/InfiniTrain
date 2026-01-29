#pragma once

#include <cub/version.cuh>

namespace infini_train::kernels::cuda {

#if defined(CUB_VERSION) && CUB_VERSION >= 200800
using CubSumOp = ::cuda::std::plus<>;
using CubMaxOp = ::cuda::maximum<>;
using CubMinOp = ::cuda::minimum<>;
#else
using CubSumOp = cub::Sum;
using CubMaxOp = cub::Max;
using CubMinOp = cub::Min;
#endif

} // namespace infini_train::kernels::cuda
