#pragma once

#include <cub/cub.cuh>

namespace infini_train::kernels::maca {

// MACA ships a CUB compatible with the pre-2.8 API (cub::Sum/Max/Min).
// Mirror the CUDA cub_compat.cuh aliases so that kernel code can refer to
// CubSumOp / CubMaxOp / CubMinOp uniformly across backends.
using CubSumOp = cub::Sum;
using CubMaxOp = cub::Max;
using CubMinOp = cub::Min;

} // namespace infini_train::kernels::maca
