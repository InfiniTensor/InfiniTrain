#include "infini_train/include/dispatcher.h"
#include "infini_train/src/kernels/cuda/common/gemm.cuh"

REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, Gemm, infini_train::kernels::cuda::Gemm)
