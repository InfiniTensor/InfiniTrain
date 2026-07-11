#pragma once

/// distribution_kernels.h
///
/// 公共 API：uniform_kernel / normal_kernel
///
/// init.cc 等上层代码调用这些包装函数，内部通过 dispatch_stub 自动分发到
/// 正确的设备后端（CPU / 未来 CUDA）。
///
/// 仿 PyTorch aten/src/ATen/native/Distributions.cpp 中的 struct UniformStub 等包装层。

#include <cstdint>
#include <optional>

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"

namespace infini_train {

class Tensor;

void uniform_kernel(Tensor &tensor, double from, double to,
                    const std::optional<Generator> &gen);

void normal_kernel(Tensor &tensor, double mean, double std,
                   const std::optional<Generator> &gen);

} // namespace infini_train
