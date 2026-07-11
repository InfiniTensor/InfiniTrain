#pragma once

/// distribution_stubs.h
///
/// 内部细节：声明 uniform 和 normal 两个 DispatchStub。
/// 上层代码不应直接调用这些 stub，而应使用 distribution_kernels.h
/// 中的 uniform_kernel() / normal_kernel() 包装函数。
///
/// 仿 PyTorch aten/src/ATen/native/UnaryOps.h 中的 DECLARE_DISPATCH 声明。

#include <cstdint>
#include <optional>

#include "infini_train/include/core/runtime/dispatch_stub.h"
#include "infini_train/include/generator.h"

namespace infini_train {

// 填 n 个 float 到 data 指向的缓冲区
DECLARE_DISPATCH(void (*)(void *data, int64_t n, double a, double b,
                          const std::optional<Generator> &gen),
                 uniform_stub);

DECLARE_DISPATCH(void (*)(void *data, int64_t n, double a, double b,
                          const std::optional<Generator> &gen),
                 normal_stub);

} // namespace infini_train
