#include <optional>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/kernel_provider/infiniops/adapter.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/src/kernels/common/gemm.h"

#include <infini/ops.h>

namespace infini_train::kernel_provider::infiniops {
namespace {

int ToInfiniOpsTrans(kernels::GemmTranspose op) {
    switch (op) {
    case kernels::GemmTranspose::kNoTranspose:
        return 0;
    case kernels::GemmTranspose::kTranspose:
        return 1;
    }
    LOG(FATAL) << "InfiniOps Gemm: unsupported transpose flag " << static_cast<int>(op);
    return 0; // unreachable
}

std::vector<int64_t> MatrixShape(int batch_count, int rows, int cols) {
    if (batch_count > 1) {
        return {batch_count, rows, cols};
    }
    return {rows, cols};
}

std::vector<int64_t> RowMajorStrides(int batch_count, int ld, long long batch_stride) {
    if (batch_count > 1) {
        return {batch_stride, ld, 1};
    }
    return {ld, 1};
}

infini::ops::Tensor MakeRowMajorTransposeView(const void *data, int batch_count, int column_major_rows,
                                              int column_major_cols, int ld, long long batch_stride, DataType dtype,
                                              const Device &device) {
    return ToOpsTensor(const_cast<void *>(data), MatrixShape(batch_count, column_major_cols, column_major_rows), dtype,
                       device, RowMajorStrides(batch_count, ld, batch_stride));
}

} // namespace

void Gemm(Device device, kernels::GemmParams p) {
    CHECK_GE(p.batch_count, 1);

    const bool trans_a = p.trans_a == kernels::GemmTranspose::kTranspose;
    const bool trans_b = p.trans_b == kernels::GemmTranspose::kTranspose;

    const int a_rows = trans_a ? p.k : p.m;
    const int a_cols = trans_a ? p.m : p.k;
    const int b_rows = trans_b ? p.n : p.k;
    const int b_cols = trans_b ? p.k : p.n;

    core::DeviceGuard guard(device);
    auto handle = GetHandle(device);
    auto a = MakeRowMajorTransposeView(p.A, p.batch_count, a_rows, a_cols, p.lda, p.stride_a, p.input_dtype, device);
    auto b = MakeRowMajorTransposeView(p.B, p.batch_count, b_rows, b_cols, p.ldb, p.stride_b, p.input_dtype, device);
    auto c = MakeRowMajorTransposeView(p.C, p.batch_count, p.m, p.n, p.ldc, p.stride_c, p.output_dtype, device);

    infini::ops::Gemm::Call(handle, {}, b, a, std::optional<float>{p.alpha}, std::optional<float>{p.beta},
                            std::optional<int>{ToInfiniOpsTrans(p.trans_b)},
                            std::optional<int>{ToInfiniOpsTrans(p.trans_a)}, c);
}

} // namespace infini_train::kernel_provider::infiniops

REGISTER_KERNEL(infini_train::Device::DeviceType::kCUDA, Gemm, infini_train::kernel_provider::infiniops::Gemm)
