#include "infini_train/include/common/cpu/common_cpu.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> Cast(std::shared_ptr<Tensor> input, DataType dtype) {
    auto device = input->GetDevice();
    auto dst_tensor = std::make_shared<Tensor>(input->Dims(), dtype, device);

    DispatchFunc<DataTypeList<INFINI_ALL_TYPES>, DataTypeList<INFINI_ALL_TYPES>>(
        {dtype, input->Dtype()},
        [=]<typename Tdst, typename Tsrc>() {
            auto dst = static_cast<Tdst *>(dst_tensor->DataPtr());
            auto src = static_cast<const Tsrc *>(input->DataPtr());
#pragma omp parallel for simd
            for (size_t i = 0; i < dst_tensor->NumElements(); ++i) { dst[i] = common::cpu::Cast<Tdst>(src[i]); }
        },
        "CPU Cast");

    return {dst_tensor};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_CAST_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_CAST_KERNEL(Cast)

#undef REGISTER_CPU_CAST_KERNEL
