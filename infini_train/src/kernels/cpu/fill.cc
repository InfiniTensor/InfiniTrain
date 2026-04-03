#include "glog/logging.h"

#include "infini_train/include/common/cpu/common_cpu.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/dtype_dispatch.h"
#include "infini_train/include/tensor.h"

#include "infini_train/src/core/runtime/cpu/cpu_dispatch.h"

namespace infini_train::kernels::cpu {
void Fill(std::shared_ptr<Tensor> tensor, double value) {
    core::cpu::DispatchCpuFunc<INFINI_ALL_TYPES>(
        tensor->Dtype(),
        [=]<typename T>() {
            auto data = reinterpret_cast<T *>(tensor->DataPtr());
            T casted_value = common::cpu::Cast<T>(value);
            std::fill(data, data + tensor->NumElements(), casted_value);
        },
        "CPU Fill");
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_FILL_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_FILL_KERNEL(Fill)

#undef REGISTER_CPU_FILL_KERNEL
