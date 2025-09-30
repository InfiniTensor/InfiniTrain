#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void Fill(std::shared_ptr<Tensor> tensor, void *value_ptr) {
    // FIXME(zbl): support other data types
    auto data = reinterpret_cast<float *>(tensor->DataPtr());
    std::fill(data, data + tensor->NumElements(), *(static_cast<float *>(value_ptr)));
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_FILL_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_FILL_KERNEL(Fill)

#undef REGISTER_CPU_FILL_KERNEL
