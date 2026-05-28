#include "infini_train/include/core/kernel_provider/infiniops/adapter.h"

#include "glog/logging.h"

#include "infini_train/src/core/runtime/cuda/cuda_runtime_common.h"

namespace infini_train::kernel_provider::infiniops {
namespace {

infini::ops::Handle MakeCudaHandle(const Device &, core::Stream *stream) {
    auto *cuda_stream = dynamic_cast<core::cuda::CudaStream *>(stream);
    CHECK_NOTNULL(cuda_stream);

    infini::ops::Handle handle;
    handle.set_stream(static_cast<void *>(cuda_stream->cuda_stream()));
    return handle;
}

const bool kCudaHandleFactoryRegistered = []() {
    RegisterHandleFactory(Device::DeviceType::kCUDA, MakeCudaHandle);
    return true;
}();

} // namespace
} // namespace infini_train::kernel_provider::infiniops
