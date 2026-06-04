#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <handle.h>

#include "data_type.h"
#include "tensor.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"

namespace infini_train {
class Tensor;
} // namespace infini_train

namespace infini_train::core {
class Stream;
} // namespace infini_train::core

namespace infini_train::kernel_provider::infiniops {

infini::ops::DataType ToOpsDataType(DataType dtype);

infini::ops::Device ToOpsDevice(const Device &device);

using HandleFactory = infini::ops::Handle (*)(const Device &device, core::Stream *stream);

void RegisterHandleFactory(Device::DeviceType type, HandleFactory factory);

infini::ops::Handle GetHandle(const Device &device);

infini::ops::Tensor ToOpsTensor(const std::shared_ptr<Tensor> &tensor);

infini::ops::Tensor ToOpsTensor(void *data, const std::vector<int64_t> &dims, DataType dtype, const Device &device);

infini::ops::Tensor ToOpsTensor(void *data, const std::vector<int64_t> &dims, DataType dtype, const Device &device,
                                const std::vector<int64_t> &strides);

} // namespace infini_train::kernel_provider::infiniops
