#include "infini_train/include/core/kernel_provider/infiniops/adapter.h"

#include <map>
#include <unordered_map>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernel_provider::infiniops {

namespace {

inline const std::unordered_map<DataType, infini::ops::DataType> kOpsDataTypeMap = {
    {DataType::kFLOAT16, infini::ops::DataType::kFloat16}, {DataType::kBFLOAT16, infini::ops::DataType::kBFloat16},
    {DataType::kFLOAT32, infini::ops::DataType::kFloat32}, {DataType::kFLOAT64, infini::ops::DataType::kFloat64},
    {DataType::kINT8, infini::ops::DataType::kInt8},       {DataType::kINT16, infini::ops::DataType::kInt16},
    {DataType::kINT32, infini::ops::DataType::kInt32},     {DataType::kINT64, infini::ops::DataType::kInt64},
    {DataType::kUINT8, infini::ops::DataType::kUInt8},     {DataType::kUINT16, infini::ops::DataType::kUInt16},
    {DataType::kUINT32, infini::ops::DataType::kUInt32},   {DataType::kUINT64, infini::ops::DataType::kUInt64},
};

inline const std::unordered_map<Device::DeviceType, infini::ops::Device::Type> kOpsDeviceTypeMap = {
    {Device::DeviceType::kCUDA, infini::ops::Device::Type::kNvidia},
    {Device::DeviceType::kCPU, infini::ops::Device::Type::kCpu},
};

std::map<Device::DeviceType, HandleFactory> &HandleFactories() {
    static std::map<Device::DeviceType, HandleFactory> factories;
    return factories;
}

} // namespace

void RegisterHandleFactory(Device::DeviceType type, HandleFactory factory) {
    CHECK(factory != nullptr);
    auto &factories = HandleFactories();
    CHECK(!factories.contains(type)) << "InfiniOps handle factory already registered for device type "
                                     << static_cast<int>(type);
    factories.emplace(type, factory);
}

infini::ops::Handle GetHandle(const Device &device) {
    auto &factories = HandleFactories();
    auto it = factories.find(device.type());
    CHECK(it != factories.end()) << "InfiniOps handle factory is not registered for device type "
                                 << static_cast<int>(device.type());

    auto *stream = core::GetDeviceGuardImpl(device.type())->GetStream(device);
    return it->second(device, stream);
}

infini::ops::DataType ToOpsDataType(DataType dtype) {
    auto it = kOpsDataTypeMap.find(dtype);
    if (it == kOpsDataTypeMap.end()) {
        LOG(FATAL) << "Unsupported DataType for InfiniOps: " << static_cast<int>(dtype);
        __builtin_unreachable();
    }
    return it->second;
}

infini::ops::Device ToOpsDevice(const Device &device) {
    auto it = kOpsDeviceTypeMap.find(device.type());
    if (it == kOpsDeviceTypeMap.end()) {
        LOG(FATAL) << "Unsupported DeviceType for InfiniOps: " << static_cast<int>(device.type());
        __builtin_unreachable();
    }
    return {it->second, device.index()};
}

namespace {
infini::ops::Tensor::Strides ComputeContiguousStrides(const std::vector<int64_t> &dims) {
    infini::ops::Tensor::Strides strides(dims.size());
    if (dims.empty()) {
        return strides;
    }
    strides.back() = 1;
    for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * static_cast<infini::ops::Tensor::Stride>(dims[i + 1]);
    }
    return strides;
}

infini::ops::Tensor::Shape ToShape(const std::vector<int64_t> &dims) {
    infini::ops::Tensor::Shape shape(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) { shape[i] = static_cast<infini::ops::Tensor::Size>(dims[i]); }
    return shape;
}

infini::ops::Tensor::Strides ToStrides(const std::vector<int64_t> &strides) {
    infini::ops::Tensor::Strides ops_strides(strides.size());
    for (size_t i = 0; i < strides.size(); ++i) {
        ops_strides[i] = static_cast<infini::ops::Tensor::Stride>(strides[i]);
    }
    return ops_strides;
}
} // namespace

infini::ops::Tensor ToOpsTensor(const std::shared_ptr<Tensor> &tensor) {
    const auto &dims = tensor->Dims();
    return {tensor->DataPtr(), ToShape(dims), ToOpsDataType(tensor->Dtype()), ToOpsDevice(tensor->GetDevice()),
            ComputeContiguousStrides(dims)};
}

infini::ops::Tensor ToOpsTensor(void *data, const std::vector<int64_t> &dims, DataType dtype, const Device &device) {
    return {data, ToShape(dims), ToOpsDataType(dtype), ToOpsDevice(device), ComputeContiguousStrides(dims)};
}

infini::ops::Tensor ToOpsTensor(void *data, const std::vector<int64_t> &dims, DataType dtype, const Device &device,
                                const std::vector<int64_t> &strides) {
    CHECK_EQ(dims.size(), strides.size());
    return {data, ToShape(dims), ToOpsDataType(dtype), ToOpsDevice(device), ToStrides(strides)};
}

} // namespace infini_train::kernel_provider::infiniops
