#include "infini_train/include/tensor.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#ifdef USE_CUDA
#include "cuda_runtime_api.h"
#endif
#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/ops.h"

namespace infini_train {
namespace {
const std::unordered_map<DataType, size_t> kDataTypeToSize = {
    {DataType::kUINT8, 1},    {DataType::kINT8, 1},    {DataType::kUINT16, 2},  {DataType::kINT16, 2},
    {DataType::kUINT32, 4},   {DataType::kINT32, 4},   {DataType::kUINT64, 8},  {DataType::kINT64, 8},
    {DataType::kBFLOAT16, 2}, {DataType::kFLOAT16, 2}, {DataType::kFLOAT32, 4}, {DataType::kFLOAT64, 8},
};

const std::unordered_map<DataType, std::string> kDataTypeToDesc = {
    {DataType::kUINT8, "uint8"},   {DataType::kINT8, "int8"},     {DataType::kUINT16, "uint16"},
    {DataType::kINT16, "int16"},   {DataType::kUINT32, "uint32"}, {DataType::kINT32, "int32"},
    {DataType::kUINT64, "uint64"}, {DataType::kINT64, "int64"},   {DataType::kBFLOAT16, "bf16"},
    {DataType::kFLOAT16, "fp16"},  {DataType::kFLOAT32, "fp32"},  {DataType::kFLOAT64, "fp64"},
};
} // namespace

TensorBuffer::TensorBuffer(Device device, size_t size) : device_(device), size_(size) {
    switch (device_.Type()) {
    case DeviceType::kCPU:
        data_ = malloc(size);
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA:
        // TODO(dcj): Maybe pin memory later.
        cudaMalloc(&data_, size);
        break;
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(device_.Type());
        break;
    }
}

TensorBuffer::~TensorBuffer() {
    switch (device_.Type()) {
    case DeviceType::kCPU:
        free(data_);
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA:
        cudaFree(data_);
        break;
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(device_.Type());
        break;
    }
}

void *TensorBuffer::DataPtr() { return data_; }

const void *TensorBuffer::DataPtr() const { return data_; }

Device TensorBuffer::GetDevice() const { return device_; }

size_t TensorBuffer::Size() const { return size_; }

// Tensor implementation
Tensor::Tensor(const std::vector<int64_t> &dims, DataType dtype, Device device) : dims_(dims), dtype_(dtype) {
    num_elements_ = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
    buffer_ = std::make_shared<TensorBuffer>(device, kDataTypeToSize.at(dtype) * num_elements_);
}

Tensor::Tensor(const Tensor &tensor, size_t offset, const std::vector<int64_t> &dims)
    : buffer_(tensor.buffer_), offset_(offset), dims_(dims),
      num_elements_(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>())), dtype_(tensor.dtype_) {
    CHECK_LE(offset_ + kDataTypeToSize.at(dtype_) * num_elements_, buffer_->Size());
}

Tensor::Tensor(const std::vector<int64_t> &dims, DataType dtype) : Tensor(dims, dtype, Device(DeviceType::kCPU, 0)) {}

Device Tensor::GetDevice() const { return buffer_->GetDevice(); }

void *Tensor::DataPtr() { return reinterpret_cast<uint8_t *>(buffer_->DataPtr()) + offset_; }

const void *Tensor::DataPtr() const { return reinterpret_cast<const uint8_t *>(buffer_->DataPtr()) + offset_; }

size_t Tensor::SizeInBytes() const { return kDataTypeToSize.at(dtype_) * num_elements_; }

const std::vector<int64_t> &Tensor::Dims() const { return dims_; }

size_t Tensor::NumElements() const { return num_elements_; }

DataType Tensor::Dtype() const { return dtype_; }

void Tensor::SetProducer(ops::Op *producer) { producer_ = producer; }

void Tensor::UseGradient() {
    if (!gradient_) {
        gradient_ = std::make_unique<Tensor>(dims_, dtype_, GetDevice());
        gradient_->Fill<float>(0.0f);
    }
}

Tensor *Tensor::Gradient() { return gradient_.get(); }

const Tensor *Tensor::Gradient() const { return gradient_.get(); }

void Tensor::ZeroGrad() {
    if (gradient_) {
        gradient_->Fill<float>(0.0f);
    }
}

void Tensor::Backward() const {
    if (producer_) {
        producer_->Backward(this);
    }
}

template <typename T> void Tensor::Fill(T value) {
    switch (GetDevice().Type()) {
    case DeviceType::kCPU: {
        std::fill(reinterpret_cast<T *>(DataPtr()), reinterpret_cast<T *>(DataPtr()) + num_elements_, value);
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        // TODO(dcj): use thrust::fill later
        std::vector<T> host_buffer(num_elements_, value);
        cudaMemcpy(DataPtr(), host_buffer.data(), num_elements_ * sizeof(T), cudaMemcpyHostToDevice);
        break;
    }
#endif
    default:
        LOG(ERROR) << "Unsupported device type for Tensor::Fill";
        break;
    }
    }
}

template void Tensor::Fill<float>(float);

Tensor Tensor::To(Device device) {
    if (device == buffer_->GetDevice()) {
        auto new_tensor = Tensor(*this, offset_, dims_);
        if (gradient_) {
            new_tensor.gradient_ = std::make_unique<Tensor>(*gradient_.get(), gradient_->offset_, gradient_->dims_);
        }
        return new_tensor;
    }

    Tensor new_tensor;
    switch (device.Type()) {
#ifdef USE_CUDA
    case DeviceType::kCPU:
        // CUDA -> CPU
        new_tensor = Tensor(dims_, dtype_, Device(DeviceType::kCPU, 0));
        cudaMemcpy(new_tensor.DataPtr(), DataPtr(), SizeInBytes(), cudaMemcpyDeviceToHost);
        break;
    case DeviceType::kCUDA:
        // CPU -> CUDA
        new_tensor = Tensor(dims_, dtype_, Device(DeviceType::kCUDA, 0));
        cudaMemcpy(new_tensor.DataPtr(), DataPtr(), SizeInBytes(), cudaMemcpyHostToDevice);
        break;
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(device.Type());
    }

    if (gradient_) {
        new_tensor.gradient_ = std::make_unique<Tensor>(gradient_->To(device));
    }

    return new_tensor;
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    os << "Tensor(data_ptr=" << static_cast<const void *>(tensor.DataPtr()) << ", dims=[";
    for (const auto &dim : tensor.Dims()) { os << dim << ", "; }
    os << "], dtype=" << kDataTypeToDesc.at(tensor.Dtype()) << ")";
    return os;
}
} // namespace infini_train
