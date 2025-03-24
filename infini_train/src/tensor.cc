#include "infini_train/include/tensor.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

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

TensorBuffer::TensorBuffer(size_t size) : data_(std::make_unique<uint8_t[]>(size)), size_(size) {}

uint8_t *TensorBuffer::DataPtr() { return data_.get(); }

const uint8_t *TensorBuffer::DataPtr() const { return data_.get(); }

size_t TensorBuffer::Size() const { return size_; }

Tensor::Tensor(const std::vector<int64_t> &dims, DataType dtype)
    : buffer_(std::make_shared<TensorBuffer>(
          kDataTypeToSize.at(dtype) * std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>()))),
      dims_(dims), num_elements_(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>())),
      dtype_(dtype) {}

Tensor::Tensor(const Tensor &tensor, size_t offset, const std::vector<int64_t> &dims)
    : buffer_(tensor.buffer_), offset_(offset), dims_(dims),
      num_elements_(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>())), dtype_(tensor.dtype_) {
    CHECK_LE(offset_ + kDataTypeToSize.at(dtype_) * num_elements_, buffer_->Size());
}

uint8_t *Tensor::DataPtr() { return buffer_->DataPtr() + offset_; }

const uint8_t *Tensor::DataPtr() const { return buffer_->DataPtr() + offset_; }

size_t Tensor::SizeInBytes() const { return kDataTypeToSize.at(dtype_) * num_elements_; }

const std::vector<int64_t> &Tensor::Dims() const { return dims_; }

size_t Tensor::NumElements() const { return num_elements_; }

DataType Tensor::Dtype() const { return dtype_; }

void Tensor::SetProducer(ops::Op *producer) { producer_ = producer; }

void Tensor::UseGradient() {
    if (!gradient_) {
        gradient_ = std::make_unique<Tensor>(dims_, dtype_);
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
    std::fill(reinterpret_cast<T *>(DataPtr()), reinterpret_cast<T *>(DataPtr()) + num_elements_, value);
}

template void Tensor::Fill<float>(float);

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    os << "Tensor(data_ptr=" << static_cast<const void *>(tensor.DataPtr()) << ", dims=[";
    for (const auto &dim : tensor.Dims()) { os << dim << ", "; }
    os << "], dtype=" << kDataTypeToDesc.at(tensor.Dtype()) << ")";
    return os;
}
} // namespace infini_train
