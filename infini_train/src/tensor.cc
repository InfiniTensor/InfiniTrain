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

#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/autograd/matmul.h"
#include "infini_train/include/autograd/misc.h"
#include "infini_train/include/autograd/transform.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"

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

Device Tensor::GetDevice() const { return buffer_->GetDevice(); }

void *Tensor::DataPtr() { return reinterpret_cast<uint8_t *>(buffer_->DataPtr()) + offset_; }

const void *Tensor::DataPtr() const { return reinterpret_cast<const uint8_t *>(buffer_->DataPtr()) + offset_; }

size_t Tensor::SizeInBytes() const { return kDataTypeToSize.at(dtype_) * num_elements_; }

const std::vector<int64_t> &Tensor::Dims() const { return dims_; }

size_t Tensor::NumElements() const { return num_elements_; }

DataType Tensor::Dtype() const { return dtype_; }

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
        if (grad_) {
            new_tensor.grad_ = std::make_unique<Tensor>(*grad_.get(), grad_->offset_, grad_->dims_);
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

    if (grad_) {
        new_tensor.grad_ = std::make_unique<Tensor>(grad_->To(device));
    }

    new_tensor.requires_grad_ = requires_grad_;

    return new_tensor;
}

// operator overloading
std::shared_ptr<Tensor> Tensor::Equals(float scalar) {
    return std::make_shared<autograd::EqualsScalar>(scalar)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Add(const std::shared_ptr<Tensor> &other) {
    CHECK_EQ(static_cast<int>(GetDevice().Type()), static_cast<int>(other->GetDevice().Type()));
    return std::make_shared<autograd::Add>()->Apply({shared_from_this(), other})[0];
}

std::shared_ptr<Tensor> Tensor::Add(float scalar) {
    return std::make_shared<autograd::AddScalar>(scalar)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Mul(const std::shared_ptr<Tensor> &other) {
    CHECK_EQ(static_cast<int>(GetDevice().Type()), static_cast<int>(other->GetDevice().Type()));
    return std::make_shared<autograd::Mul>()->Apply({shared_from_this(), other})[0];
}

std::shared_ptr<Tensor> Tensor::Mul(float scalar) {
    return std::make_shared<autograd::MulScalar>(scalar)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Tanh() { return std::make_shared<autograd::Tanh>()->Apply({shared_from_this()})[0]; }

std::shared_ptr<Tensor> Tensor::Pow(float exponent) {
    return std::make_shared<autograd::Pow>(exponent)->Apply({shared_from_this()})[0];
}

std::vector<std::shared_ptr<Tensor>> Tensor::Split(int split_size, int dim) {
    return std::make_shared<autograd::Split>(split_size, dim)->Apply({shared_from_this()});
}

std::shared_ptr<Tensor> Tensor::View(const std::vector<int64_t> &dims) {
    return std::make_shared<autograd::NoOp>(dims)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Contiguous() {
    return std::make_shared<autograd::NoOp>(dims_)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Slice(const std::vector<int64_t> &starts, const std::vector<int64_t> &ends,
                                      const std::vector<int64_t> &steps) {
    return std::make_shared<autograd::Slice>(starts, ends, steps)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Transpose(int dim0, int dim1) {
    return std::make_shared<autograd::Transpose>(dim0, dim1)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::MaskedFill(const std::shared_ptr<Tensor> &mask, float value) {
    return std::make_shared<autograd::Mask>(mask, value)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Matmul(const std::shared_ptr<Tensor> &other) {
    return std::make_shared<autograd::Matmul>()->Apply({shared_from_this(), other})[0];
}

// distribution
std::shared_ptr<Tensor> Tensor::Uniform(float from, float to, std::optional<std::mt19937> generator) {
    return nn::init::Uniform(shared_from_this(), from, to, generator);
}

// autograd related
std::shared_ptr<Tensor> Tensor::RequiresGrad() {
    requires_grad_ = true;
    if (!grad_) {
        grad_ = std::make_unique<Tensor>(dims_, dtype_, GetDevice());
        grad_->Fill<float>(0.0f);
    }
    return shared_from_this();
}

void Tensor::Backward(std::shared_ptr<Tensor> gradient, bool retain_graph, bool create_graph) const {
    CHECK(!retain_graph && !create_graph) << "Not implemented yet!";
    if (grad_fn_) {
        if (!gradient) {
            CHECK_EQ(dims_.size(), 0);
            gradient = std::make_shared<Tensor>(std::vector<int64_t>{}, dtype_, GetDevice());
            gradient->Fill<float>(1.0f);
        } else {
            CHECK_EQ(static_cast<int>(GetDevice().Type()), static_cast<int>(gradient->GetDevice().Type()));
            CHECK_EQ(static_cast<int>(dtype_), static_cast<int>(gradient->Dtype()));
            CHECK_EQ(dims_.size(), gradient->Dims().size());
            for (int idx = 0; idx < dims_.size(); ++idx) { CHECK_EQ(dims_[idx], gradient->Dims()[idx]); }
        }
        grad_fn_->BackwardPartial(gradient, output_idx_);
    }
}

void Tensor::ZeroGrad() {
    if (grad_) {
        grad_->Fill<float>(0.0f);
    }
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    os << "Tensor(data_ptr=" << static_cast<const void *>(tensor.DataPtr()) << ", dims=[";
    for (const auto &dim : tensor.Dims()) { os << dim << ", "; }
    os << "], dtype=" << kDataTypeToDesc.at(tensor.Dtype()) << ")";
    return os;
}

std::shared_ptr<Tensor> operator==(const std::shared_ptr<Tensor> &t, float scalar) { return t->Equals(scalar); }

std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &t1, const std::shared_ptr<Tensor> &t2) {
    return t1->Add(t2);
}

std::shared_ptr<Tensor> operator+(float scalar, const std::shared_ptr<Tensor> &t) { return t->Add(scalar); }

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &t1, const std::shared_ptr<Tensor> &t2) {
    return t1->Mul(t2);
}

std::shared_ptr<Tensor> operator*(float scalar, const std::shared_ptr<Tensor> &t) { return t->Mul(scalar); }

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &t, float scalar) { return t->Mul(scalar); }
} // namespace infini_train
