#include "infini_train/include/tensor.h"

#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>

#include "Eigen/Dense"
#include "glog/logging.h"

#include "infini_train/include/autograd/accumulate.h"
#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/autograd/function.h"
#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/autograd/matmul.h"
#include "infini_train/include/autograd/misc.h"
#include "infini_train/include/autograd/outer.h"
#include "infini_train/include/autograd/reduction.h"
#include "infini_train/include/autograd/transform.h"
#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/init.h"

namespace infini_train {
TensorBuffer::TensorBuffer(Device device, size_t size) : device_(device), size_(size) {
    core::DeviceGuard guard(device);
    auto *impl = core::GetDeviceGuardImpl(device.type());
    impl->MallocAsync(&data_, size, impl->GetStream(device));
}

TensorBuffer::~TensorBuffer() {
    core::DeviceGuard guard(device_);
    auto *impl = core::GetDeviceGuardImpl(device_.type());
    impl->FreeAsync(data_, impl->GetStream(device_));
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
    : buffer_(tensor.buffer_), offset_(tensor.offset_ + offset), dims_(dims),
      num_elements_(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>())), dtype_(tensor.dtype_) {
    CHECK_LE(offset_ + kDataTypeToSize.at(dtype_) * num_elements_, buffer_->Size());
}

Tensor::Tensor(const float *data, const std::vector<int64_t> &dims, DataType dtype, Device device)
    : dims_(dims), dtype_(dtype),
      num_elements_(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>())) {
    // TODO(dcj): support more datatype
    CHECK(dtype == DataType::kFLOAT32);

    buffer_ = std::make_shared<TensorBuffer>(device, kDataTypeToSize.at(dtype) * num_elements_);

    core::DeviceGuard guard(device);
    auto *impl = core::GetDeviceGuardImpl(device.type());
    impl->MemcpyAsync(buffer_->DataPtr(), data, buffer_->Size(),
                      device.type() == Device::DeviceType::kCPU ? core::MemcpyKind::kD2D : core::MemcpyKind::kH2D,
                      impl->GetStream(device));
    impl->SynchronizeStream(impl->GetStream(device));
}

void Tensor::SetData(const Tensor &tensor, size_t offset, bool preserve_data) {
    CHECK(tensor.GetDevice() == GetDevice());
    CHECK(tensor.Dtype() == Dtype());
    CHECK_LE(tensor.offset_ + offset + SizeInBytes(), tensor.buffer_->Size());

    if (preserve_data) {
        // Create a view of original tensor buffer
        auto new_tensor = Tensor(tensor, offset, Dims());
        // Copy in original data
        new_tensor.CopyFrom(*this);
    }

    buffer_ = tensor.buffer_;
    offset_ = tensor.offset_ + offset;
}

Device Tensor::GetDevice() const { return buffer_->GetDevice(); }

void *Tensor::DataPtr() { return reinterpret_cast<uint8_t *>(buffer_->DataPtr()) + offset_; }

const void *Tensor::DataPtr() const { return reinterpret_cast<const uint8_t *>(buffer_->DataPtr()) + offset_; }

size_t Tensor::SizeInBytes() const { return kDataTypeToSize.at(dtype_) * num_elements_; }

const std::vector<int64_t> &Tensor::Dims() const { return dims_; }

size_t Tensor::NumElements() const { return num_elements_; }

DataType Tensor::Dtype() const { return dtype_; }

template <typename T> void Tensor::Fill(T value) {
    auto device = GetDevice();
    core::DeviceGuard guard(device);

    DataType dtype = Dtype();

    uint64_t storage = 0;

    DispatchFunc<INFINI_ALL_TYPES>(Dtype(), [&storage, value]<typename TargetT>() {
        TargetT casted_value = static_cast<TargetT>(value);
        std::memcpy((void *)(&storage), &casted_value, sizeof(TargetT));
    });

    auto kernel = Dispatcher::Instance().GetKernel({device.type(), "Fill"});
    kernel.Call<void>(shared_from_this(), static_cast<void *>(&storage));
}

template void Tensor::Fill<uint8_t>(uint8_t);
template void Tensor::Fill<int8_t>(int8_t);
template void Tensor::Fill<uint16_t>(uint16_t);
template void Tensor::Fill<int16_t>(int16_t);
template void Tensor::Fill<uint32_t>(uint32_t);
template void Tensor::Fill<int32_t>(int32_t);
template void Tensor::Fill<uint64_t>(uint64_t);
template void Tensor::Fill<int64_t>(int64_t);
template void Tensor::Fill<float>(float);
template void Tensor::Fill<double>(double);
#ifdef USE_CUDA
template void Tensor::Fill<nv_bfloat16>(nv_bfloat16);
template void Tensor::Fill<half>(half);
#endif

Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Tensor::EigenMatrix() {
    const int64_t bs = std::accumulate(dims_.rbegin() + 1, dims_.rend(), 1, std::multiplies<int64_t>());
    return Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        reinterpret_cast<float *>(DataPtr()), bs, *dims_.rbegin());
}

Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>> Tensor::EigenVector() {
    CHECK_EQ(dims_.size(), 1);
    return Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>(reinterpret_cast<float *>(DataPtr()), 1,
                                                                                dims_[0]);
}

Tensor Tensor::To(Device device) {
    const auto buffer_device = buffer_->GetDevice();
    if (device == buffer_device) {
        auto new_tensor = Tensor(*this, offset_, dims_);
        if (grad_) {
            new_tensor.grad_ = std::make_unique<Tensor>(*grad_.get(), grad_->offset_, grad_->dims_);
        }
        return new_tensor;
    }

    Tensor new_tensor;
    if (device.type() == Device::DeviceType::kCPU) {
        // D2H
        new_tensor = Tensor(dims_, dtype_, Device());
        core::DeviceGuard guard(buffer_device);
        auto impl = core::GetDeviceGuardImpl(buffer_device.type());
        impl->MemcpyAsync(new_tensor.DataPtr(), DataPtr(), SizeInBytes(), core::MemcpyKind::kD2H,
                          impl->GetStream(buffer_device));
        impl->SynchronizeStream(impl->GetStream(buffer_device));

    } else if (buffer_device.type() == Device::DeviceType::kCPU) {
        new_tensor = Tensor(dims_, dtype_, device);
        // H2D
        core::DeviceGuard guard(device);
        auto *impl = core::GetDeviceGuardImpl(device.type());
        impl->MemcpyAsync(new_tensor.DataPtr(), DataPtr(), SizeInBytes(), core::MemcpyKind::kH2D,
                          impl->GetStream(device));
        impl->SynchronizeStream(impl->GetStream(device));
    } else {
        new_tensor = Tensor(dims_, dtype_, device);
        // P2P
        //  1. D2H
        Tensor cpu_tensor = To(Device());
        //  2. H2D
        core::DeviceGuard guard(buffer_device);
        auto *impl = core::GetDeviceGuardImpl(buffer_device.type());
        impl->MemcpyAsync(new_tensor.DataPtr(), cpu_tensor.DataPtr(), SizeInBytes(), core::MemcpyKind::kH2D,
                          impl->GetStream(buffer_device));
        impl->SynchronizeStream(impl->GetStream(buffer_device));
    }

    if (grad_) {
        new_tensor.grad_ = std::make_unique<Tensor>(grad_->To(device));
    }

    new_tensor.requires_grad_ = requires_grad_;

    return new_tensor;
}

Tensor Tensor::To(DataType dtype) {
    if (dtype == dtype_) {
        auto new_tensor = Tensor(*this, offset_, dims_);
        if (grad_) {
            new_tensor.grad_ = std::make_unique<Tensor>(*grad_.get(), grad_->offset_, grad_->dims_);
        }
        return new_tensor;
    }

    auto device = GetDevice();
    core::DeviceGuard guard(device);

    auto kernel = Dispatcher::Instance().GetKernel({device.type(), "Cast"});
    auto new_tensor = *kernel.Call<std::shared_ptr<Tensor>>(shared_from_this(), dtype);

    if (grad_) {
        new_tensor.grad_ = std::make_unique<Tensor>(grad_->To(dtype));
    }

    new_tensor.requires_grad_ = requires_grad_;

    return new_tensor;
}

void Tensor::CopyFrom(const Tensor &src) {
    CHECK(Dtype() == src.Dtype()) << "Tensor::CopyFrom dtype mismatch: dst=" << static_cast<int>(Dtype())
                                  << " src=" << static_cast<int>(src.Dtype());
    CHECK_EQ(NumElements(), src.NumElements()) << "Tensor::CopyFrom element count mismatch";
    CHECK(Dims() == src.Dims()) << "Tensor::CopyFrom shape mismatch";

    const size_t nbytes = SizeInBytes();
    const Device dst_dev = GetDevice();
    const Device src_dev = src.GetDevice();

    if (dst_dev == src_dev) {
        core::DeviceGuard guard(dst_dev);
        auto *impl = core::GetDeviceGuardImpl(dst_dev.type());
        impl->MemcpyAsync(DataPtr(), src.DataPtr(), nbytes, core::MemcpyKind::kD2D, impl->GetStream(dst_dev));
        impl->SynchronizeStream(impl->GetStream(dst_dev));
    } else if (dst_dev.type() == Device::DeviceType::kCPU) {
        // D2H
        core::DeviceGuard guard(src_dev);
        auto *impl = core::GetDeviceGuardImpl(src_dev.type());
        impl->MemcpyAsync(DataPtr(), src.DataPtr(), nbytes, core::MemcpyKind::kD2H, impl->GetStream(src_dev));
        impl->SynchronizeStream(impl->GetStream(src_dev));
    } else if (src_dev.type() == Device::DeviceType::kCPU) {
        // H2D
        core::DeviceGuard guard(dst_dev);
        auto *impl = core::GetDeviceGuardImpl(dst_dev.type());
        impl->MemcpyAsync(DataPtr(), src.DataPtr(), nbytes, core::MemcpyKind::kH2D, impl->GetStream(dst_dev));
        impl->SynchronizeStream(impl->GetStream(dst_dev));
    } else {
        // TODO(dcj): maybe support p2p api later
        // P2P
        //  1. D2H
        Tensor cpu_tensor(dims_, dtype_, Device());
        cpu_tensor.CopyFrom(src);
        //  2. H2D
        core::DeviceGuard guard(dst_dev);
        auto *impl = core::GetDeviceGuardImpl(dst_dev.type());
        impl->MemcpyAsync(DataPtr(), cpu_tensor.DataPtr(), nbytes, core::MemcpyKind::kH2D, impl->GetStream(dst_dev));
        impl->SynchronizeStream(impl->GetStream(dst_dev));
    }
}

void Tensor::CopyFrom(const std::shared_ptr<Tensor> &src) { CopyFrom(*src); }

// operator overloading
std::shared_ptr<Tensor> Tensor::Equals(const std::shared_ptr<Tensor> &other) {
    return std::make_shared<autograd::Equals>()->Apply({shared_from_this(), other})[0];
}

std::shared_ptr<Tensor> Tensor::Equals(float scalar) {
    return std::make_shared<autograd::EqualsScalar>(scalar)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Lt(const std::shared_ptr<Tensor> &other) {
    return std::make_shared<autograd::Lt>()->Apply({shared_from_this(), other})[0];
}

std::shared_ptr<Tensor> Tensor::Lt(float scalar) {
    return std::make_shared<autograd::LtScalar>(scalar)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Gt(const std::shared_ptr<Tensor> &other) {
    return std::make_shared<autograd::Gt>()->Apply({shared_from_this(), other})[0];
}

std::shared_ptr<Tensor> Tensor::Gt(float scalar) {
    return std::make_shared<autograd::GtScalar>(scalar)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Le(const std::shared_ptr<Tensor> &other) {
    return std::make_shared<autograd::Le>()->Apply({shared_from_this(), other})[0];
}

std::shared_ptr<Tensor> Tensor::Le(float scalar) {
    return std::make_shared<autograd::LeScalar>(scalar)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Ge(const std::shared_ptr<Tensor> &other) {
    return std::make_shared<autograd::Ge>()->Apply({shared_from_this(), other})[0];
}

std::shared_ptr<Tensor> Tensor::Ge(float scalar) {
    return std::make_shared<autograd::GeScalar>(scalar)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::And(const std::shared_ptr<Tensor> &other) {
    return std::make_shared<autograd::And>()->Apply({shared_from_this(), other})[0];
}

std::shared_ptr<Tensor> Tensor::Or(const std::shared_ptr<Tensor> &other) {
    return std::make_shared<autograd::Or>()->Apply({shared_from_this(), other})[0];
}

std::shared_ptr<Tensor> Tensor::Add(const std::shared_ptr<Tensor> &other) {
    CHECK_EQ(static_cast<int>(GetDevice().type()), static_cast<int>(other->GetDevice().type()));
    return std::make_shared<autograd::Add>()->Apply({shared_from_this(), other})[0];
}

std::shared_ptr<Tensor> Tensor::Add(float scalar) {
    return std::make_shared<autograd::AddScalar>(scalar)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Sub(const std::shared_ptr<Tensor> &other) {
    CHECK_EQ(static_cast<int>(GetDevice().type()), static_cast<int>(other->GetDevice().type()));
    return std::make_shared<autograd::Sub>()->Apply({shared_from_this(), other})[0];
}

std::shared_ptr<Tensor> Tensor::Mul(const std::shared_ptr<Tensor> &other) {
    CHECK_EQ(static_cast<int>(GetDevice().type()), static_cast<int>(other->GetDevice().type()));
    return std::make_shared<autograd::Mul>()->Apply({shared_from_this(), other})[0];
}

std::shared_ptr<Tensor> Tensor::Mul(float scalar) {
    return std::make_shared<autograd::MulScalar>(scalar)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Div(const std::shared_ptr<Tensor> &other) {
    CHECK_EQ(static_cast<int>(GetDevice().type()), static_cast<int>(other->GetDevice().type()));
    return std::make_shared<autograd::Div>()->Apply({shared_from_this(), other})[0];
}

std::shared_ptr<Tensor> Tensor::Neg() { return std::make_shared<autograd::Neg>()->Apply({shared_from_this()})[0]; }

std::shared_ptr<Tensor> Tensor::Reciprocal() {
    return std::make_shared<autograd::Reciprocal>()->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Sin() { return std::make_shared<autograd::Sin>()->Apply({shared_from_this()})[0]; }

std::shared_ptr<Tensor> Tensor::Cos() { return std::make_shared<autograd::Cos>()->Apply({shared_from_this()})[0]; }

std::shared_ptr<Tensor> Tensor::Tanh() { return std::make_shared<autograd::Tanh>()->Apply({shared_from_this()})[0]; }

std::shared_ptr<Tensor> Tensor::Pow(float exponent) {
    return std::make_shared<autograd::Pow>(exponent)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Rsqrt() { return std::make_shared<autograd::Rsqrt>()->Apply({shared_from_this()})[0]; }

std::shared_ptr<Tensor> Tensor::Exp() { return std::make_shared<autograd::Exp>()->Apply({shared_from_this()})[0]; }

std::shared_ptr<Tensor> Tensor::Log() { return std::make_shared<autograd::Log>()->Apply({shared_from_this()})[0]; }

std::shared_ptr<Tensor> Tensor::Mean(int64_t dim, bool keep_dim) {
    return std::make_shared<autograd::Mean>(dim, keep_dim)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Sum(int64_t dim, bool keep_dim) {
    return std::make_shared<autograd::Sum>(dim, keep_dim)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Min(int64_t dim, bool keep_dim) {
    return std::make_shared<autograd::Min>(dim, keep_dim)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Max(int64_t dim, bool keep_dim) {
    return std::make_shared<autograd::Max>(dim, keep_dim)->Apply({shared_from_this()})[0];
}

std::vector<std::shared_ptr<Tensor>> Tensor::Split(int split_size, int dim) {
    if (dim < 0) {
        dim += dims_.size();
    }
    return std::make_shared<autograd::Split>(split_size, dim)->Apply({shared_from_this()});
}

std::shared_ptr<Tensor> Tensor::Gather(int dim, const std::shared_ptr<Tensor> &index) {
    CHECK(GetDevice() == index->GetDevice()) << "index must be on the same device as input.";
    return std::make_shared<autograd::IndexGather>(dim)->Apply({shared_from_this(), index})[0];
}

std::shared_ptr<Tensor> Tensor::RepeatInterleave(int64_t repeat, int64_t dim) {
    if (dim < 0) {
        dim += dims_.size();
    }
    return std::make_shared<autograd::RepeatInterleave>(repeat, dim)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::View(const std::vector<int64_t> &dims) {
    return std::make_shared<autograd::NoOp>(dims)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Contiguous() {
    return std::make_shared<autograd::NoOp>(dims_)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Flatten(int64_t start, int64_t end) {
    auto ndim = dims_.size();
    auto start_dim = start >= 0 ? start : start + ndim;
    auto end_dim = end >= 0 ? end : end + ndim;
    CHECK(start_dim >= 0 && end_dim >= start_dim && end_dim <= ndim);

    std::vector<int64_t> new_shape;
    int64_t flatten_size = 1;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i < start_dim || i > end_dim) {
            new_shape.push_back(dims_[i]);
        } else {
            flatten_size *= dims_[i];
            if (i == end_dim) {
                new_shape.push_back(flatten_size);
            }
        }
    }

    return Contiguous()->View(new_shape);
}

std::shared_ptr<Tensor> Tensor::Squeeze(int64_t dim) {
    std::vector<int64_t> new_shape = dims_;
    if (dim < 0) {
        dim += new_shape.size();
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, new_shape.size());
    CHECK_EQ(new_shape[dim], 1) << "Cannot squeeze dim " << dim << " because size (" << new_shape[dim] << ") != 1.";

    new_shape.erase(new_shape.begin() + dim);

    return Contiguous()->View(new_shape);
}

std::shared_ptr<Tensor> Tensor::Unsqueeze(int64_t dim) {
    if (dim < 0) {
        dim += dims_.size() + 1;
    }
    CHECK_GE(dim, 0);
    CHECK_LE(dim, dims_.size());

    std::vector<int64_t> new_shape = dims_;
    new_shape.insert(new_shape.begin() + dim, 1);

    return Contiguous()->View(new_shape);
}

std::shared_ptr<Tensor> Tensor::Slice(const std::vector<int64_t> &starts, const std::vector<int64_t> &ends,
                                      const std::vector<int64_t> &steps) {
    return std::make_shared<autograd::Slice>(starts, ends, steps)->Apply({shared_from_this()})[0];
}

std::shared_ptr<Tensor> Tensor::Slice(int64_t dim, int64_t start, int64_t end, int64_t step) {
    // Slice only on one dimension
    if (dim < 0) {
        dim += dims_.size();
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, dims_.size());
    std::vector<int64_t> starts(dims_.size(), 0);
    std::vector<int64_t> ends = dims_;
    std::vector<int64_t> steps(dims_.size(), 1);

    starts[dim] = start;
    ends[dim] = end;
    steps[dim] = step;
    return Slice(starts, ends, steps);
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

std::shared_ptr<Tensor> Tensor::Outer(const std::shared_ptr<Tensor> &other) {
    // TODO(zbl): Use Reshape + MatMul
    return std::make_shared<autograd::Outer>()->Apply({shared_from_this(), other})[0];
}

// distribution
std::shared_ptr<Tensor> Tensor::Uniform(float from, float to, std::optional<std::mt19937> generator) {
    return nn::init::Uniform(shared_from_this(), from, to, generator);
}

// autograd related
std::shared_ptr<Tensor> Tensor::RequiresGrad() {
    requires_grad_ = true;
    return shared_from_this();
}

std::shared_ptr<Tensor> Tensor::grad() const { return grad_; };
void Tensor::set_grad(const std::shared_ptr<Tensor> &grad) {
    if (grad) {
        CHECK(grad->GetDevice() == GetDevice());
        CHECK(grad->Dtype() == Dtype());
        CHECK(grad->Dims() == Dims());
        grad_ = grad;
    } else {
        grad_.reset();
    }
}

bool Tensor::requires_grad() const { return requires_grad_; }
void Tensor::set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

bool Tensor::is_leaf() const { return is_leaf_; }
void Tensor::set_is_leaf(bool is_leaf) { is_leaf_ = is_leaf; }

std::shared_ptr<autograd::Function> Tensor::grad_fn() const { return grad_fn_; }
void Tensor::set_grad_fn(std::shared_ptr<autograd::Function> grad_fn) { grad_fn_ = grad_fn; }

int Tensor::output_idx() const { return output_idx_; }
void Tensor::set_output_idx(int output_idx) { output_idx_ = output_idx; }

void Tensor::ZeroGrad(bool set_to_none) {
    if (grad_) {
        if (set_to_none) {
            grad_.reset();
        } else {
            grad_->Fill<float>(0.0f);
        }
    }
}

void Tensor::MarkGradOverwriteOnNextAccum() { grad_overwrite_once_ = true; }

bool Tensor::ConsumeGradOverwriteFlag() {
    bool flag = grad_overwrite_once_;
    grad_overwrite_once_ = false;
    return flag;
}

void Tensor::Backward(std::shared_ptr<Tensor> gradient, bool retain_graph, bool create_graph) const {
    CHECK(!retain_graph && !create_graph) << "Not implemented yet!";
    if (grad_fn_) {
        if (!gradient) {
            CHECK_EQ(dims_.size(), 0);
            gradient = std::make_shared<Tensor>(std::vector<int64_t>{}, dtype_, GetDevice());
            gradient->Fill<float>(1.0f);
        } else {
            CHECK_EQ(static_cast<int>(GetDevice().type()), static_cast<int>(gradient->GetDevice().type()));
            CHECK_EQ(static_cast<int>(dtype_), static_cast<int>(gradient->Dtype()));
            CHECK_EQ(dims_.size(), gradient->Dims().size());
            for (int idx = 0; idx < dims_.size(); ++idx) { CHECK_EQ(dims_[idx], gradient->Dims()[idx]); }
        }
        grad_fn_->BackwardPartial(gradient, output_idx_);
    }
}

std::shared_ptr<autograd::AccumulateGrad> Tensor::grad_accumulator() {
    CHECK_EQ(grad_fn_, nullptr) << "grad_accumulator() should only be called on leaf tensors";

    if (!requires_grad_) {
        return nullptr;
    }

    // TODO(dcj): should lock in multi-thread environment

    if (grad_accumulator_ == nullptr) {
        grad_accumulator_ = std::make_shared<autograd::AccumulateGrad>(shared_from_this());
    }

    return grad_accumulator_;
}

void Tensor::ResetAccumulator() {
    if (grad_accumulator_) {
        grad_accumulator_.reset();
    }
}

void Tensor::RegisterPostAccumulateGradHook(std::shared_ptr<autograd::PostAccumulateGradHook> hook) {
    CHECK(requires_grad_) << "cannot register a hook on a tensor that doesn't require gradient";

    CHECK_EQ(grad_fn_, nullptr) << "post accumulate grad hooks cannot be registered on non-leaf tensors";

    post_accumulate_grad_hook_ = hook;
}

autograd::PostAccumulateGradHook *Tensor::post_accumulate_grad_hook() const { return post_accumulate_grad_hook_.get(); }

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    os << "Tensor(data_ptr=" << static_cast<const void *>(tensor.DataPtr()) << ", dims=[";
    for (const auto &dim : tensor.Dims()) { os << dim << ", "; }
    os << "], dtype=" << kDataTypeToDesc.at(tensor.Dtype()) << ", requires_grad=" << tensor.requires_grad() << ")";
    return os;
}

std::shared_ptr<Tensor> operator==(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return a->Equals(b);
}

std::shared_ptr<Tensor> operator==(const std::shared_ptr<Tensor> &t, float scalar) { return t->Equals(scalar); }

std::shared_ptr<Tensor> operator<(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return a->Lt(b);
}

std::shared_ptr<Tensor> operator<(const std::shared_ptr<Tensor> &a, float scalar) { return a->Lt(scalar); }

std::shared_ptr<Tensor> operator<(float scalar, const std::shared_ptr<Tensor> &a) { return a->Gt(scalar); }

std::shared_ptr<Tensor> operator>(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return a->Gt(b);
}

std::shared_ptr<Tensor> operator>(const std::shared_ptr<Tensor> &a, float scalar) { return a->Gt(scalar); }

std::shared_ptr<Tensor> operator>(float scalar, const std::shared_ptr<Tensor> &a) { return a->Lt(scalar); }

std::shared_ptr<Tensor> operator<=(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return a->Le(b);
}

std::shared_ptr<Tensor> operator<=(const std::shared_ptr<Tensor> &a, float scalar) { return a->Le(scalar); }

std::shared_ptr<Tensor> operator<=(float scalar, const std::shared_ptr<Tensor> &a) { return a->Ge(scalar); }

std::shared_ptr<Tensor> operator>=(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return a->Ge(b);
}

std::shared_ptr<Tensor> operator>=(const std::shared_ptr<Tensor> &a, float scalar) { return a->Ge(scalar); }

std::shared_ptr<Tensor> operator>=(float scalar, const std::shared_ptr<Tensor> &a) { return a->Le(scalar); }

std::shared_ptr<Tensor> operator&(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return a->And(b);
}

std::shared_ptr<Tensor> operator|(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return a->Or(b);
}

std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &t1, const std::shared_ptr<Tensor> &t2) {
    return t1->Add(t2);
}

std::shared_ptr<Tensor> operator+(float scalar, const std::shared_ptr<Tensor> &t) { return t->Add(scalar); }

std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &t, float scalar) { return t->Add(scalar); }

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &t1, const std::shared_ptr<Tensor> &t2) {
    return t1->Sub(t2);
}

std::shared_ptr<Tensor> operator-(float scalar, const std::shared_ptr<Tensor> &t) { return t->Neg()->Add(scalar); }

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &t, float scalar) { return t->Add(-scalar); }

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &t) { return t->Neg(); }

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &t1, const std::shared_ptr<Tensor> &t2) {
    return t1->Mul(t2);
}

std::shared_ptr<Tensor> operator*(float scalar, const std::shared_ptr<Tensor> &t) { return t->Mul(scalar); }

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &t, float scalar) { return t->Mul(scalar); }

std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor> &t1, const std::shared_ptr<Tensor> &t2) {
    return t1->Div(t2);
}

std::shared_ptr<Tensor> operator/(float scalar, const std::shared_ptr<Tensor> &t) {
    return t->Reciprocal()->Mul(scalar);
}

std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor> &t, float scalar) { return t->Mul(1.0f / scalar); }

void Tensor::SaveAsNpy(const std::string &path) const {
    // TODO(zbl): support other dtypes
    CHECK(dtype_ == DataType::kFLOAT32);

    const size_t num_elements = NumElements();
    const size_t num_bytes = num_elements * sizeof(float);

    // Prepare host buffer
    auto impl = core::GetDeviceGuardImpl(GetDevice().type());

    impl->SynchronizeDevice(GetDevice());

    Tensor cpu_tensor(dims_, dtype_, Device());
    cpu_tensor.CopyFrom(*this);

    impl->SynchronizeDevice(GetDevice());

    // Write .npy file
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << "Failed to open file for writing: " << path;

    // Write magic string
    file.write("\x93NUMPY", 6);

    // Write version
    uint8_t major = 1;
    uint8_t minor = 0;
    file.put(major);
    file.put(minor);

    // Construct header
    std::ostringstream header_ss;
    header_ss << "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < dims_.size(); ++i) {
        header_ss << dims_[i];
        if (i != dims_.size() - 1) {
            header_ss << ", ";
        }
    }
    if (dims_.size() == 1) {
        header_ss << ",";
    }
    header_ss << ") }";

    std::string header = header_ss.str();

    // Pad header to 16-byte alignment
    size_t header_len = header.size() + 1; // +1 for newline
    size_t padding = 16 - ((10 + header_len) % 16);
    header.append(padding, ' ');
    header += '\n';

    uint16_t header_size = static_cast<uint16_t>(header.size());
    file.write(reinterpret_cast<const char *>(&header_size), sizeof(header_size));
    file.write(header.c_str(), header.size());

    // Write data
    file.write(reinterpret_cast<const char *>(cpu_tensor.DataPtr()), num_bytes);

    file.close();
}

void Tensor::SetPrintOptions(std::optional<int64_t> precision, std::optional<int64_t> threshold,
                             std::optional<int64_t> edge_items, std::optional<int64_t> linewidth,
                             std::optional<std::string> profile, std::optional<bool> sci_mode) {
    PrintOptions &opts = PrintOptions::Get();
    if (profile) {
        // ref: https://github.com/pytorch/pytorch/blob/main/torch/_tensor_str.py
        std::string &profile_name = *profile;
        std::transform(profile_name.begin(), profile_name.end(), profile_name.begin(), ::tolower);
        if (profile_name == "default") {
            opts.precision = 4;
            opts.threshold = 1000;
            opts.edge_items = 3;
            opts.linewidth = 80;
            opts.sci_mode = std::nullopt;
        } else if (profile_name == "short") {
            opts.precision = 4;
            opts.threshold = 100;
            opts.edge_items = 2;
            opts.linewidth = 80;
            opts.sci_mode = std::nullopt;
        } else if (profile_name == "full") {
            opts.precision = 4;
            opts.threshold = std::numeric_limits<int64_t>::max();
            opts.edge_items = 3;
            opts.linewidth = 80;
            opts.sci_mode = std::nullopt;
        } else {
            LOG(WARNING) << "Undefined profile name: " << profile_name;
        }
    }

    if (precision) {
        opts.precision = *precision;
    }
    if (threshold) {
        opts.threshold = *threshold;
    }
    if (edge_items) {
        opts.edge_items = *edge_items;
    }
    if (linewidth) {
        opts.linewidth = *linewidth;
    }
    if (sci_mode) {
        opts.sci_mode = *sci_mode;
    }
}

void Tensor::Print(std::ostream &os) const {
    /*
        Print tensor in torch.tensor/np.array style.
    */
    // TODO(zbl): support other dtypes
    CHECK(dtype_ == DataType::kFLOAT32);

    const size_t num_elements = NumElements();
    const size_t num_bytes = num_elements * sizeof(float);

    auto impl = core::GetDeviceGuardImpl(GetDevice().type());

    impl->SynchronizeDevice(GetDevice());

    Tensor cpu_tensor(dims_, dtype_, Device());
    cpu_tensor.CopyFrom(*this);

    impl->SynchronizeDevice(GetDevice());

    const float *buffer = static_cast<float *>(cpu_tensor.DataPtr());

    const PrintOptions &opts = PrintOptions::Get();
    const int64_t precision = opts.precision;
    const int64_t threshold = opts.threshold;
    const int64_t edge_items = opts.edge_items;
    const int64_t linewidth = opts.linewidth;
    const int64_t base_indent = 8; // length of "tensor(["

    bool use_sci = opts.sci_mode.value_or(false);
    if (!opts.sci_mode.has_value()) {
        for (int idx = 0; idx < NumElements(); ++idx) {
            const auto v = buffer[idx];
            float abs_v = std::fabs(v);
            if ((abs_v > 0.0f && abs_v < 1e-4f) || abs_v >= 1e+4f) {
                use_sci = true;
                break;
            }
        }
    }

    auto format_float = [&](float val) -> std::string {
        std::ostringstream ss;
        if (use_sci) {
            ss << std::scientific << std::setprecision(precision);
        } else {
            ss << std::fixed << std::setprecision(precision);
        }
        ss << val;
        return ss.str();
    };

    std::vector<std::string> str_vals(num_elements);
    size_t max_width = 0;
    for (size_t i = 0; i < num_elements; ++i) {
        str_vals[i] = format_float(buffer[i]);
        max_width = std::max(max_width, str_vals[i].length());
    }

    const int ndim = dims_.size();

    if (ndim == 0) {
        os << "Tensor(";
        if (num_elements == 0) {
            os << "[], ";
        } else {
            os << str_vals[0] << ", ";
        }
        os << "dtype=float32, shape=())\n";
        return;
    }

    std::function<void(int, size_t, int)> print_rec;
    print_rec = [&](int dim, size_t offset, int indent) {
        os << "[";
        size_t step = 1;
        for (int d = dim + 1; d < ndim; ++d) { step *= dims_[d]; }
        int n = dims_[dim];

        if (dim == ndim - 1) {
            if (n <= 2 * edge_items || num_elements <= threshold) {
                int line_len = base_indent + indent + 1;
                for (int i = 0; i < n; ++i) {
                    if (i > 0) {
                        os << ", ";
                        line_len += 2;
                    }
                    std::string item = str_vals[offset + i];
                    if (linewidth > 0 && line_len + max_width > linewidth) {
                        os << "\n" << std::string(base_indent + indent + 1, ' ');
                        line_len = base_indent + indent + 1;
                    }
                    os << std::setw(max_width) << item;
                    line_len += max_width;
                }
            } else {
                int line_len = base_indent + indent + 1;
                for (int i = 0; i < edge_items; ++i) {
                    if (i > 0) {
                        os << ", ";
                        line_len += 2;
                    }
                    std::string item = str_vals[offset + i];
                    if (linewidth > 0 && line_len + max_width > linewidth) {
                        os << "\n" << std::string(base_indent + indent + 1, ' ');
                        line_len = base_indent + indent + 1;
                    }
                    os << std::setw(max_width) << item;
                    line_len += max_width;
                }
                os << ", ...";
                line_len += 5; // length of ", ..."
                if (linewidth > 0 && line_len + max_width > linewidth) {
                    os << "\n" << std::string(base_indent + indent + 1, ' ');
                    line_len = base_indent + indent + 1;
                } else {
                    os << ", ";
                    line_len += 2;
                }
                for (int i = n - edge_items; i < n; ++i) {
                    if (i > n - edge_items) {
                        os << ", ";
                        line_len += 2;
                    }
                    std::string item = str_vals[offset + i];
                    if (linewidth > 0 && line_len + max_width > linewidth) {
                        os << "\n" << std::string(base_indent + indent + 1, ' ');
                        line_len = base_indent + indent + 1;
                    }
                    os << std::setw(max_width) << item;
                    line_len += max_width;
                }
            }
        } else {
            if (n <= 2 * edge_items || num_elements <= threshold) {
                for (int i = 0; i < n; ++i) {
                    if (i > 0) {
                        if (dim < ndim - 2) {
                            os << ",\n\n" << std::string(base_indent + indent, ' ');
                        } else {
                            os << ",\n" << std::string(base_indent + indent, ' ');
                        }
                    }
                    print_rec(dim + 1, offset + i * step, indent + 1);
                }
            } else {
                for (int i = 0; i < edge_items; ++i) {
                    if (i > 0) {
                        if (dim < ndim - 2) {
                            os << ",\n\n" << std::string(base_indent + indent, ' ');
                        } else {
                            os << ",\n" << std::string(base_indent + indent, ' ');
                        }
                    }
                    print_rec(dim + 1, offset + i * step, indent + 1);
                }
                os << ",\n"
                   << std::string(base_indent + indent, ' ') << "...\n"
                   << std::string(base_indent + indent, ' ');
                for (int i = n - edge_items; i < n; ++i) {
                    if (i > n - edge_items) {
                        if (dim < ndim - 2) {
                            os << ",\n\n" << std::string(base_indent + indent, ' ');
                        } else {
                            os << ",\n" << std::string(base_indent + indent, ' ');
                        }
                    }
                    print_rec(dim + 1, offset + i * step, indent + 1);
                }
            }
        }
        os << "]";
    };

    os << "Tensor(";
    if (num_elements == 0) {
        os << "[], ";
    } else {
        print_rec(0, 0, 0);
        os << ", \n";
    }

    os << std::string(base_indent - 1, ' ') << "dtype=float32, shape=(";
    for (size_t i = 0; i < dims_.size(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        os << dims_[i];
    }
    if (dims_.size() == 1) {
        os << ",";
    }
    os << "))\n";
}
} // namespace infini_train
