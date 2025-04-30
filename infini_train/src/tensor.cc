#include "infini_train/include/tensor.h"

#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <unordered_map>
#include <vector>

#ifdef USE_CUDA
#include "cuda_runtime_api.h"
#endif
#include "Eigen/Dense"
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

void Tensor::SaveAsNpy(const std::string &path) const {
    // TODO(zbl): support other dtypes
    CHECK(dtype_ == DataType::kFLOAT32);

    const size_t num_elements = NumElements();
    const size_t num_bytes = num_elements * sizeof(float);

    // Prepare host buffer
    std::vector<float> host_buffer(num_elements);

    if (GetDevice().Type() == DeviceType::kCPU) {
        // If on CPU, direct copy
        std::memcpy(host_buffer.data(), DataPtr(), num_bytes);
    }
#ifdef USE_CUDA
    else if (GetDevice().Type() == DeviceType::kCUDA) {
        // If on CUDA, copy back to host
        cudaError_t err = cudaMemcpy(host_buffer.data(), DataPtr(), num_bytes, cudaMemcpyDeviceToHost);
        CHECK_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);
    }
#endif
    else {
        LOG(FATAL) << "Unsupported device type for SaveAsNpy.";
    }

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
    file.write(reinterpret_cast<const char *>(host_buffer.data()), num_bytes);

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

    std::vector<float> host_buffer(num_elements);

    if (GetDevice().Type() == DeviceType::kCPU) {
        std::memcpy(host_buffer.data(), DataPtr(), num_bytes);
    }
#ifdef USE_CUDA
    else if (GetDevice().Type() == DeviceType::kCUDA) {
        cudaDeviceSynchronize();
        cudaError_t err = cudaMemcpy(host_buffer.data(), DataPtr(), num_bytes, cudaMemcpyDeviceToHost);
        CHECK_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);
    }
#endif
    else {
        LOG(FATAL) << "Unsupported device type for Print.";
    }

    const PrintOptions &opts = PrintOptions::Get();
    const int precision = opts.precision;
    const int threshold = opts.threshold;
    const int edge_items = opts.edge_items;
    const int linewidth = opts.linewidth;
    const int base_indent = 8; // length of "tensor(["

    bool use_sci = opts.sci_mode.value_or(false);
    if (!opts.sci_mode.has_value()) {
        for (float v : host_buffer) {
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
        str_vals[i] = format_float(host_buffer[i]);
        max_width = std::max(max_width, str_vals[i].length());
    }

    const int ndim = dims_.size();

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
                os << ", ..., ";
                line_len += 5; // length of "..., "
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
