#pragma once

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train {
namespace autograd {
class Function;
}

enum class DataType : int8_t {
    kUINT8,
    kINT8,
    kUINT16,
    kINT16,
    kUINT32,
    kINT32,
    kUINT64,
    kINT64,
    kBFLOAT16,
    kFLOAT16,
    kFLOAT32,
    kFLOAT64,
};

class TensorBuffer {
public:
    TensorBuffer(Device device, size_t size);
    ~TensorBuffer();

    void *DataPtr();
    const void *DataPtr() const;

    Device GetDevice() const;
    size_t Size() const;

private:
    Device device_;
    size_t size_ = 0;
    void *data_ = nullptr;
};

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    Tensor() = default;

    Tensor(const std::vector<int64_t> &dims, DataType dtype, Device device);
    Tensor(const std::vector<int64_t> &dims, DataType dtype) : Tensor(dims, dtype, Device(DeviceType::kCPU, 0)) {}
    Tensor(const Tensor &tensor, size_t offset, const std::vector<int64_t> &dims);

    Device GetDevice() const;

    void *DataPtr();
    const void *DataPtr() const;

    size_t SizeInBytes() const;

    const std::vector<int64_t> &Dims() const;
    size_t NumElements() const;
    DataType Dtype() const;

    template <typename T> void Fill(T value);

    // TODO(dcj): return shared_ptr<Tensor> instead of Tensor later
    Tensor To(Device device);

    // operator overloading
    std::shared_ptr<Tensor> Equals(float scalar);
    std::shared_ptr<Tensor> Add(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> Add(float scalar);
    std::shared_ptr<Tensor> Mul(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> Mul(float scalar);
    std::shared_ptr<Tensor> Tanh();
    std::shared_ptr<Tensor> Pow(float exponent);

    std::vector<std::shared_ptr<Tensor>> Split(int split_size, int dim = 0);
    std::shared_ptr<Tensor> Transpose(int dim0, int dim1);
    std::shared_ptr<Tensor> Slice(const std::vector<int64_t> &starts, const std::vector<int64_t> &ends,
                                  const std::vector<int64_t> &steps);

    std::shared_ptr<Tensor> View(const std::vector<int64_t> &dims);
    std::shared_ptr<Tensor> Contiguous();

    // distribution
    std::shared_ptr<Tensor> Uniform(float from = 0.0f, float to = 1.0f,
                                    std::optional<std::mt19937> generator = std::nullopt);

    std::shared_ptr<Tensor> Matmul(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> MaskedFill(const std::shared_ptr<Tensor> &mask, float value);

    friend std::shared_ptr<Tensor> operator==(const std::shared_ptr<Tensor> &t, float scalar);
    friend std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &t1, const std::shared_ptr<Tensor> &t2);
    friend std::shared_ptr<Tensor> operator+(float scalar, const std::shared_ptr<Tensor> &t);
    friend std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &t1, const std::shared_ptr<Tensor> &t2);
    friend std::shared_ptr<Tensor> operator*(float scalar, const std::shared_ptr<Tensor> &t);
    friend std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &t, float scalar);

    friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

private:
    std::shared_ptr<TensorBuffer> buffer_;
    size_t offset_ = 0;
    std::vector<int64_t> dims_;
    size_t num_elements_ = 0;
    DataType dtype_;

    // autograd related
public:
    std::shared_ptr<Tensor> RequiresGrad();

    std::shared_ptr<Tensor> grad() const { return grad_; };
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

    bool is_leaf() const { return is_leaf_; }
    void set_is_leaf(bool is_leaf) { is_leaf_ = is_leaf; }

    std::shared_ptr<autograd::Function> grad_fn() const { return grad_fn_; }
    void set_grad_fn(std::shared_ptr<autograd::Function> grad_fn) { grad_fn_ = grad_fn; }

    int output_idx() const { return output_idx_; }
    void set_output_idx(int output_idx) { output_idx_ = output_idx; }

    void ZeroGrad();

    void Backward(std::shared_ptr<Tensor> gradient = nullptr, bool retain_graph = false,
                  bool create_graph = false) const;

private:
    std::shared_ptr<Tensor> grad_ = nullptr;
    bool requires_grad_ = false;
    bool is_leaf_ = true;
    std::shared_ptr<autograd::Function> grad_fn_ = nullptr;
    int output_idx_ = -1;
};

std::shared_ptr<Tensor> operator==(const std::shared_ptr<Tensor> &t, float scalar);
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &t1, const std::shared_ptr<Tensor> &t2);
std::shared_ptr<Tensor> operator+(float scalar, const std::shared_ptr<Tensor> &t);
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &t1, const std::shared_ptr<Tensor> &t2);
std::shared_ptr<Tensor> operator*(float scalar, const std::shared_ptr<Tensor> &t);
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &t, float scalar);
} // namespace infini_train
