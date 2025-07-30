#pragma once

#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "Eigen/Dense"
#include "datatype.h"
#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"

namespace infini_train {
namespace autograd {
class Function;
}

namespace {
struct PrintOptions {
    // ref: https://pytorch.org/docs/stable/generated/torch.set_printoptions.html#torch.set_printoptions
    int64_t precision = 4;
    int64_t threshold = 1000;
    int64_t edge_items = 3;
    int64_t linewidth = 80;
    std::optional<bool> sci_mode;

    static PrintOptions &Get() {
        static PrintOptions instance;
        return instance;
    }
};
} // namespace

class TensorBuffer {
public:
    TensorBuffer(const Device *device, size_t size);
    ~TensorBuffer();

    void *DataPtr();
    const void *DataPtr() const;

    const Device *GetDevice() const;
    size_t Size() const;

private:
    const Device *device_ = nullptr;
    size_t size_ = 0;
    void *data_ = nullptr;
};

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    Tensor() = default;

    Tensor(const std::vector<int64_t> &dims, DataType dtype, const Device *device);
    Tensor(const std::vector<int64_t> &dims, DataType dtype)
        : Tensor(dims, dtype, DeviceManager::Instance()->GetDevice(DeviceType::kCPU, 0)) {}
    Tensor(const Tensor &tensor, size_t offset, const std::vector<int64_t> &dims);

    const Device *GetDevice() const;

    void *DataPtr();
    const void *DataPtr() const;

    size_t SizeInBytes() const;

    const std::vector<int64_t> &Dims() const;
    size_t NumElements() const;
    DataType Dtype() const;

    template <typename T> void Fill(T value);

    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> EigenMatrix();
    Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>> EigenVector();

    // TODO(dcj): return shared_ptr<Tensor> instead of Tensor later
    Tensor To(const Device *device);
    Tensor To(DataType dtype);

    // operator overloading
    std::shared_ptr<Tensor> Equals(float scalar);
    std::shared_ptr<Tensor> Add(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> Add(float scalar);
    std::shared_ptr<Tensor> Sub(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> Mul(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> Mul(float scalar);
    std::shared_ptr<Tensor> Div(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> Neg();
    std::shared_ptr<Tensor> Reciprocal();
    std::shared_ptr<Tensor> Sin();
    std::shared_ptr<Tensor> Cos();
    std::shared_ptr<Tensor> Tanh();
    std::shared_ptr<Tensor> Pow(float exponent);
    std::shared_ptr<Tensor> Rsqrt();

    std::vector<std::shared_ptr<Tensor>> Split(int split_size, int dim = 0);
    std::shared_ptr<Tensor> Transpose(int dim0, int dim1);
    std::shared_ptr<Tensor> Slice(const std::vector<int64_t> &starts, const std::vector<int64_t> &ends,
                                  const std::vector<int64_t> &steps);
    std::shared_ptr<Tensor> Slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1);

    std::shared_ptr<Tensor> RepeatInterleave(int64_t repeat, int64_t dim);

    std::shared_ptr<Tensor> View(const std::vector<int64_t> &dims);
    std::shared_ptr<Tensor> Contiguous();
    std::shared_ptr<Tensor> Flatten(int64_t start = 0, int64_t end = -1);
    std::shared_ptr<Tensor> Squeeze(int64_t dim);

    // distribution
    std::shared_ptr<Tensor> Uniform(float from = 0.0f, float to = 1.0f,
                                    std::optional<std::mt19937> generator = std::nullopt);

    std::shared_ptr<Tensor> Matmul(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> Outer(const std::shared_ptr<Tensor> &other);
    std::shared_ptr<Tensor> MaskedFill(const std::shared_ptr<Tensor> &mask, float value);

    friend std::shared_ptr<Tensor> operator==(const std::shared_ptr<Tensor> &t, float scalar);
    friend std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &t1, const std::shared_ptr<Tensor> &t2);
    friend std::shared_ptr<Tensor> operator+(float scalar, const std::shared_ptr<Tensor> &t);
    friend std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> &t, float scalar);
    friend std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &t1, const std::shared_ptr<Tensor> &t2);
    friend std::shared_ptr<Tensor> operator-(float scalar, const std::shared_ptr<Tensor> &t);
    friend std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &t, float scalar);
    friend std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> &t);
    friend std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &t1, const std::shared_ptr<Tensor> &t2);
    friend std::shared_ptr<Tensor> operator*(float scalar, const std::shared_ptr<Tensor> &t);
    friend std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> &t, float scalar);
    friend std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor> &t1, const std::shared_ptr<Tensor> &t2);
    friend std::shared_ptr<Tensor> operator/(float scalar, const std::shared_ptr<Tensor> &t);
    friend std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor> &t, float scalar);

    friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

    void SaveAsNpy(const std::string &path) const;
    void Print(std::ostream &os = std::cout) const;
    static void
    SetPrintOptions(std::optional<int64_t> precision = std::nullopt, std::optional<int64_t> threshold = std::nullopt,
                    std::optional<int64_t> edge_items = std::nullopt, std::optional<int64_t> linewidth = std::nullopt,
                    std::optional<std::string> profile = std::nullopt, std::optional<bool> sci_mode = std::nullopt);

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
    void set_grad(const std::shared_ptr<Tensor> &grad) { grad_ = grad; }
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
