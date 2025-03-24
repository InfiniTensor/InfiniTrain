#pragma once

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train {
namespace ops {
class Op;
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

class Tensor {
public:
    Tensor() = default;

    Tensor(const std::vector<int64_t> &dims, DataType dtype, Device device);
    Tensor(const std::vector<int64_t> &dims, DataType dtype);
    Tensor(const Tensor &tensor, size_t offset, const std::vector<int64_t> &dims);

    Device GetDevice() const;

    void *DataPtr();
    const void *DataPtr() const;

    size_t SizeInBytes() const;

    const std::vector<int64_t> &Dims() const;
    size_t NumElements() const;
    DataType Dtype() const;

    void SetProducer(ops::Op *producer);

    void UseGradient();
    Tensor *Gradient();
    const Tensor *Gradient() const;
    void ZeroGrad();

    void Backward() const;

    template <typename T> void Fill(T value);

    Tensor To(Device device);

    friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

private:
    std::shared_ptr<TensorBuffer> buffer_;
    size_t offset_ = 0;
    std::vector<int64_t> dims_;
    size_t num_elements_ = 0;
    DataType dtype_;

    ops::Op *producer_ = nullptr;
    std::unique_ptr<Tensor> gradient_ = nullptr;
};
} // namespace infini_train
