#pragma once

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "glog/logging.h"

namespace infini_train {
class AllowBackward {
public:
    virtual void Backward() = 0;
};

enum class DataType {
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
    explicit TensorBuffer(size_t size);

    uint8_t *DataPtr();
    const uint8_t *DataPtr() const;
    size_t Size() const;

private:
    std::unique_ptr<uint8_t[]> data_ = nullptr;
    size_t size_ = 0;
};

class Tensor {
public:
    Tensor() = default;

    Tensor(const std::vector<int64_t> &dims, DataType dtype);
    Tensor(const Tensor &tensor, size_t offset, const std::vector<int64_t> &dims);

    uint8_t *DataPtr();
    const uint8_t *DataPtr() const;

    size_t SizeInBytes() const;

    std::vector<int64_t> Dims() const;
    size_t NumElements() const;
    DataType Dtype() const;

    void SetProducer(AllowBackward *producer);

    void UseGradient();
    Tensor *Gradient();
    const Tensor *Gradient() const;
    void ZeroGrad();

    void Backward() const;

    template <typename T>
    void Fill(T value);

    friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

private:
    std::shared_ptr<TensorBuffer> buffer_;
    size_t offset_ = 0;
    std::vector<int64_t> dims_;
    size_t num_elements_ = 0;
    DataType dtype_;

    AllowBackward *producer_ = nullptr;
    std::unique_ptr<Tensor> gradient_ = nullptr;
};
} // namespace infini_train
