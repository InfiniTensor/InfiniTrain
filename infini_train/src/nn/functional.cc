#include "infini_train/include/nn/functional.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/autograd/scatter_src.h"
#include "infini_train/include/autograd/reduction.h"
#include "infini_train/include/autograd/softmax.h"
#include "infini_train/include/autograd/transform.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::function {
std::shared_ptr<Tensor> Tril(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    return std::make_shared<autograd::Tril>(diagonal)->Apply({input})[0];
}

std::shared_ptr<Tensor> Triu(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    return std::make_shared<autograd::Triu>(diagonal)->Apply({input})[0];
}

std::shared_ptr<Tensor> Ones(const std::vector<int64_t> size) {
    auto ones = std::make_shared<Tensor>(size, DataType::kFLOAT32);
    return init::Ones(ones);
}

std::shared_ptr<Tensor> Reciprocal(const std::shared_ptr<Tensor> &input) { return input->Reciprocal(); }

std::shared_ptr<Tensor> Sin(const std::shared_ptr<Tensor> &input) { return input->Sin(); }

std::shared_ptr<Tensor> Cos(const std::shared_ptr<Tensor> &input) { return input->Cos(); }

std::shared_ptr<Tensor> Tanh(const std::shared_ptr<Tensor> &input) { return input->Tanh(); }

std::shared_ptr<Tensor> Pow(const std::shared_ptr<Tensor> &input, float exponent) { return input->Pow(exponent); }

std::shared_ptr<Tensor> Pow(float base, const std::shared_ptr<Tensor> &input) {
    return std::make_shared<autograd::Pow>(base, true)->Apply({input})[0];
}

std::shared_ptr<Tensor> Rsqrt(const std::shared_ptr<Tensor> &input) { return input->Rsqrt(); }

std::shared_ptr<Tensor> Mean(const std::shared_ptr<Tensor> &input, int64_t dim, bool keep_dim) {
    return std::make_shared<autograd::Mean>(dim, keep_dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Sum(const std::shared_ptr<Tensor> &input, int64_t dim, bool keep_dim) {
    return std::make_shared<autograd::Sum>(dim, keep_dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Min(const std::shared_ptr<Tensor> &input, int64_t dim, bool keep_dim) {
    return std::make_shared<autograd::Min>(dim, keep_dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Max(const std::shared_ptr<Tensor> &input, int64_t dim, bool keep_dim) {
    return std::make_shared<autograd::Max>(dim, keep_dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Slice(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &starts,
                              const std::vector<int64_t> &ends, const std::vector<int64_t> &steps) {
    return input->Slice(starts, ends, steps);
}

std::shared_ptr<Tensor> Stack(const std::vector<std::shared_ptr<Tensor>> &inputs, int64_t dim) {
    return std::make_shared<autograd::Stack>(dim)->Apply(inputs)[0];
}

std::shared_ptr<Tensor> Concat(const std::vector<std::shared_ptr<Tensor>> &inputs, int64_t dim) {
    return std::make_shared<autograd::Concat>(dim)->Apply(inputs)[0];
}

std::shared_ptr<Tensor> Scatter(const std::shared_ptr<Tensor> &input, int64_t dim,
                                const std::shared_ptr<Tensor> &index,
                                const std::shared_ptr<Tensor> &src) {
    CHECK_EQ(input->Dims().size(), 3);
    CHECK_EQ(src->Dims().size(), 3);
    CHECK_EQ(index->Dims().size(), 2);
    CHECK(index->Dtype() == DataType::kINT64);
    CHECK(input->Dtype() == src->Dtype());
    CHECK(input->GetDevice() == src->GetDevice());
    CHECK(input->GetDevice() == index->GetDevice());
    CHECK_EQ(dim, 1) << "DCU Scatter currently supports dim=1";

    const int64_t batch = input->Dims()[0];
    const int64_t hidden = input->Dims()[2];
    const int64_t rows = src->Dims()[1];
    CHECK_EQ(src->Dims()[0], batch);
    CHECK_EQ(src->Dims()[2], hidden);
    CHECK_EQ(index->Dims()[0], batch);
    CHECK_EQ(index->Dims()[1], rows);

    return std::make_shared<autograd::ScatterSrc>(dim)->Apply({input, index, src})[0];
}

std::shared_ptr<Tensor> Softmax(const std::shared_ptr<Tensor> &input, int64_t dim) {
    return std::make_shared<autograd::Softmax>(dim)->Apply({input})[0];
}

std::shared_ptr<Tensor> Sigmoid(const std::shared_ptr<Tensor> &input) {
    return std::make_shared<autograd::Sigmoid>()->Apply({input})[0];
}
} // namespace infini_train::nn::function
