#include "infini_train/include/kernels/cuda/elementwise.h"

#include <cmath>
#include <functional>
#include <memory>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {
namespace {
std::shared_ptr<Tensor> UnaryForward(const std::shared_ptr<Tensor> &input, std::function<float(float)> unary_fn) {
    return nullptr;
}

std::shared_ptr<Tensor> UnaryBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &a,
                                      std::function<float(float)> unary_fn) {
    return nullptr;
}

std::shared_ptr<Tensor> BinaryForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b,
                                      std::function<float(float, float)> binary_fn) {
    return nullptr;
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> BinaryBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                           const std::shared_ptr<Tensor> &a,
                                                                           const std::shared_ptr<Tensor> &b,
                                                                           std::function<float(float, float)> fn_a,
                                                                           std::function<float(float, float)> fn_b) {
    return {nullptr, nullptr};
}
} // namespace

std::shared_ptr<Tensor> TanhForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForward(input, [](float x) { return tanhf(x); });
}

std::shared_ptr<Tensor> TanhBackward(const std::shared_ptr<Tensor> &grad_output,
                                     const std::shared_ptr<Tensor> &output) {
    return UnaryBackward(grad_output, output, [](float x) { return 1.0 - x * x; });
}

std::shared_ptr<Tensor> PowForward(const std::shared_ptr<Tensor> &input, float exponent) {
    return UnaryForward(input, [exponent](float x) { return powf(x, exponent); });
}

std::shared_ptr<Tensor> PowBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                    float exponent) {
    return UnaryBackward(grad_output, input, [exponent](float x) { return exponent * powf(x, exponent - 1.0f); });
}

std::shared_ptr<Tensor> EqualsScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    return UnaryForward(a, [scalar](float x) { return x == scalar ? 1.0f : 0.0f; });
}

std::shared_ptr<Tensor> AddForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return BinaryForward(a, b, [](float x, float y) { return x + y; });
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> AddBackward(const std::shared_ptr<Tensor> &grad_output) {
    return BinaryBackward(
        grad_output, nullptr, nullptr, [](float, float) { return 1.0f; }, [](float, float) { return 1.0f; });
}

std::shared_ptr<Tensor> AddScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    return UnaryForward(a, [scalar](float x) { return x + scalar; });
}

std::shared_ptr<Tensor> AddScalarBackward(const std::shared_ptr<Tensor> &grad_output) {
    return UnaryBackward(grad_output, nullptr, [](float) { return 1.0f; });
}

std::shared_ptr<Tensor> MulForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return BinaryForward(a, b, [](float x, float y) { return x * y; });
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> MulBackward(const std::shared_ptr<Tensor> &a,
                                                                        const std::shared_ptr<Tensor> &b,
                                                                        const std::shared_ptr<Tensor> &grad_output) {
    return BinaryBackward(
        grad_output, a, b, [](float, float y) { return y; }, [](float x, float) { return x; });
}

std::shared_ptr<Tensor> MulScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    return UnaryForward(a, [scalar](float x) { return x * scalar; });
}

std::shared_ptr<Tensor> MulScalarBackward(const std::shared_ptr<Tensor> &grad_output, float scalar) {
    return UnaryBackward(grad_output, nullptr, [scalar](float) { return scalar; });
}
} // namespace infini_train::kernels::cuda
