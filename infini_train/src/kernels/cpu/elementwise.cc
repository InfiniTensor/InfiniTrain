#include "infini_train/include/kernels/cpu/elementwise.h"

#include <cmath>
#include <functional>
#include <memory>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {
std::shared_ptr<Tensor> UnaryForward(const std::shared_ptr<Tensor> &input, std::function<float(float)> unary_fn) {
    auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    for (int64_t idx = 0; idx < output->NumElements(); ++idx) {
        reinterpret_cast<float *>(output->DataPtr())[idx] = unary_fn(reinterpret_cast<float *>(input->DataPtr())[idx]);
    }
    return output;
}

std::shared_ptr<Tensor> UnaryBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &a,
                                      std::function<float(float)> unary_fn) {
    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), DataType::kFLOAT32);
    for (int idx = 0; idx < grad_input->NumElements(); ++idx) {
        const float x = a ? reinterpret_cast<float *>(a->DataPtr())[idx] : 0.0f;
        const float grad = reinterpret_cast<float *>(grad_output->DataPtr())[idx];
        reinterpret_cast<float *>(grad_input->DataPtr())[idx] = grad * unary_fn(x);
    }
    return grad_input;
}

std::shared_ptr<Tensor> BinaryForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b,
                                      std::function<float(float, float)> binary_fn) {
    // TODO(dcj): Use broadcast rule instead later.
    CHECK(a->NumElements() >= b->NumElements() && a->NumElements() % b->NumElements() == 0);

    auto output = std::make_shared<Tensor>(a->Dims(), DataType::kFLOAT32);
    for (int idx = 0; idx < output->NumElements(); ++idx) {
        reinterpret_cast<float *>(output->DataPtr())[idx]
            = binary_fn(reinterpret_cast<float *>(a->DataPtr())[idx],
                        reinterpret_cast<float *>(b->DataPtr())[idx % b->NumElements()]);
    }

    return output;
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> BinaryBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                           const std::shared_ptr<Tensor> &a,
                                                                           const std::shared_ptr<Tensor> &b,
                                                                           std::function<float(float, float)> fn_a,
                                                                           std::function<float(float, float)> fn_b) {
    // TODO(dcj): Use broadcast rule instead later.
    CHECK(a->NumElements() >= b->NumElements() && a->NumElements() % b->NumElements() == 0);

    auto grad_a = std::make_shared<Tensor>(a->Dims(), DataType::kFLOAT32);
    auto grad_b = std::make_shared<Tensor>(b->Dims(), DataType::kFLOAT32);
    for (int idx = 0; idx < a->NumElements(); ++idx) {
        const float x = a ? reinterpret_cast<float *>(a->DataPtr())[idx] : 0.0f;
        const float y = b ? reinterpret_cast<float *>(b->DataPtr())[idx % b->NumElements()] : 0.0f;
        const float grad = reinterpret_cast<float *>(grad_output->DataPtr())[idx];
        reinterpret_cast<float *>(grad_a->DataPtr())[idx] = grad * fn_a(x, y);
        reinterpret_cast<float *>(grad_b->DataPtr())[idx] = grad * fn_a(x, y);
    }
    return {grad_a, grad_b};
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
} // namespace infini_train::kernels::cpu
