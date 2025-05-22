#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {
std::shared_ptr<Tensor> UnaryForward(const std::shared_ptr<Tensor> &input, std::function<float(float)> unary_fn) {
    auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    for (int64_t idx = 0; idx < output->NumElements(); ++idx) {
        static_cast<float *>(output->DataPtr())[idx] = unary_fn(static_cast<float *>(input->DataPtr())[idx]);
    }
    return output;
}

std::shared_ptr<Tensor> UnaryBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &a,
                                      std::function<float(float)> unary_fn) {
    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), DataType::kFLOAT32);
    for (int idx = 0; idx < grad_input->NumElements(); ++idx) {
        const float x = a ? static_cast<float *>(a->DataPtr())[idx] : 0.0f;
        const float grad = static_cast<float *>(grad_output->DataPtr())[idx];
        static_cast<float *>(grad_input->DataPtr())[idx] = grad * unary_fn(x);
    }
    return grad_input;
}

std::shared_ptr<Tensor> BinaryForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b,
                                      std::function<float(float, float)> binary_fn) {
    // TODO(dcj): Broadcasting will be supported in the future.
    // Currently, only one-way broadcasting from b to a is assumed by default.
    CHECK(a->NumElements() >= b->NumElements() && a->NumElements() % b->NumElements() == 0);

    auto output = std::make_shared<Tensor>(a->Dims(), DataType::kFLOAT32);
    for (int idx = 0; idx < output->NumElements(); ++idx) {
        static_cast<float *>(output->DataPtr())[idx] = binary_fn(
            static_cast<float *>(a->DataPtr())[idx], static_cast<float *>(b->DataPtr())[idx % b->NumElements()]);
    }

    return output;
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
BinaryBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &a,
               const std::shared_ptr<Tensor> &b, const std::vector<int64_t> &a_dims, const std::vector<int64_t> &b_dims,
               std::function<float(float, float)> fn_a, std::function<float(float, float)> fn_b) {
    // TODO(dcj): Use broadcast rule instead later.
    const auto a_num_elements = std::accumulate(a_dims.begin(), a_dims.end(), 1, std::multiplies<int64_t>());
    const auto b_num_elements = std::accumulate(b_dims.begin(), b_dims.end(), 1, std::multiplies<int64_t>());

    CHECK(a_num_elements >= b_num_elements && a_num_elements % b_num_elements == 0);
    if (a) {
        CHECK(a_num_elements == a->NumElements());
    }
    if (b) {
        CHECK(b_num_elements == b->NumElements());
    }

    auto grad_a = std::make_shared<Tensor>(a_dims, DataType::kFLOAT32);
    auto grad_b = std::make_shared<Tensor>(b_dims, DataType::kFLOAT32);
    grad_a->Fill<float>(0.0f);
    grad_b->Fill<float>(0.0f);
    for (int idx = 0; idx < a_num_elements; ++idx) {
        const float x = a ? static_cast<float *>(a->DataPtr())[idx] : 0.0f;
        const float y = b ? static_cast<float *>(b->DataPtr())[idx % b_num_elements] : 0.0f;
        const float grad = static_cast<float *>(grad_output->DataPtr())[idx];
        static_cast<float *>(grad_a->DataPtr())[idx] = grad * fn_a(x, y);
        static_cast<float *>(grad_b->DataPtr())[idx % b_num_elements] += grad * fn_b(x, y);
    }
    return {grad_a, grad_b};
}
} // namespace

std::shared_ptr<Tensor> NegForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForward(input, [](float x) { return -x; });
}

std::shared_ptr<Tensor> NegBackward(const std::shared_ptr<Tensor> &grad_output) {
    return UnaryBackward(grad_output, nullptr, [](float) { return -1.0f; });
}

std::shared_ptr<Tensor> ReciprocalForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForward(input, [](float x) { return 1.0f / x; });
}

std::shared_ptr<Tensor> ReciprocalBackward(const std::shared_ptr<Tensor> &grad_output,
                                           const std::shared_ptr<Tensor> &input) {
    return UnaryBackward(grad_output, input, [](float x) { return -1.0f / (x * x); });
}

std::shared_ptr<Tensor> SinForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForward(input, [](float x) { return sinf(x); });
}

std::shared_ptr<Tensor> SinBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input) {
    return UnaryBackward(grad_output, input, [](float x) { return cosf(x); });
}

std::shared_ptr<Tensor> CosForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForward(input, [](float x) { return cosf(x); });
}

std::shared_ptr<Tensor> CosBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input) {
    return UnaryBackward(grad_output, input, [](float x) { return -sinf(x); });
}

std::shared_ptr<Tensor> TanhForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForward(input, [](float x) { return tanhf(x); });
}

std::shared_ptr<Tensor> TanhBackward(const std::shared_ptr<Tensor> &grad_output,
                                     const std::shared_ptr<Tensor> &output) {
    return UnaryBackward(grad_output, output, [](float x) { return 1.0 - x * x; });
}

std::shared_ptr<Tensor> PowForward(const std::shared_ptr<Tensor> &input, float scalar, bool scalar_is_base) {
    if (scalar_is_base) {
        return UnaryForward(input, [scalar](float x) { return powf(scalar, x); });
    } else {
        return UnaryForward(input, [scalar](float x) { return powf(x, scalar); });
    }
}

std::shared_ptr<Tensor> PowBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                    float scalar, bool scalar_is_base) {
    if (scalar_is_base) {
        return UnaryBackward(grad_output, input, [scalar](float x) { return std::log(scalar) * powf(scalar, x); });
    } else {
        return UnaryBackward(grad_output, input, [scalar](float x) { return scalar * powf(x, scalar - 1.0f); });
    }
}

std::shared_ptr<Tensor> RsqrtForward(const std::shared_ptr<Tensor> &input) {
    return UnaryForward(input, [](float x) { return 1.0f / std::sqrt(x); });
}

std::shared_ptr<Tensor> RsqrtBackward(const std::shared_ptr<Tensor> &grad_output,
                                      const std::shared_ptr<Tensor> &input) {
    return UnaryBackward(grad_output, input, [](float x) { return -0.5f / (x * std::sqrt(x)); });
}

std::shared_ptr<Tensor> EqualsScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    return UnaryForward(a, [scalar](float x) { return x == scalar ? 1.0f : 0.0f; });
}

std::shared_ptr<Tensor> AddForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    return BinaryForward(a, b, [](float x, float y) { return x + y; });
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> AddBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::vector<int64_t> &a_dims,
                                                                        const std::vector<int64_t> &b_dims) {

    return BinaryBackward(
        grad_output, nullptr, nullptr, a_dims, b_dims, [](float, float) { return 1.0f; },
        [](float, float) { return 1.0f; });
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

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> MulBackward(const std::shared_ptr<Tensor> &grad_output,
                                                                        const std::shared_ptr<Tensor> &a,
                                                                        const std::shared_ptr<Tensor> &b) {
    return BinaryBackward(
        grad_output, a, b, a->Dims(), b->Dims(), [](float, float y) { return y; }, [](float x, float) { return x; });
}

std::shared_ptr<Tensor> MulScalarForward(const std::shared_ptr<Tensor> &a, float scalar) {
    return UnaryForward(a, [scalar](float x) { return x * scalar; });
}

std::shared_ptr<Tensor> MulScalarBackward(const std::shared_ptr<Tensor> &grad_output, float scalar) {
    return UnaryBackward(grad_output, nullptr, [scalar](float) { return scalar; });
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ELEMENTWISE_KERNEL(kernel_name)                                                                   \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ELEMENTWISE_KERNEL(NegForward)
REGISTER_CPU_ELEMENTWISE_KERNEL(NegBackward)
REGISTER_CPU_ELEMENTWISE_KERNEL(ReciprocalForward)
REGISTER_CPU_ELEMENTWISE_KERNEL(ReciprocalBackward)
REGISTER_CPU_ELEMENTWISE_KERNEL(SinForward)
REGISTER_CPU_ELEMENTWISE_KERNEL(SinBackward)
REGISTER_CPU_ELEMENTWISE_KERNEL(CosForward)
REGISTER_CPU_ELEMENTWISE_KERNEL(CosBackward)
REGISTER_CPU_ELEMENTWISE_KERNEL(TanhForward)
REGISTER_CPU_ELEMENTWISE_KERNEL(TanhBackward)
REGISTER_CPU_ELEMENTWISE_KERNEL(PowForward)
REGISTER_CPU_ELEMENTWISE_KERNEL(PowBackward)
REGISTER_CPU_ELEMENTWISE_KERNEL(RsqrtForward)
REGISTER_CPU_ELEMENTWISE_KERNEL(RsqrtBackward)
REGISTER_CPU_ELEMENTWISE_KERNEL(EqualsScalarForward)
REGISTER_CPU_ELEMENTWISE_KERNEL(AddForward)
REGISTER_CPU_ELEMENTWISE_KERNEL(AddBackward)
REGISTER_CPU_ELEMENTWISE_KERNEL(AddScalarForward)
REGISTER_CPU_ELEMENTWISE_KERNEL(AddScalarBackward)
REGISTER_CPU_ELEMENTWISE_KERNEL(MulForward)
REGISTER_CPU_ELEMENTWISE_KERNEL(MulBackward)
REGISTER_CPU_ELEMENTWISE_KERNEL(MulScalarForward)
REGISTER_CPU_ELEMENTWISE_KERNEL(MulScalarBackward)

#undef REGISTER_CPU_ELEMENTWISE_KERNEL
