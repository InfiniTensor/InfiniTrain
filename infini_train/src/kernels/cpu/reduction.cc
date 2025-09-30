#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> ReduceOpForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim,
                                        const std::function<float(const float *, int64_t)> &reduce_fn) {
    const auto &input_dims = input->Dims();
    int64_t actual_dim = dim < 0 ? dim + input_dims.size() : dim;
    CHECK_GE(actual_dim, 0);
    CHECK_LT(actual_dim, input_dims.size());

    std::vector<int64_t> output_dims = input_dims;
    const int64_t reduce_size = input_dims[dim];
    if (keep_dim) {
        output_dims[actual_dim] = 1;
    } else {
        output_dims.erase(output_dims.begin() + actual_dim);
    }

    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    int64_t N = std::accumulate(input_dims.begin(), input_dims.begin() + actual_dim, 1, std::multiplies<int64_t>());
    int64_t H = input_dims[actual_dim];
    int64_t W = std::accumulate(input_dims.begin() + actual_dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());

    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t w = 0; w < W; ++w) {
            const float *segment = &input_ptr[(n * H) * W + w];
            output_ptr[n * W + w] = reduce_fn(segment, H);
        }
    }

    return output;
}

std::shared_ptr<Tensor> ReduceOpBackwardMask(const std::shared_ptr<Tensor> &input,
                                             const std::shared_ptr<Tensor> &grad_output,
                                             const std::shared_ptr<Tensor> &reduced,
                                             const std::vector<int64_t> &input_dims, const int64_t dim,
                                             const bool keep_dim, const std::function<bool(float, float)> &mask_fn) {
    std::vector<int64_t> grad_output_dims = grad_output->Dims();
    int64_t actual_dim = dim < 0 ? dim + input_dims.size() : dim;
    if (!keep_dim) {
        grad_output_dims.insert(grad_output_dims.begin() + actual_dim, 1);
    }

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);

    int64_t N = std::accumulate(input_dims.begin(), input_dims.begin() + actual_dim, 1, std::multiplies<int64_t>());
    int64_t H = input_dims[actual_dim];
    int64_t W = std::accumulate(input_dims.begin() + actual_dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());

    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    const float *reduced_ptr = static_cast<const float *>(reduced->DataPtr());
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t w = 0; w < W; ++w) {
            int64_t reduced_idx = n * W + w;
            float target = reduced_ptr[reduced_idx];

            for (int64_t h = 0; h < H; ++h) {
                int64_t input_idx = (n * H + h) * W + w;
                float value = input_ptr[input_idx];
                grad_input_ptr[input_idx] = (value == target) ? grad_output_ptr[reduced_idx] : 0.0f;
            }
        }
    }

    return grad_input;
}

std::shared_ptr<Tensor> ReduceOpBackward(const std::shared_ptr<Tensor> &grad_output,
                                         const std::vector<int64_t> &input_dims, const int64_t dim, const bool keep_dim,
                                         const std::function<float(float, int64_t)> &scale_fn) {
    int64_t actual_dim = dim < 0 ? dim + input_dims.size() : dim;
    CHECK_GE(actual_dim, 0);
    CHECK_LT(actual_dim, input_dims.size());

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);

    int64_t N = std::accumulate(input_dims.begin(), input_dims.begin() + actual_dim, 1, std::multiplies<int64_t>());
    int64_t H = input_dims[actual_dim];
    int64_t W = std::accumulate(input_dims.begin() + actual_dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());

    const float *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t w = 0; w < W; ++w) {
                int64_t output_idx = n * W + w;
                int64_t input_idx = (n * H + h) * W + w;
                grad_input_ptr[input_idx] = scale_fn(grad_output_ptr[output_idx], H);
            }
        }
    }

    return grad_input;
}

std::shared_ptr<Tensor> MeanForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim) {
    return ReduceOpForward(input, dim, keep_dim, [](const float *data, int64_t len) {
        float sum = 0.0f;
        for (int64_t i = 0; i < len; ++i) { sum += data[i]; }
        return sum / static_cast<float>(len);
    });
}

std::shared_ptr<Tensor> MeanBackward(const std::shared_ptr<Tensor> &grad_output, const std::vector<int64_t> &input_dims,
                                     const int64_t dim, const bool keep_dim) {
    return ReduceOpBackward(grad_output, input_dims, dim, keep_dim,
                            [](float grad, int64_t len) { return grad / static_cast<float>(len); });
}

std::shared_ptr<Tensor> SumForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim) {
    return ReduceOpForward(input, dim, keep_dim, [](const float *data, int64_t len) {
        float sum = 0.0f;
        for (int64_t i = 0; i < len; ++i) { sum += data[i]; }
        return sum;
    });
}

std::shared_ptr<Tensor> SumBackward(const std::shared_ptr<Tensor> &grad_output, const std::vector<int64_t> &input_dims,
                                    const int64_t dim, const bool keep_dim) {
    return ReduceOpBackward(grad_output, input_dims, dim, keep_dim, [](float grad, int64_t) { return grad; });
}

std::shared_ptr<Tensor> MaxForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim) {
    return ReduceOpForward(input, dim, keep_dim, [](const float *data, int64_t len) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < len; ++i) { max_val = std::max(max_val, data[i]); }
        return max_val;
    });
}

std::shared_ptr<Tensor> MaxBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                    const std::shared_ptr<Tensor> &reduced, const int64_t dim, bool keep_dim) {
    return ReduceOpBackwardMask(input, grad_output, reduced, input->Dims(), dim, keep_dim,
                                [](float val, float current) { return val > current; });
}

std::shared_ptr<Tensor> MinForward(const std::shared_ptr<Tensor> &input, const int64_t dim, const bool keep_dim) {
    return ReduceOpForward(input, dim, keep_dim, [](const float *data, int64_t len) {
        float min_val = std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < len; ++i) { min_val = std::min(min_val, data[i]); }
        return min_val;
    });
}

std::shared_ptr<Tensor> MinBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                    const std::shared_ptr<Tensor> &reduced, const int64_t dim, bool keep_dim) {
    return ReduceOpBackwardMask(input, grad_output, reduced, input->Dims(), dim, keep_dim,
                                [](float val, float current) { return val < current; });
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_REDUCTION_KERNEL(kernel_name)                                                                     \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_REDUCTION_KERNEL(MeanForward)
REGISTER_CPU_REDUCTION_KERNEL(MeanBackward)
REGISTER_CPU_REDUCTION_KERNEL(SumForward)
REGISTER_CPU_REDUCTION_KERNEL(SumBackward)
REGISTER_CPU_REDUCTION_KERNEL(MaxForward)
REGISTER_CPU_REDUCTION_KERNEL(MaxBackward)
REGISTER_CPU_REDUCTION_KERNEL(MinForward)
REGISTER_CPU_REDUCTION_KERNEL(MinBackward)

#undef REGISTER_CPU_REDUCTION_KERNEL
