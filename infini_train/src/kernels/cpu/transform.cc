#include <cmath>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::shared_ptr<Tensor> TrilForward(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    CHECK_EQ(input->Dims().size(), 2);

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    for (int i = 0; i < input->NumElements(); ++i) {
        int64_t row = i / input->Dims()[1];
        int64_t col = i % input->Dims()[1];
        if (row - col + diagonal >= 0) {
            static_cast<float *>(output->DataPtr())[i] = static_cast<float *>(input->DataPtr())[i];
        } else {
            static_cast<float *>(output->DataPtr())[i] = 0.0;
        }
    }
    return output;
}

std::shared_ptr<Tensor> TrilBackward(const std::shared_ptr<Tensor> &grad_output, int64_t diagonal) {
    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());
    for (int i = 0; i < grad_output->NumElements(); ++i) {
        int64_t row = i / grad_output->Dims()[1];
        int64_t col = i % grad_output->Dims()[1];
        if (row - col + diagonal >= 0) {
            static_cast<float *>(grad_input->DataPtr())[i] = static_cast<float *>(grad_output->DataPtr())[i];
        } else {
            static_cast<float *>(grad_input->DataPtr())[i] = 0.0;
        }
    }
    return grad_input;
}

std::shared_ptr<Tensor> TriuForward(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    CHECK_EQ(input->Dims().size(), 2);

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    for (int i = 0; i < input->NumElements(); ++i) {
        int64_t row = i / input->Dims()[1];
        int64_t col = i % input->Dims()[1];
        if (row - col + diagonal <= 0) {
            static_cast<float *>(output->DataPtr())[i] = static_cast<float *>(input->DataPtr())[i];
        } else {
            static_cast<float *>(output->DataPtr())[i] = 0.0f;
        }
    }
    return output;
}

std::shared_ptr<Tensor> TriuBackward(const std::shared_ptr<Tensor> &grad_output, int64_t diagonal) {
    CHECK_EQ(grad_output->Dims().size(), 2);

    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());
    for (int i = 0; i < grad_output->NumElements(); ++i) {
        int64_t row = i / grad_output->Dims()[1];
        int64_t col = i % grad_output->Dims()[1];
        if (row - col + diagonal <= 0) {
            static_cast<float *>(grad_input->DataPtr())[i] = static_cast<float *>(grad_output->DataPtr())[i];
        } else {
            static_cast<float *>(grad_input->DataPtr())[i] = 0.0f;
        }
    }
    return grad_input;
}

std::shared_ptr<Tensor> TransposeForward(const std::shared_ptr<Tensor> &input, int64_t dim0, int64_t dim1) {
    dim0 = dim0 < 0 ? dim0 + input->Dims().size() : dim0;
    dim1 = dim1 < 0 ? dim1 + input->Dims().size() : dim1;
    CHECK(dim0 >= 0 && dim0 < input->Dims().size() && dim1 >= 0 && dim1 < input->Dims().size());

    auto in_dims = input->Dims();
    std::vector<int64_t> out_dims = in_dims;
    std::swap(out_dims[dim0], out_dims[dim1]);

    auto output = std::make_shared<Tensor>(out_dims, input->Dtype(), input->GetDevice());

    const float *in_ptr = static_cast<const float *>(input->DataPtr());
    float *out_ptr = static_cast<float *>(output->DataPtr());

    // compute strides of in_dims and out_dims
    std::vector<int64_t> in_strides(in_dims.size(), 1);
    std::vector<int64_t> out_strides(out_dims.size(), 1);
    for (int i = in_dims.size() - 2; i >= 0; --i) {
        in_strides[i] = in_strides[i + 1] * in_dims[i + 1];
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
    }

    for (int64_t idx = 0; idx < output->NumElements(); ++idx) {
        // multi-dimensional indices from flat index of input
        int64_t temp = idx;
        std::vector<int64_t> in_index(in_dims.size());
        for (int i = 0; i < in_dims.size(); ++i) {
            in_index[i] = temp / in_strides[i];
            temp %= in_strides[i];
        }

        // swap indices at dim0 and dim1
        std::swap(in_index[dim0], in_index[dim1]);

        // flat index of output
        int64_t out_idx = 0;
        for (int i = 0; i < out_dims.size(); ++i) { out_idx += in_index[i] * out_strides[i]; }

        out_ptr[out_idx] = in_ptr[idx];
    }

    return output;
}

std::shared_ptr<Tensor> TransposeBackward(const std::shared_ptr<Tensor> &grad_output, int64_t dim0, int64_t dim1) {
    return TransposeForward(grad_output, dim1, dim0);
}

std::shared_ptr<Tensor> MaskForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &mask,
                                    float value) {
    CHECK_EQ(input->NumElements() % mask->NumElements(), 0);
    CHECK_EQ(static_cast<int>(input->Dtype()), static_cast<int>(mask->Dtype()));
    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());

    const float *in_ptr = static_cast<const float *>(input->DataPtr());

    for (int i = 0; i < input->NumElements(); ++i) {
        // TODO(dcj): use bool mask when dtype is enabled.
        if ((std::abs(static_cast<const float *>(mask->DataPtr())[i % mask->NumElements()] - 1.0f) < 1e-5)) {
            static_cast<float *>(output->DataPtr())[i] = value;
        } else {
            static_cast<float *>(output->DataPtr())[i] = in_ptr[i];
        }
    }
    return output;
}

std::shared_ptr<Tensor> MaskBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &mask) {
    CHECK_EQ(grad_output->NumElements() % mask->NumElements(), 0);
    CHECK_EQ(static_cast<int>(grad_output->Dtype()), static_cast<int>(mask->Dtype()));
    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());

    for (int i = 0; i < grad_output->NumElements(); ++i) {
        if ((std::abs(static_cast<const float *>(mask->DataPtr())[i % mask->NumElements()] - 1.0f) < 1e-5)) {
            static_cast<float *>(grad_input->DataPtr())[i] = 0.0;
        } else {
            static_cast<float *>(grad_input->DataPtr())[i] = static_cast<const float *>(grad_output->DataPtr())[i];
        }
    }
    return grad_input;
}

std::shared_ptr<Tensor> RepeatInterleaveForward(const std::shared_ptr<Tensor> &input, int64_t repeat, int64_t dim) {
    CHECK_GT(repeat, 0);
    CHECK_GE(dim, 0);
    CHECK_LT(dim, input->Dims().size());

    const auto &input_dims = input->Dims();
    const int64_t outer = std::accumulate(input_dims.begin(), input_dims.begin() + dim, 1, std::multiplies<int64_t>());
    const int64_t inner
        = std::accumulate(input_dims.begin() + dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());
    const int64_t dim_size = input_dims[dim];

    std::vector<int64_t> output_dims = input_dims;
    output_dims[dim] = dim_size * repeat;
    auto output = std::make_shared<Tensor>(output_dims, input->Dtype(), input->GetDevice());

    const float *input_ptr = static_cast<const float *>(input->DataPtr());
    float *output_ptr = static_cast<float *>(output->DataPtr());

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < dim_size; ++i) {
            for (int r = 0; r < repeat; ++r) {
                std::memcpy(output_ptr + ((o * dim_size * repeat + i * repeat + r) * inner),
                            input_ptr + ((o * dim_size + i) * inner), sizeof(float) * inner);
            }
        }
    }

    return output;
}

std::shared_ptr<Tensor> RepeatInterleaveBackward(const std::shared_ptr<Tensor> &grad_output,
                                                 const std::vector<int64_t> &input_dims, int64_t dim) {
    CHECK_GE(dim, 0);
    CHECK_LT(dim, input_dims.size());

    const int64_t outer = std::accumulate(input_dims.begin(), input_dims.begin() + dim, 1, std::multiplies<int64_t>());
    const int64_t inner
        = std::accumulate(input_dims.begin() + dim + 1, input_dims.end(), 1, std::multiplies<int64_t>());
    const int64_t dim_size = input_dims[dim];

    int repeat = grad_output->Dims()[dim] / dim_size;
    CHECK_EQ(grad_output->Dims()[dim], dim_size * repeat);

    auto grad_input = std::make_shared<Tensor>(input_dims, grad_output->Dtype(), grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);

    const float *grad_out_ptr = static_cast<const float *>(grad_output->DataPtr());
    float *grad_in_ptr = static_cast<float *>(grad_input->DataPtr());

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t i = 0; i < dim_size; ++i) {
            for (int64_t j = 0; j < inner; ++j) {
                float sum = 0.0f;
                for (int r = 0; r < repeat; ++r) {
                    sum += grad_out_ptr[((o * dim_size * repeat + i * repeat + r) * inner) + j];
                }
                grad_in_ptr[(o * dim_size + i) * inner + j] = sum;
            }
        }
    }

    return grad_input;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_TRANSFORM_KERNEL(kernel_name)                                                                     \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_TRANSFORM_KERNEL(TrilForward)
REGISTER_CPU_TRANSFORM_KERNEL(TrilBackward)
REGISTER_CPU_TRANSFORM_KERNEL(TriuForward)
REGISTER_CPU_TRANSFORM_KERNEL(TriuBackward)
REGISTER_CPU_TRANSFORM_KERNEL(TransposeForward)
REGISTER_CPU_TRANSFORM_KERNEL(TransposeBackward)
REGISTER_CPU_TRANSFORM_KERNEL(MaskForward)
REGISTER_CPU_TRANSFORM_KERNEL(MaskBackward)
REGISTER_CPU_TRANSFORM_KERNEL(RepeatInterleaveForward)
REGISTER_CPU_TRANSFORM_KERNEL(RepeatInterleaveBackward)

#undef REGISTER_CPU_TRANSFORM_KERNEL
