#include "infini_train/include/kernels/cpu/transform.h"

#include <cmath>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

std::shared_ptr<Tensor> TrilForward(const std::shared_ptr<Tensor> &input, int64_t diagonal) {
    CHECK_EQ(input->Dims().size(), 2);

    auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    for (int i = 0; i < input->NumElements(); ++i) {
        int64_t row = i / input->Dims()[1];
        int64_t col = i % input->Dims()[1];
        if (row - col + diagonal >= 0) {
            reinterpret_cast<float *>(output->DataPtr())[i] = reinterpret_cast<float *>(input->DataPtr())[i];
        } else {
            reinterpret_cast<float *>(output->DataPtr())[i] = 0.0;
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
            reinterpret_cast<float *>(grad_input->DataPtr())[i] = reinterpret_cast<float *>(grad_output->DataPtr())[i];
        } else {
            reinterpret_cast<float *>(grad_input->DataPtr())[i] = 0.0;
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

    const float *in_ptr = reinterpret_cast<const float *>(input->DataPtr());
    float *out_ptr = reinterpret_cast<float *>(output->DataPtr());

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

    const float *in_ptr = reinterpret_cast<const float *>(input->DataPtr());

    for (int i = 0; i < input->NumElements(); ++i) {
        // TODO(dcj): use bool mask when dtype is enabled.
        if (reinterpret_cast<const float *>(mask->DataPtr())[i % mask->NumElements()] == 1.0) {
            reinterpret_cast<float *>(output->DataPtr())[i] = value;
        } else {
            reinterpret_cast<float *>(output->DataPtr())[i] = in_ptr[i];
        }
    }
    return output;
}

std::shared_ptr<Tensor> MaskBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &mask) {
    CHECK_EQ(grad_output->NumElements() % mask->NumElements(), 0);
    CHECK_EQ(static_cast<int>(grad_output->Dtype()), static_cast<int>(mask->Dtype()));
    auto grad_input = std::make_shared<Tensor>(grad_output->Dims(), grad_output->Dtype(), grad_output->GetDevice());

    for (int i = 0; i < grad_output->NumElements(); ++i) {
        if (reinterpret_cast<const float *>(mask->DataPtr())[i % mask->NumElements()] == 1.0) {
            reinterpret_cast<float *>(grad_input->DataPtr())[i] = 0.0;
        } else {
            reinterpret_cast<float *>(grad_input->DataPtr())[i]
                = reinterpret_cast<const float *>(grad_output->DataPtr())[i];
        }
    }
    return grad_input;
}
} // namespace infini_train::kernels::cpu
