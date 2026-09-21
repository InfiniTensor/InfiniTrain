#include <cmath>
#include <memory>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RMSNormForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, const float eps) {
    /*
        x: [..., embed_dim]
        -> RMSNorm (w: [embed_dim])
        -> o: [..., embed_dim]
    */
    // The composite path (Mean(-1)/Pow/Rsqrt/Mul) supports any rank, so the fused kernel keeps the
    // same generality: one row per leading index, reducing over the last dimension.
    CHECK_GE(input->Dims().size(), 2);
    CHECK_EQ(input->Dims().back(), weight->Dims()[0]);
    CHECK(input->Dtype() == DataType::kFLOAT32 && weight->Dtype() == DataType::kFLOAT32);

    auto input_c = input->IsContiguous() ? input : input->Contiguous();

    const int embed_dim = static_cast<int>(input_c->Dims().back());
    const int64_t rows = input_c->NumElements() / embed_dim;

    auto output = std::make_shared<Tensor>(input_c->Dims(), DataType::kFLOAT32);
    auto rstd = std::make_shared<Tensor>(std::vector<int64_t>(input_c->Dims().begin(), input_c->Dims().end() - 1),
                                         DataType::kFLOAT32);

    for (int64_t t = 0; t < rows; ++t) {
        float sqsum = 0.0f;
        for (int i = 0; i < embed_dim; ++i) {
            float x = static_cast<float *>(input_c->DataPtr())[t * embed_dim + i];
            sqsum += x * x;
        }
        float s = 1.0f / sqrtf(sqsum / embed_dim + eps);

        for (int i = 0; i < embed_dim; ++i) {
            float x = static_cast<float *>(input_c->DataPtr())[t * embed_dim + i];
            float n = x * s;                                          // normalize
            float o = n * static_cast<float *>(weight->DataPtr())[i]; // scale
            static_cast<float *>(output->DataPtr())[t * embed_dim + i] = o;
        }
        // cache rstd for the backward pass later
        static_cast<float *>(rstd->DataPtr())[t] = s;
    }

    return {output, rstd};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
RMSNormBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                const std::shared_ptr<Tensor> &rstd, const std::shared_ptr<Tensor> &grad_output) {
    CHECK_GE(input->Dims().size(), 2);
    CHECK_EQ(input->Dims().back(), weight->Dims()[0]);
    CHECK_NE(rstd, nullptr);
    CHECK(input->Dtype() == DataType::kFLOAT32 && weight->Dtype() == DataType::kFLOAT32
          && grad_output->Dtype() == DataType::kFLOAT32);

    auto input_c = input->IsContiguous() ? input : input->Contiguous();

    const int embed_dim = static_cast<int>(input_c->Dims().back());
    const int64_t rows = input_c->NumElements() / embed_dim;

    auto grad_input = std::make_shared<Tensor>(input_c->Dims(), DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight->Dims(), DataType::kFLOAT32);
    // grad_weight accumulates across rows; grad_input rows are fully overwritten below.
    grad_weight->Fill(0.0);

    for (int64_t t = 0; t < rows; ++t) {
        float rstd_t = static_cast<float *>(rstd->DataPtr())[t];

        // S1 = sum_i(g_i * w_i * x_i); K = (S1 / H) * rstd
        float S1 = 0.0f;
        for (int i = 0; i < embed_dim; ++i) {
            float x = static_cast<float *>(input_c->DataPtr())[t * embed_dim + i];
            float w = static_cast<float *>(weight->DataPtr())[i];
            float g = static_cast<float *>(grad_output->DataPtr())[t * embed_dim + i];
            S1 += g * w * x;
        }
        float K = (S1 / embed_dim) * rstd_t;

        for (int i = 0; i < embed_dim; ++i) {
            float x = static_cast<float *>(input_c->DataPtr())[t * embed_dim + i];
            float w = static_cast<float *>(weight->DataPtr())[i];
            float g = static_cast<float *>(grad_output->DataPtr())[t * embed_dim + i];
            float n = x * rstd_t;
            static_cast<float *>(grad_input->DataPtr())[t * embed_dim + i] = (g * w - K * n) * rstd_t;
            static_cast<float *>(grad_weight->DataPtr())[i] += g * n;
        }
    }
    return {grad_input, grad_weight};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_RMSNORM_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_RMSNORM_KERNEL(RMSNormForward)
REGISTER_CPU_RMSNORM_KERNEL(RMSNormBackward)

#undef REGISTER_CPU_RMSNORM_KERNEL
