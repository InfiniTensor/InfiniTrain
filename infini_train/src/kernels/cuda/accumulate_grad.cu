#include "infini_train/include/kernels/cuda/accumulate_grad.h"

#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = reinterpret_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = reinterpret_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}

__global__ void AdamAccumulateGradKernel(const float *grad_data, float *param_data, size_t num_elements, float *m_data,
                                         float *v_data, float learning_rate, float beta1, float beta2, float eps,
                                         const float bias_correction_m, const float bias_correction_v) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        m_data[idx] = __fmaf_rn(beta1, m_data[idx], (1 - beta1) * grad_data[idx]);
        v_data[idx] = __fmaf_rn(beta2, v_data[idx], (1 - beta2) * grad_data[idx] * grad_data[idx]);

        const float m_hat = __fdiv_rn(m_data[idx], bias_correction_m);
        const float v_hat = __fdiv_rn(v_data[idx], bias_correction_v);

        param_data[idx] -= learning_rate * m_hat * __frcp_rn(__fsqrt_rn(v_hat) + eps);
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    size_t num_elements = grad->NumElements();

    const float *grad_data = reinterpret_cast<const float *>(grad->DataPtr());
    float *m_data = reinterpret_cast<float *>(m->DataPtr());
    float *v_data = reinterpret_cast<float *>(v->DataPtr());
    float *param_data = reinterpret_cast<float *>(param->DataPtr());

    const float bias_correction_m = 1.0f - std::pow(beta1, t);
    const float bias_correction_v = 1.0f - std::pow(beta2, t);

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_data, param_data, num_elements, m_data, v_data,
                                                                learning_rate, beta1, beta2, eps, bias_correction_m,
                                                                bias_correction_v);
}
} // namespace infini_train::kernels::cuda
