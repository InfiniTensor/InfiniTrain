#include <cmath>

#include "cub/block/block_reduce.cuh"
#include "cuda_runtime.h"
#include "math_constants.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/common/cuda/kernel_helper.cuh"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

template <size_t BLOCK_SIZE, typename Tinput, typename Tmask, typename Tindex>
__global__ void
VocabParallelCrossEntropyBackwardKernel(const Tinput *__restrict__ softmax_local,   // [rows, V_local]
                                        Tinput *__restrict__ grad_input,            // [rows, V_local]
                                        const Tindex *__restrict__ masked_target,   // [rows]
                                        const Tmask *__restrict__ target_mask_row,  // [rows]，0/1
                                        const Tmask *__restrict__ valid_mask_local, // [rows, V_local] or [1, V_local]
                                        const Tinput *__restrict__ dloss_buf,       // [1] or [rows]
                                        int rows, int V_local,
                                        int dloss_is_scalar,             // 1=scalaer，0=by row
                                        float one_minus_label_smoothing, // 1 - label_smoothing
                                        float smoothing_term             // label_smoothing / vocab_size_original
) {
    const int r = blockIdx.x;
    if (r >= rows) {
        return;
    }

    const float dm = common::cuda::Cast<float>(dloss_is_scalar ? dloss_buf[0] : dloss_buf[r]);
    const float vm_row = 1.0f - common::cuda::Cast<float>(target_mask_row[r]);
    const float row_scale = dm * one_minus_label_smoothing * vm_row;
    const Tindex t = masked_target[r];

    for (int j = threadIdx.x; j < V_local; j += BLOCK_SIZE) {
        const int idx = r * V_local + j;

        const float s = common::cuda::Cast<float>(softmax_local[idx]);
        const float vm = common::cuda::Cast<float>(valid_mask_local[j]);

        float grad = dm * s;

        if (t >= 0 && j == static_cast<int>(t)) {
            grad -= row_scale;
        }

        grad -= dm * smoothing_term * vm;
        grad *= vm;

        grad_input[idx] = common::cuda::Cast<Tinput>(grad);
    }
}

std::shared_ptr<Tensor>
VocabParallelCrossEntropyBackward(const std::shared_ptr<Tensor> &grad_output,      // [rows]
                                  const std::shared_ptr<Tensor> &softmax_local,    // [rows, V_local]
                                  const std::shared_ptr<Tensor> &target_mask,      // [rows]
                                  const std::shared_ptr<Tensor> &masked_target,    // [rows]，int64
                                  const std::shared_ptr<Tensor> &valid_mask_local, // [1, V_local]
                                  const int64_t vocab_size_local, const int64_t vocab_size_original,
                                  float label_smoothing) {

    const int64_t rows = softmax_local->NumElements() / vocab_size_local;
    CHECK_EQ(masked_target->NumElements(), rows);
    CHECK_EQ(target_mask->NumElements(), rows);
    CHECK_EQ(valid_mask_local->NumElements(), vocab_size_local);

    int dloss_is_scalar = 0;
    if (grad_output->Dims().size() == 0) {
        dloss_is_scalar = 1;
    } else {
        CHECK(grad_output->NumElements() == rows || grad_output->NumElements() == 1)
            << "grad_output must be scalar or length rows";
        dloss_is_scalar = (grad_output->NumElements() == 1);
    }

    const auto *cuda_device = dynamic_cast<const CudaDevice *>(grad_output->GetDevice());

    // logits should be [rows, V_local]
    auto grad_input = std::make_shared<Tensor>(softmax_local->Dims(), softmax_local->Dtype(), cuda_device);

    const float one_minus_label_smoothing = 1.0f - label_smoothing;
    const float smoothing_term = (label_smoothing > 0.f && vocab_size_original > 0)
                                   ? (label_smoothing / static_cast<float>(vocab_size_original))
                                   : 0.0f;

    constexpr int threads_per_block = 256;
    const int num_blocks = static_cast<int>(rows);

    DispatchFunc<DataTypeList<DataType::kUINT8, DataType::kINT64>, DataTypeList<INFINI_ALL_FLOATING_TYPES>>(
        {masked_target->Dtype(), softmax_local->Dtype()},
        [=]<typename Tindex, typename Tinput>() {
            using Tmask = Tinput;

            const Tinput *softmax_ptr = static_cast<const Tinput *>(softmax_local->DataPtr());
            const Tmask *tmask_ptr = static_cast<const Tmask *>(target_mask->DataPtr());
            const Tmask *vml_ptr = static_cast<const Tmask *>(valid_mask_local->DataPtr());
            const Tindex *mtarget_ptr = static_cast<const Tindex *>(masked_target->DataPtr());
            const Tinput *grad_output_ptr = static_cast<const Tinput *>(grad_output->DataPtr());
            Tinput *grad_input_ptr = static_cast<Tinput *>(grad_input->DataPtr());

            VocabParallelCrossEntropyBackwardKernel<threads_per_block, Tinput, Tmask, Tindex>
                <<<num_blocks, threads_per_block, 0, cuda_device->Stream()>>>(
                    softmax_ptr, grad_input_ptr, mtarget_ptr, tmask_ptr, vml_ptr, grad_output_ptr,
                    static_cast<int>(rows), static_cast<int>(vocab_size_local), dloss_is_scalar,
                    one_minus_label_smoothing, smoothing_term);
        },
        "CUDA VocabParallelCrossEntropyBackward");

    return grad_input;
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_VOCAB_PARALLEL_CROSS_ENTROPY_KERNEL(kernel_name)                                                 \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_VOCAB_PARALLEL_CROSS_ENTROPY_KERNEL(VocabParallelCrossEntropyBackward)

#undef REGISTER_CUDA_CROSS_ENTROPY_KERNEL
