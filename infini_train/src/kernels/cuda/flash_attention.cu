#include "infini_train/src/kernels/cuda/flash_attention.h"

#include <cuda_runtime.h>
#include <iostream>

#include "glog/logging.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

// TODO: Implement actual FlashAttention CUDA kernel
// For now, we will just print a message and maybe do a naive copy or fill to avoid crashes

__global__ void FlashAttentionForwardKernelStub(const float* q, const float* k, const float* v, float* output,
                                                int B, int T, int H, int D) {
    // Placeholder kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * T * H * D;
    if (idx < total_elements) {
        // Just copy Q to output as a dummy operation so output is not uninitialized
        output[idx] = q[idx]; 
    }
}

void FlashAttentionForward(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &output, Tensor &softmax_lse,
                           float dropout_p, float softmax_scale, bool is_causal, const Device &device) {
    // LOG(INFO) << "Launching FlashAttentionForward Kernel (Stub)";

    // Get dimensions
    auto dims = q.Dims(); // (B, T, H, D)
    int B = dims[0];
    int T = dims[1];
    int H = dims[2];
    int D = dims[3];

    // Check data type - only supporting float32 for stub
    if (q.GetDType() != DataType::kFLOAT32) {
        LOG(WARNING) << "FlashAttention stub currently only supports float32";
        return;
    }

    const float* q_ptr = q.data<float>();
    const float* k_ptr = k.data<float>();
    const float* v_ptr = v.data<float>();
    float* out_ptr = output.mutable_data<float>();

    // Launch stub kernel
    int total_elements = B * T * H * D;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    
    // Ensure we are on the correct stream if provided (device.stream)
    // TODO: get stream from Device object correctly
    cudaStream_t stream = 0; 
    
    FlashAttentionForwardKernelStub<<<blocks, threads, 0, stream>>>(q_ptr, k_ptr, v_ptr, out_ptr, B, T, H, D);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG(ERROR) << "CUDA Error in FlashAttentionForward: " << cudaGetErrorString(err);
    }
}

} // namespace infini_train::kernels::cuda
