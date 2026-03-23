#include "infini_train/include/tensor.h"
#include "infini_train/include/device.h"
#include "infini_train/src/kernels/cuda/flash_attention.h"
#include <cuda_runtime.h>
#include <random>
#include <cmath>
#include <iostream>
#include <cassert>

namespace infini_train {

// Reference forward declaration (if not in header)
namespace kernels::cuda {
void FlashAttentionReferenceForward(const float* Q, const float* K, const float* V, float* O, float* L,
                                  int B, int H, int T, int D, float softmax_scale, bool is_causal, cudaStream_t stream);
}

// Minimal Test Framework since gtest is missing
#define EXPECT_GT(val, thresh) \
    if ((val) <= (thresh)) { \
        std::cerr << "Assertion failed: " << #val << " (" << (val) << ") > " << #thresh << " (" << (thresh) << ")" << std::endl; \
        std::exit(1); \
    }

class FlashAttentionPrecisionTest {
public:
    void SetUp() {
        // Initialize random seed
        std::srand(42);
    }

    // Helper to generate random tensor with specific condition number
    void GenerateRandomTensor(Tensor& t, float min_val, float max_val) {
        float* data = static_cast<float*>(t.DataPtr());
        int size = t.NumElements();
        for (int i = 0; i < size; ++i) {
            float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            data[i] = min_val + r * (max_val - min_val);
        }
    }
    
    // Check error
    void CheckError(const Tensor& ref, const Tensor& val, float max_rel_err_limit, const char* name) {
        const float* ref_data = static_cast<const float*>(ref.DataPtr());
        const float* val_data = static_cast<const float*>(val.DataPtr());
        int size = ref.NumElements();
        
        float max_rel_err = 0.0f;
        float rms_err = 0.0f;
        int exceed_count = 0;
        
        for (int i = 0; i < size; ++i) {
            float r = ref_data[i];
            float v = val_data[i];
            float diff = std::abs(r - v);
            float rel_err = diff / (std::abs(r) + 1e-6f); // Avoid div by zero
            
            if (rel_err > max_rel_err) max_rel_err = rel_err;
            rms_err += diff * diff;
            
            if (rel_err > max_rel_err_limit) {
                exceed_count++;
                if (exceed_count < 5) {
                    printf("Mismatch at %d: ref=%f, val=%f, rel_err=%e\n", i, r, v, rel_err);
                }
            }
        }
        
        rms_err = std::sqrt(rms_err / size);
        
        printf("[%s] Max Rel Err: %e (Limit: %e), RMS Err: %e\n", name, max_rel_err, max_rel_err_limit, rms_err);
        
        // 99.9% check
        float pass_ratio = 1.0f - (float)exceed_count / size;
        EXPECT_GT(pass_ratio, 0.999f);
    }
    
    void Run() {
        SetUp();
        ForwardFP32Comparison();
        std::cout << "Test Passed!" << std::endl;
    }

    void ForwardFP32Comparison() {
        int B = 1;
        int H = 4;
        int T = 128;
        int D = 64;
        
        Device device(Device::DeviceType::kCUDA, 0);
        
        // Tensor constructor: dims, dtype, device
        Tensor q({B, H, T, D}, DataType::kFLOAT32, device);
        Tensor k({B, H, T, D}, DataType::kFLOAT32, device);
        Tensor v({B, H, T, D}, DataType::kFLOAT32, device);
        Tensor o({B, H, T, D}, DataType::kFLOAT32, device);
        Tensor l({B, H, T}, DataType::kFLOAT32, device);
        
        // Host Tensors for init
        Tensor q_cpu({B, H, T, D}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
        Tensor k_cpu({B, H, T, D}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
        Tensor v_cpu({B, H, T, D}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
        
        GenerateRandomTensor(q_cpu, -1.0f, 1.0f);
        GenerateRandomTensor(k_cpu, -1.0f, 1.0f);
        GenerateRandomTensor(v_cpu, -1.0f, 1.0f);
        
        q.CopyFrom(q_cpu);
        k.CopyFrom(k_cpu);
        v.CopyFrom(v_cpu);
        
        // Reference output
        Tensor o_ref({B, H, T, D}, DataType::kFLOAT32, device);
        Tensor l_ref({B, H, T}, DataType::kFLOAT32, device);
        
        // Run Reference
        float softmax_scale = 1.0f / std::sqrt((float)D);
        kernels::cuda::FlashAttentionReferenceForward(
            (float*)q.DataPtr(), (float*)k.DataPtr(), (float*)v.DataPtr(),
            (float*)o_ref.DataPtr(), (float*)l_ref.DataPtr(),
            B, H, T, D, softmax_scale, true, 0
        );
        
        // Run FlashAttention
        kernels::cuda::FlashAttentionForward(q, k, v, o, l, 0.0f, softmax_scale, true, device);
        
        // Check
        Tensor o_host({B, H, T, D}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
        Tensor o_ref_host({B, H, T, D}, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
        
        o_host.CopyFrom(o);
        o_ref_host.CopyFrom(o_ref);
        
        CheckError(o_ref_host, o_host, 1e-5f, "Forward Output");
    }

};

} // namespace infini_train

int main() {
    infini_train::FlashAttentionPrecisionTest test;
    test.Run();
    return 0;
}
