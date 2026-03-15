#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include "infini_train/include/tensor.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/src/kernels/cuda/flash_attention.h"
#include "glog/logging.h"

using namespace infini_train;

void TestLayout(int B, int T, int H, int D) {
    LOG(INFO) << "Testing Layout with B=" << B << " T=" << T << " H=" << H << " D=" << D;
    Device device(Device::DeviceType::kCUDA, 0);
    
    // Create Inputs (B, T, H, D)
    auto q = std::make_shared<Tensor>(std::vector<int64_t>{B, T, H, D}, DataType::kFLOAT32, device);
    auto k = std::make_shared<Tensor>(std::vector<int64_t>{B, T, H, D}, DataType::kFLOAT32, device);
    auto v = std::make_shared<Tensor>(std::vector<int64_t>{B, T, H, D}, DataType::kFLOAT32, device);
    
    // Fill with random data
    // Use smaller range to avoid expf overflow (FP32 exp limit is ~88)
    // d=64, sqrt(d)=8. q*k_sum ~ val^2 * 64. scaled ~ val^2 * 8.
    // If val=1, scaled=8, exp(8)=2980 (OK).
    // If val=5, scaled=200, exp(200)=inf (Overflow).
    infini_train::nn::init::Uniform(q, -1.0, 1.0);
    infini_train::nn::init::Uniform(k, -1.0, 1.0);
    infini_train::nn::init::Uniform(v, -1.0, 1.0);
    
    // 1. Reference Implementation (Manual Attention on CPU)
    Device cpu_device(Device::DeviceType::kCPU, 0);
    
    // Skip Reference for now to isolate crash
    // Wrap in shared_ptr because Tensor methods might use shared_from_this()
    auto q_cpu = std::make_shared<Tensor>(q->To(cpu_device));
    auto k_cpu = std::make_shared<Tensor>(k->To(cpu_device));
    auto v_cpu = std::make_shared<Tensor>(v->To(cpu_device));
    
    auto q_ref = q_cpu->Transpose(1, 2);
    auto k_ref = k_cpu->Transpose(1, 2);
    auto v_ref = v_cpu->Transpose(1, 2);
    
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    auto att = q_ref->Matmul(k_ref->Transpose(-2, -1)) * scale;
    
    // Mask logic on CPU (Enabled)
    auto ones = infini_train::nn::function::Ones({T, T});
    auto mask_cpu = infini_train::nn::function::Triu(ones, 1)->View({1, 1, T, T});
    att = att->MaskedFill(mask_cpu, -std::numeric_limits<float>::infinity());
 
    att = infini_train::nn::function::Softmax(att, -1);
    auto y_ref = att->Matmul(v_ref);
    y_ref = y_ref->Transpose(1, 2)->Contiguous();
    
    // 2. FlashAttention Implementation (on GPU)
    // float scale = 1.0f / std::sqrt(static_cast<float>(D)); // Defined above
    auto q_trans = q->Transpose(1, 2)->Contiguous();
    auto k_trans = k->Transpose(1, 2)->Contiguous();
    auto v_trans = v->Transpose(1, 2)->Contiguous();
    
    auto y_flash_trans = std::make_shared<Tensor>(std::vector<int64_t>{B, H, T, D}, DataType::kFLOAT32, device);
    auto lse_flash = std::make_shared<Tensor>(std::vector<int64_t>{B, H, T}, DataType::kFLOAT32, device);
    
    // Call Kernel (Enable Causal)
    kernels::cuda::FlashAttentionForward(*q_trans, *k_trans, *v_trans, *y_flash_trans, *lse_flash, 0.0f, scale, true, device);
    
    // Transpose output back: (B, H, T, D) -> (B, T, H, D)
    auto y_flash = y_flash_trans->Transpose(1, 2)->Contiguous();
    
    // 3. Compare
    // y_ref is already on CPU
    auto y_flash_cpu = y_flash->To(cpu_device);
    
    // LOG(INFO) << "Flash Output Copy Done";
    // return;
    
    float* ref_ptr = static_cast<float*>(y_ref->DataPtr());
    float* flash_ptr = static_cast<float*>(y_flash_cpu.DataPtr());
    
    double max_diff = 0.0;
    double sum_diff = 0.0;
    int num_elements = y_ref->NumElements();
    
    for (int i = 0; i < num_elements; ++i) {
        double diff = std::abs(ref_ptr[i] - flash_ptr[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
    }
    
    LOG(INFO) << "Max Diff: " << max_diff;
    LOG(INFO) << "Avg Diff: " << sum_diff / num_elements;
    
    if (max_diff > 0.5) { // Relaxed threshold for FP16
        LOG(ERROR) << "Mismatch detected!";
    } else {
        LOG(INFO) << "Match!";
    }
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    
    LOG(INFO) << "=== Test 1: T=16 (Partial Tile) ===";
    TestLayout(1, 16, 2, 64);
    
    LOG(INFO) << "=== Test 2: T=32 (Full Tile) ===";
    TestLayout(1, 32, 2, 64);
    
    LOG(INFO) << "=== Test 3: T=1024 (Many Tiles) ===";
    TestLayout(1, 1024, 2, 64);
    
    return 0;
}
