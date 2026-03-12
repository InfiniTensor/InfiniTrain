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
    infini_train::nn::init::Uniform(q, -1.0, 1.0);
    infini_train::nn::init::Uniform(k, -1.0, 1.0);
    infini_train::nn::init::Uniform(v, -1.0, 1.0);
    
    // 1. Reference Implementation (Manual Attention using Tensor Ops)
    // Matches "Baseline" in net.cc
    // q, k, v: (B, T, H, D)
    // -> (B, H, T, D)
    auto q_ref = q->Transpose(1, 2);
    auto k_ref = k->Transpose(1, 2);
    auto v_ref = v->Transpose(1, 2);
    
    // Attn = Softmax(Q @ K.T / sqrt(D)) @ V
    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    auto att = q_ref->Matmul(k_ref->Transpose(-2, -1)) * scale;
    // Causal Mask (Full lower triangular)
    // Create mask (1, 1, T, T)
    auto ones = infini_train::nn::function::Ones({T, T});
    // Triu -> View -> To -> make_shared
    auto mask_cpu = infini_train::nn::function::Triu(ones, 1)->View({1, 1, T, T});
    auto mask = std::make_shared<Tensor>(mask_cpu->To(device));
    att = att->MaskedFill(mask, -std::numeric_limits<float>::infinity());
    
    att = infini_train::nn::function::Softmax(att, -1);
    auto y_ref = att->Matmul(v_ref);
    // (B, H, T, D) -> (B, T, H, D)
    y_ref = y_ref->Transpose(1, 2)->Contiguous();
    
    // 2. FlashAttention Implementation
    // Kernel expects (B, H, T, D) layout.
    // So we must Transpose inputs: (B, T, H, D) -> (B, H, T, D)
    auto q_trans = q->Transpose(1, 2)->Contiguous();
    auto k_trans = k->Transpose(1, 2)->Contiguous();
    auto v_trans = v->Transpose(1, 2)->Contiguous();
    
    auto y_flash_trans = std::make_shared<Tensor>(std::vector<int64_t>{B, H, T, D}, DataType::kFLOAT32, device);
    auto lse_flash = std::make_shared<Tensor>(std::vector<int64_t>{B, H, T}, DataType::kFLOAT32, device);
    
    // Verify v_trans data
    auto v_trans_cpu = v_trans->To(Device(Device::DeviceType::kCPU, 0));
    float* v_ptr = static_cast<float*>(v_trans_cpu.DataPtr());
    std::cout << "v_trans[0]=" << v_ptr[0] << ", v_trans[1]=" << v_ptr[1] << ", v_trans[2]=" << v_ptr[2] << std::endl;
    
    // Call Kernel
    kernels::cuda::FlashAttentionForward(*q_trans, *k_trans, *v_trans, *y_flash_trans, *lse_flash, 0.0f, scale, true, device);
    
    // Transpose output back: (B, H, T, D) -> (B, T, H, D)
    auto y_flash = y_flash_trans->Transpose(1, 2)->Contiguous();
    
    // 3. Compare
    // Move to CPU
    auto y_ref_cpu = y_ref->To(Device(Device::DeviceType::kCPU, 0));
    auto y_flash_cpu = y_flash->To(Device(Device::DeviceType::kCPU, 0));
    
    float* ref_ptr = static_cast<float*>(y_ref_cpu.DataPtr());
    float* flash_ptr = static_cast<float*>(y_flash_cpu.DataPtr());
    
    double max_diff = 0.0;
    double sum_diff = 0.0;
    int num_elements = y_ref->NumElements();
    
    std::cout << "First 10 elements (Ref vs Flash):" << std::endl;
    for (int i = 0; i < num_elements; ++i) {
        double diff = std::abs(ref_ptr[i] - flash_ptr[i]);
        max_diff = std::max(max_diff, diff);
        sum_diff += diff;
        if (i < 10) {
            std::cout << "i=" << i << ": " << ref_ptr[i] << " vs " << flash_ptr[i] << " (diff=" << diff << ")" << std::endl;
        }
    }
    
    std::cout << "Max Diff: " << max_diff << std::endl;
    std::cout << "Avg Diff: " << sum_diff / num_elements << std::endl;
    
    if (max_diff > 1e-3) {
        LOG(ERROR) << "Mismatch detected!";
    } else {
        LOG(INFO) << "Match!";
    }
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    // Test Case: B=1, T=4, H=2, D=64 (Small)
    TestLayout(1, 4, 2, 64);
    return 0;
}
