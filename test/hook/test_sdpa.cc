#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;
namespace fn = infini_train::nn::function;

float MaxAbsDiff(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    CHECK(a != nullptr);
    CHECK(b != nullptr);

    auto a_cpu = std::make_shared<Tensor>(a->To(Device()));
    auto b_cpu = std::make_shared<Tensor>(b->To(Device()));

    CHECK_EQ(static_cast<int>(a_cpu->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(static_cast<int>(b_cpu->Dtype()), static_cast<int>(DataType::kFLOAT32));
    CHECK_EQ(a_cpu->NumElements(), b_cpu->NumElements());

    const auto *pa = static_cast<const float *>(a_cpu->DataPtr());
    const auto *pb = static_cast<const float *>(b_cpu->DataPtr());

    float max_diff = 0.0f;
    for (size_t i = 0; i < a_cpu->NumElements(); ++i) {
        max_diff = std::max(max_diff, std::abs(pa[i] - pb[i]));
    }
    return max_diff;
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1);

    // Simple deterministic SDPA correctness check on CPU.
    // q/k/v: (B=1, H=1, T=4, D=2)
    const std::vector<int64_t> dims = {1, 1, 4, 2};

    const float q_data[] = {
        0.1f, 0.2f,
        0.3f, 0.4f,
        0.5f, 0.6f,
        0.7f, 0.8f,
    };

    const float k_data[] = {
        0.2f, 0.1f,
        0.4f, 0.3f,
        0.6f, 0.5f,
        0.8f, 0.7f,
    };

    const float v_data[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
        0.5f, 0.5f,
    };

    auto q = std::make_shared<Tensor>(q_data, dims, DataType::kFLOAT32, Device());
    auto k = std::make_shared<Tensor>(k_data, dims, DataType::kFLOAT32, Device());
    auto v = std::make_shared<Tensor>(v_data, dims, DataType::kFLOAT32, Device());

    const int64_t T = dims[2];
    const int64_t D = dims[3];
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Reference (manual) causal SDPA
    auto att = q->Matmul(k->Transpose(-2, -1)) * scale;
    auto ones = std::make_shared<Tensor>(fn::Ones({T, T})->To(Device()));
    auto causal_mask = fn::Triu(ones, 1)->View({1, 1, T, T});
    att = att->MaskedFill(causal_mask, std::numeric_limits<float>::lowest());
    att = fn::Softmax(att, -1);
    auto y_ref = att->Matmul(v);

    // API under test
    auto y = fn::ScaledDotProductAttention(q, k, v, /*attn_mask=*/nullptr,
                                          /*dropout_p=*/0.0, /*is_causal=*/true);

    const float diff = MaxAbsDiff(y, y_ref);
    std::cout << "MaxAbsDiff = " << diff << std::endl;

    if (diff > 1e-4f) {
        std::cerr << "FAIL: SDPA output mismatch" << std::endl;
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
