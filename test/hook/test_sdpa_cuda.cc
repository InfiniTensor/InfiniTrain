#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

namespace {
float MaxAbsDiffCPU(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    CHECK(a != nullptr);
    CHECK(b != nullptr);
    auto a_cpu = std::make_shared<Tensor>(a->To(Device(Device::DeviceType::kCPU, 0)));
    auto b_cpu = std::make_shared<Tensor>(b->To(Device(Device::DeviceType::kCPU, 0)));
    CHECK(a_cpu->Dtype() == DataType::kFLOAT32);
    CHECK(b_cpu->Dtype() == DataType::kFLOAT32);
    CHECK_EQ(a_cpu->NumElements(), b_cpu->NumElements());

    const float *pa = static_cast<const float *>(a_cpu->DataPtr());
    const float *pb = static_cast<const float *>(b_cpu->DataPtr());
    float m = 0.0f;
    for (size_t i = 0; i < a_cpu->NumElements(); ++i) { m = std::max(m, std::abs(pa[i] - pb[i])); }
    return m;
}

std::shared_ptr<Tensor> ReferenceSDPA(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                      const std::shared_ptr<Tensor> &v, bool is_causal) {
    const auto &q_shape = q->Dims();
    CHECK_EQ(q_shape.size(), 4);
    const int64_t T = q_shape[2];
    const int64_t D = q_shape[3];
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    auto scores = q->Matmul(k->Transpose(-2, -1)) * scale;

    if (is_causal) {
        // mask: (1,1,T,T) upper triangle excluding diag
        auto ones_cpu = nn::function::Ones({T, T});
        auto ones = std::make_shared<Tensor>(ones_cpu->To(q->GetDevice()));
        auto causal = nn::function::Triu(ones, 1)->View({1, 1, T, T});
        scores = scores->MaskedFill(causal, std::numeric_limits<float>::lowest());
    }

    auto probs = nn::function::Softmax(scores, -1);
    return probs->Matmul(v);
}
} // namespace

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1);

#ifndef USE_CUDA
    std::cout << "SKIP: USE_CUDA is OFF" << std::endl;
    return 0;
#else
    Device cuda(Device::DeviceType::kCUDA, 0);

    const int64_t B = 1;
    const int64_t H = 2;
    const int64_t T = 4;
    const int64_t D = 8;

    std::vector<int64_t> dims = {B, H, T, D};

    // deterministic host data
    std::vector<float> hq(B * H * T * D);
    std::vector<float> hk(B * H * T * D);
    std::vector<float> hv(B * H * T * D);
    for (size_t i = 0; i < hq.size(); ++i) {
        hq[i] = 0.01f * static_cast<float>(i % 17);
        hk[i] = 0.02f * static_cast<float>((i + 3) % 19);
        hv[i] = 0.03f * static_cast<float>((i + 7) % 23);
    }

    // Fused path (ScaledDotProductAttention -> autograd -> CUDA kernel)
    auto q1 = std::make_shared<Tensor>(hq.data(), dims, DataType::kFLOAT32, cuda)->RequiresGrad();
    auto k1 = std::make_shared<Tensor>(hk.data(), dims, DataType::kFLOAT32, cuda)->RequiresGrad();
    auto v1 = std::make_shared<Tensor>(hv.data(), dims, DataType::kFLOAT32, cuda)->RequiresGrad();

    auto out1 = nn::function::ScaledDotProductAttention(q1, k1, v1, nullptr, 0.0, true);
    auto loss1 = out1->Sum(-1, false)->Sum(-1, false)->Sum(-1, false)->Sum(-1, false);
    loss1->Backward();

    // Reference path (matmul + causal mask + softmax + matmul)
    auto q2 = std::make_shared<Tensor>(hq.data(), dims, DataType::kFLOAT32, cuda)->RequiresGrad();
    auto k2 = std::make_shared<Tensor>(hk.data(), dims, DataType::kFLOAT32, cuda)->RequiresGrad();
    auto v2 = std::make_shared<Tensor>(hv.data(), dims, DataType::kFLOAT32, cuda)->RequiresGrad();

    auto out2 = ReferenceSDPA(q2, k2, v2, true);
    auto loss2 = out2->Sum(-1, false)->Sum(-1, false)->Sum(-1, false)->Sum(-1, false);
    loss2->Backward();

    const float out_diff = MaxAbsDiffCPU(out1, out2);
    const float dq_diff = MaxAbsDiffCPU(q1->grad(), q2->grad());
    const float dk_diff = MaxAbsDiffCPU(k1->grad(), k2->grad());
    const float dv_diff = MaxAbsDiffCPU(v1->grad(), v2->grad());

    std::cout << "MaxAbsDiff(out) = " << out_diff << std::endl;
    std::cout << "MaxAbsDiff(dq)  = " << dq_diff << std::endl;
    std::cout << "MaxAbsDiff(dk)  = " << dk_diff << std::endl;
    std::cout << "MaxAbsDiff(dv)  = " << dv_diff << std::endl;

    const float tol = 5e-3f;
    if (out_diff < tol && dq_diff < tol && dk_diff < tol && dv_diff < tol) {
        std::cout << "PASS" << std::endl;
        return 0;
    }
    std::cout << "FAIL" << std::endl;
    return 1;
#endif
}
