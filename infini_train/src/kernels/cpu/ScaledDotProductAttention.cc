#include <cmath>
#include <cstdint>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {

static inline double DefaultScale(const std::shared_ptr<Tensor> &k /*(B,H,T,D)*/) {
    const auto &kdims = k->Dims();
    CHECK(!kdims.empty());
    const int64_t D = kdims.back();
    return 1.0 / std::sqrt(static_cast<double>(D));
}

static std::shared_ptr<Tensor> BuildCausalMaskLike(const std::shared_ptr<Tensor> & /*scores*/) { return nullptr; }

static inline std::shared_ptr<Tensor> CallSoftmaxForward(const std::shared_ptr<Tensor> &input, int64_t dim) {
    auto dev = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({dev, "SoftmaxForward"});

    return kernel.template Call<std::shared_ptr<Tensor>>(input, dim);
}

std::shared_ptr<Tensor>
ScaledDotProductAttentionForward(const std::shared_ptr<Tensor> &q, const std::shared_ptr<Tensor> &k,
                                 const std::shared_ptr<Tensor> &v, const std::shared_ptr<Tensor> &attn_mask,
                                 bool is_causal, double dropout_p, bool scale_has, double scale_val, bool enable_gqa) {
    (void)dropout_p;
    (void)enable_gqa;

    const double scale = scale_has ? scale_val : DefaultScale(k);

    // scores = scale * (Q @ K^T)  -> (B,H,Tq,Tk)
    std::shared_ptr<Tensor> scores = q->Matmul(k->Transpose(-2, -1)) * scale;

    if (attn_mask) {
        scores = scores + attn_mask;
    } else if (is_causal) {
        std::shared_ptr<Tensor> causal = BuildCausalMaskLike(scores);
        if (causal) {
            scores = scores + causal;
        }
    }

    std::shared_ptr<Tensor> probs = CallSoftmaxForward(scores, -1);

    // Y = P @ V  -> (B,H,Tq,D)
    std::shared_ptr<Tensor> y = probs->Matmul(v);
    return y;
}

std::vector<std::shared_ptr<Tensor>>
ScaledDotProductAttentionBackward(const std::shared_ptr<Tensor> &dY, const std::shared_ptr<Tensor> &Q,
                                  const std::shared_ptr<Tensor> &K, const std::shared_ptr<Tensor> &V,
                                  const std::shared_ptr<Tensor> &attn_mask, bool is_causal, double dropout_p,
                                  bool scale_has, double scale_val, bool enable_gqa) {
    (void)dropout_p;
    (void)enable_gqa;

    const double scale = scale_has ? scale_val : DefaultScale(K);

    std::shared_ptr<Tensor> scores = Q->Matmul(K->Transpose(-2, -1)) * scale;
    if (attn_mask) {
        scores = scores + attn_mask;
    } else if (is_causal) {
        std::shared_ptr<Tensor> causal = BuildCausalMaskLike(scores);
        if (causal) {
            scores = scores + causal;
        }
    }

    std::shared_ptr<Tensor> P = CallSoftmaxForward(scores, -1); // (B,H,Tq,Tk)

    // dV = P^T @ dY
    std::shared_ptr<Tensor> dV = P->Transpose(-2, -1)->Matmul(dY); // (B,H,Tk,D)

    // dP = dY @ V^T
    std::shared_ptr<Tensor> dP = dY->Matmul(V->Transpose(-2, -1)); // (B,H,Tq,Tk)

    // dS = (dP - (dP ⊙ P).sum(-1, keepdim)) ⊙ P
    const auto &pdims = P->Dims(); // P: (B,H,Tq,Tk)
    const int64_t Tk = pdims.back();
    auto ones = std::make_shared<Tensor>(std::vector<int64_t>{1, 1, Tk, 1}, P->Dtype(), P->GetDevice());
    ones->Fill<float>(1.0f);

    std::shared_ptr<Tensor> tmp = (dP * P)->Matmul(ones); // (B,H,Tq,1)
    std::shared_ptr<Tensor> dS = (dP - tmp)->Mul(P);      // (B,H,Tq,Tk)

    // dQ = (dS @ K) * scale
    std::shared_ptr<Tensor> dQ = dS->Matmul(K) * scale; // (B,H,Tq,D)

    // dK = (dS^T @ Q) * scale
    std::shared_ptr<Tensor> dK = dS->Transpose(-2, -1)->Matmul(Q) * scale; // (B,H,Tk,D)

    return {dQ, dK, dV};
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_SDPA_KERNEL(kernel_name)                                                                          \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_SDPA_KERNEL(ScaledDotProductAttentionForward);
REGISTER_CPU_SDPA_KERNEL(ScaledDotProductAttentionBackward);

#undef REGISTER_CPU_SDPA_KERNEL
