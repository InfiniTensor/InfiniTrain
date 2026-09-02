#include "infini_train/include/nn/modules/transformer/utils.h"

#include <memory>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"

namespace infini_train {
std::shared_ptr<Tensor> PrecomputeFreqsCis(int64_t dim, int64_t end, float theta, bool use_scaled, Device device,
                                           const std::vector<float> &freq_factors) {
    auto dtype = DataType::kFLOAT32;
    CHECK_GE(dim, 2) << "dim must be >= 2 for slicing";

    auto arange = nn::init::Arange(0, dim, dtype, device)->Slice(0, 0, dim, 2);
    auto freqs = 1.0f / nn::function::Pow(theta, arange / float(dim));
    if (!freq_factors.empty()) {
        CHECK_EQ(static_cast<int64_t>(freq_factors.size()), dim / 2);
        Tensor factors_cpu(freq_factors.data(), std::vector<int64_t>{dim / 2}, dtype);
        auto factors = std::make_shared<Tensor>(factors_cpu.To(device));
        freqs = freqs / factors;
    }
    // TODO(zbl): use_scaled
    // if (use_scaled) {
    //     freqs = ApplyScaling(freqs, 8192.0f);
    // }
    auto t = nn::init::Arange(0, end, dtype, device);
    // (end, dim / 2)
    auto freqs_outer = t->Outer(freqs);
    auto cos = nn::function::Cos(freqs_outer);
    auto sin = nn::function::Sin(freqs_outer);
    // NOTE(zbl): torch script uses cis expression, here use stack
    // (end, dim / 2, 2)
    auto freqs_cis = nn::function::Stack(std::vector<std::shared_ptr<Tensor>>{cos, sin}, -1)->Contiguous();

    return freqs_cis;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
ApplyFM9GRotaryEmbedding(const std::shared_ptr<Tensor> &xq, const std::shared_ptr<Tensor> &xk,
                         const std::shared_ptr<Tensor> &freqs_cis) {
    const int64_t T = xq->Dims()[1];
    const int64_t D = xq->Dims()[3];
    CHECK(xq->Dtype() == xk->Dtype()) << "FM9G rotary Q/K dtype mismatch";
    auto cis = std::make_shared<Tensor>(freqs_cis->To(xq->Dtype()))->View({1, T, 1, D / 2, 2});
    auto cos_half = cis->Slice(-1, 0, 1, 1)->Squeeze(-1);
    auto sin_half = cis->Slice(-1, 1, 2, 1)->Squeeze(-1);
    auto cos = nn::function::Concat(std::vector<std::shared_ptr<Tensor>>{cos_half, cos_half}, -1);
    auto sin = nn::function::Concat(std::vector<std::shared_ptr<Tensor>>{sin_half, sin_half}, -1);
    auto rotate_half = [D](const std::shared_ptr<Tensor> &x) {
        auto first = x->Slice(-1, 0, D / 2);
        auto second = x->Slice(-1, D / 2, D);
        return nn::function::Concat(std::vector<std::shared_ptr<Tensor>>{-second, first}, -1);
    };
    return {xq * cos + rotate_half(xq) * sin, xk * cos + rotate_half(xk) * sin};
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
ApplyRotaryEmbedding(const std::shared_ptr<Tensor> &xq, const std::shared_ptr<Tensor> &xk,
                     const std::shared_ptr<Tensor> &freqs_cis) {
    const auto &x_shape = xq->Dims(); // (B, T, H, D)
    const int64_t T = x_shape[1];
    const int64_t D = x_shape[3];

    std::vector<int64_t> target_shape = {1, T, 1, D / 2, 2};
    auto cos_sin = freqs_cis->View(target_shape); // -> (1, T, 1, D/2, 2)

    auto cos = cos_sin->Slice(-1, 0, 1, 1)->Squeeze(-1); // (1, T, 1, D/2)
    auto sin = cos_sin->Slice(-1, 1, 2, 1)->Squeeze(-1); // (1, T, 1, D/2)

    auto slice_pair = [](const std::shared_ptr<Tensor> &x) {
        auto even = x->Slice(-1, 0, x->Dims().back(), 2);
        auto odd = x->Slice(-1, 1, x->Dims().back(), 2);
        return std::make_pair(even, odd);
    };

    auto [q_even, q_odd] = slice_pair(xq);
    auto q_rotated_left = q_even * cos - q_odd * sin;
    auto q_rotated_right = q_even * sin + q_odd * cos;
    auto q_rotated
        = nn::function::Stack(std::vector<std::shared_ptr<Tensor>>{q_rotated_left, q_rotated_right}, -1)->Flatten(-2);

    auto [k_even, k_odd] = slice_pair(xk);
    auto k_rotated_left = k_even * cos - k_odd * sin;
    auto k_rotated_right = k_even * sin + k_odd * cos;
    auto k_rotated
        = nn::function::Stack(std::vector<std::shared_ptr<Tensor>>{k_rotated_left, k_rotated_right}, -1)->Flatten(-2);

    return {q_rotated, k_rotated};
}
} // namespace infini_train
