#include "infini_train/include/nn/modules/transformer/utils.h"

#include <memory>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"

namespace infini_train {
std::shared_ptr<Tensor> PrecomputeFreqsCis(int64_t dim, int64_t end, float theta, bool use_scaled, Device device) {
    auto dtype = DataType::kFLOAT32;
    CHECK_GE(dim, 2) << "dim must be >= 2 for slicing";

    auto arange = nn::init::Arange(0, dim, dtype, device)->Slice(0, 0, dim, 2);
    auto freqs = 1.0f / nn::function::Pow(theta, arange / float(dim));
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
ApplyRotaryEmbedding(const std::shared_ptr<Tensor> &xq, const std::shared_ptr<Tensor> &xk,
                     const std::shared_ptr<Tensor> &freqs_cis, bool rotary_interleaved) {
    const auto &x_shape = xq->Dims(); // (B, T, H, D)
    const int64_t T = x_shape[1];
    const int64_t D = x_shape[3];

    std::vector<int64_t> target_shape = {1, T, 1, D / 2, 2};
    auto cos_sin = freqs_cis->View(target_shape); // -> (1, T, 1, D/2, 2)

    auto cos = cos_sin->Slice(-1, 0, 1, 1)->Squeeze(-1); // (1, T, 1, D/2)
    auto sin = cos_sin->Slice(-1, 1, 2, 1)->Squeeze(-1); // (1, T, 1, D/2)

    auto slice_pair = [rotary_interleaved](const std::shared_ptr<Tensor> &x) {
        const auto dim = x->Dims().back();
        if (rotary_interleaved) {
            return std::make_pair(x->Slice(-1, 0, dim, 2), x->Slice(-1, 1, dim, 2));
        }
        return std::make_pair(x->Slice(-1, 0, dim / 2), x->Slice(-1, dim / 2, dim));
    };

    auto rotate = [&](const std::shared_ptr<Tensor> &x) {
        auto [left, right] = slice_pair(x);
        auto rotated_left = left * cos - right * sin;
        auto rotated_right = left * sin + right * cos;
        if (rotary_interleaved) {
            return nn::function::Stack(std::vector<std::shared_ptr<Tensor>>{rotated_left, rotated_right}, -1)
                ->Flatten(-2);
        }
        return nn::function::Concat(std::vector<std::shared_ptr<Tensor>>{rotated_left, rotated_right}, -1);
    };

    return {rotate(xq), rotate(xk)};
}
} // namespace infini_train
