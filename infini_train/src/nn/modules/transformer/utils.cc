#include "infini_train/include/nn/modules/transformer/utils.h"

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
} // namespace infini_train
