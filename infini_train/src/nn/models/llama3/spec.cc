#include "infini_train/include/models/llama3/spec.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/config.h"
#include "infini_train/include/nn/modules/transformer/spec.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;
namespace nn = infini_train::nn;

namespace {
std::shared_ptr<Tensor> PrecomputeFreqsCis(int64_t dim, int64_t end, float theta = 10000.0f, bool use_scaled = false,
                                           const infini_train::Device *device
                                           = DeviceManager::Instance()->GetDefaultDevice()) {
    DataType dtype = DataType::kFLOAT32;
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

} // namespace

namespace infini_train::nn {

LLaMA3ChunkABI::LLaMA3ChunkABI(const TransformerConfig &config, int start_layer, int end_layer,
                               std::shared_ptr<TransformerKernel> kernel)
    : TransformerChunkABI(config, start_layer, end_layer, std::move(kernel)) {
    std::vector<std::shared_ptr<nn::Module>> layers;

    for (int i = start_layer; i < end_layer; ++i) { layers.push_back(kernel->MakeBlock(config_)); }

    // HF ABI: transformer.h = ModuleList
    modules_[kHLayerName] = std::make_shared<nn::ModuleList>(std::move(layers));
}

std::vector<std::shared_ptr<Tensor>> LLaMA3ChunkABI::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    auto x1 = x[0];
    const auto device = x1->GetDevice();

    if (buffers_[kFreqsCisName] == nullptr) {
        buffers_[kFreqsCisName] = PrecomputeFreqsCis(config_.n_embd / config_.n_head, config_.block_size * 2,
                                                     config_.rope_theta, config_.use_scaled_rope, device);
    }

    const auto t = x1->Dims()[1] * nn::parallel::global::GetSequenceParallelSize();

    int64_t start_pos = 0;
    auto freqs_view = buffers_[kFreqsCisName]->Slice(0, start_pos, start_pos + t, 1);

    auto ones = std::make_shared<Tensor>(nn::function::Ones({t, t})->To(device));
    auto mask = nn::function::Triu(ones, 1)->View({1, 1, t, t});

    std::shared_ptr<Tensor> start_pos_ptr = nullptr;

    for (auto &h : *std::dynamic_pointer_cast<nn::ModuleList>(modules_[kHLayerName])) {
        x1 = h->Forward({x1, freqs_view, start_pos_ptr, mask})[0];
    }
    return {x1};
}

} // namespace infini_train::nn