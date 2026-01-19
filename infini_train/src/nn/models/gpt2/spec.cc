#include "infini_train/include/models/gpt2/spec.h"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/models/gpt2/gpt2.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/config.h"
#include "infini_train/include/nn/modules/transformer/spec.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

// std::vector<std::shared_ptr<Tensor>> GPT2ChunkABI::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
//     auto h = x[0];
//     auto &layers = *std::dynamic_pointer_cast<nn::ModuleList>(modules_[kHLayerName]);

//     for (auto &layer : layers) { h = layer->Forward({h})[0]; }
//     return {h};
// }

} // namespace infini_train::nn