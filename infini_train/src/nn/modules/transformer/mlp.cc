#include "infini_train/include/nn/modules/transformer/mlp.h"

#include <cmath>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/spec_utils.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

MLP::MLP(const TransformerConfig &config, const ModuleSpec &spec) : CloneableModule(kType) {
    // c_fc: ColumnParallel (input full, output parallel)
    modules_[kCFcLayerName] = BuildModule(config, spec.submodules_.at(kCFcLayerName));

    // For SwiGLU, add second projection
    if (spec.submodules_.contains(kCFc2LayerName)) {
        modules_[kCFc2LayerName] = BuildModule(config, spec.submodules_.at(kCFc2LayerName));
    }

    // Activation: check for GELU or SwiGLU
    if (spec.submodules_.contains(kGeluLayerName)) {
        modules_[kGeluLayerName] = BuildModule(config, spec.submodules_.at(kGeluLayerName));
    } else if (spec.submodules_.contains(kSiluLayerName)) {
        modules_[kSiluLayerName] = BuildModule(config, spec.submodules_.at(kSiluLayerName));
    }

    // c_proj: RowParallel (input parallel, output full)
    modules_[kCProjLayerName] = BuildModule(config, spec.submodules_.at(kCProjLayerName));
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MLP::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    bool is_swiglu = modules_.contains(kCFc2LayerName) && modules_.contains(kSiluLayerName);

    if (is_swiglu) {
        // SwiGLU forward pass
        // (B, T, C) -> ColumnParallelLinear(C, hidden_dim) -> (B, T, hidden_dim)
        auto x1 = (*modules_[kCFcLayerName])(x)[0];
        // (B, T, C) -> ColumnParallelLinear(C, hidden_dim) -> (B, T, hidden_dim)
        auto x2 = (*modules_[kCFc2LayerName])(x)[0];
        // (B, T, hidden_dim) -> SiLU -> (B, T, hidden_dim)
        x2 = (*modules_[kSiluLayerName])({x2})[0];
        // (B, T, hidden_dim) -> element-wise mul -> (B, T, hidden_dim)
        auto x3 = x1 * x2;
        // (B, T, hidden_dim) -> RowParallelLinear(hidden_dim, C) -> (B, T, C)
        auto x4 = (*modules_[kCProjLayerName])({x3});
        return x4;
    } else {
        // GELU forward pass (standard)
        // (B, T, C) -> ColumnParallelLinear(C, 4*C) -> (B, T, 4*C_local)
        auto x1 = (*modules_[kCFcLayerName])(x);
        // (B, T, 4*C_local) -> GELU -> (B, T, 4*C_local)
        auto x2 = (*modules_[kGeluLayerName])(x1);
        // (B, T, 4*C_local) -> RowParallelLinear(4*C, C) -> (B, T, C)
        auto x3 = (*modules_[kCProjLayerName])(x2);
        return x3;
    }
}

} // namespace infini_train::nn
