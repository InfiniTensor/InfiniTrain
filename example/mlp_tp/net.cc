#include "example/mlp_tp/net.h"

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

namespace nn = infini_train::nn;

std::vector<std::shared_ptr<infini_train::Tensor>>
NewGELU::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto &input = x[0];
    return {0.5 * input
            * (1.0 + nn::function::Tanh(std::sqrt(2.0 / M_PI) * (input + 0.044715 * nn::function::Pow(input, 3.0))))};
}

TensorParallelMLP::TensorParallelMLP(int64_t n_embd,
                                     int64_t hidden_dim,
                                     const infini_train::nn::parallel::TensorParallelGroup &tp_group,
                                     bool fc_bias,
                                     bool proj_bias)
    : hidden_dim_(hidden_dim),
      tp_group_(tp_group) {
    CHECK_GT(n_embd, 0);
    CHECK_GT(hidden_dim, 0);
    CHECK_GT(tp_group_.WorldSize(), 0) << "TP group not initialized";

    // c_fc：ColumnParallelLinear
    // - 不 gather 输出（保持最后一维分片），便于就地 GELU
    // - 输入默认视为非并行（input_is_parallel=false），由模块内部做 autograd-aware copy
    // - 不做 bias fusion（skip_bias_add=false）
    modules_[kCFcLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
        /*in_features=*/n_embd,
        /*out_features=*/hidden_dim_,
        /*bias=*/fc_bias,
        /*tp_group=*/tp_group_,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false);

    // gelu
    modules_[kGeluLayerName] = std::make_shared<NewGELU>();

    // c_proj：RowParallelLinear
    // - 输入已经是并行的（来自 c_fc 分片输出），input_is_parallel=true
    // - 末端 reduce 输出到完整（reduce_output=true）
    // - 不做 bias fusion（skip_bias_add=false）
    modules_[kCProjLayerName] = std::make_shared<nn::parallel::RowParallelLinear>(
        /*in_features=*/hidden_dim_,
        /*out_features=*/n_embd,
        /*bias=*/proj_bias,
        /*tp_group=*/tp_group_,
        /*reduce_output=*/true,
        /*input_is_parallel=*/true,
        /*skip_bias_add=*/false);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
TensorParallelMLP::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1) << "TensorParallelMLP takes exactly one input tensor";
    auto input = x[0];

    // x -> c_fc (sharded last-dim)
    auto h = modules_.at(kCFcLayerName)->Forward({input})[0];

    // h(sharded) -> gelu (elementwise, local)
    h = modules_.at(kGeluLayerName)->Forward({h})[0];

    // h(sharded) -> c_proj -> (reduced to full)
    auto out = modules_.at(kCProjLayerName)->Forward({h})[0];

    return {out};
}
