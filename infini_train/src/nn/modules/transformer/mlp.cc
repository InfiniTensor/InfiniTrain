#include "infini_train/include/nn/modules/transformer/mlp.h"

#include <stdint.h>

#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/lora/lora_parallel_linear.h"
#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/transformer/moe/moe_utils.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"

namespace infini_train::nn {

MLP::MLP(const TransformerConfig &config, bool is_expert)
    : CloneableModule(kType) {
  int64_t ffn_hidden = config.ffn_hidden_size.value_or(4 * config.n_embd);

  if (config.ffn_type == FFNType::kMoE) {
    const auto &moe_config = moe::RequireMoEConfig(config);
    CHECK_GT(moe_config.moe_ffn_hidden_size, 0);

    ffn_hidden = moe_config.moe_ffn_hidden_size;
  }
  CHECK_GT(ffn_hidden, 0);

  // c_fc: ColumnParallel (input full, output parallel)
  RegisterModule(kCFcLayerName,
                 std::make_shared<parallel::ColumnParallelLinear>(
                     /*in_features=*/config.n_embd,
                     /*out_features=*/config.activation_type == MLPType::kSwiGLU
                         ? 2 * ffn_hidden
                         : ffn_hidden,
                     /*bias=*/config.add_bias_linear,
                     /*gather_output=*/false,
                     /*input_is_parallel=*/false,
                     /*skip_bias_add=*/false,
                     /*sequence_parallel=*/
                     parallel::global::GetSequenceParallelEnabled()));

  // Activation: check for GELU or SwiGLU
  if (config.activation_type == MLPType::kGELU) {
    RegisterModule(kGeluLayerName, std::make_shared<NewGELU>());
  } else if (config.activation_type == MLPType::kSwiGLU) {
    RegisterModule(kSwiGLULayerName, std::make_shared<SwiGLU>());
  }

  // c_proj: RowParallel (input parallel, output full)
  RegisterModule(kCProjLayerName,
                 std::make_shared<nn::parallel::RowParallelLinear>(
                     /*in_features=*/ffn_hidden, /*out_features=*/config.n_embd,
                     /*bias=*/config.add_bias_linear,
                     /*reduce_output=*/true,
                     /*input_is_parallel=*/true,
                     /*skip_bias_add=*/false,
                     /*sequence_parallel=*/
                     nn::parallel::global::GetSequenceParallelEnabled()));

  const bool use_expert_process_groups =
      is_expert && (parallel::global::GetExpertParallelSize() > 1 ||
                    parallel::global::GetExpertTensorParallelSize() !=
                        parallel::global::GetTensorParallelSize());
  if (use_expert_process_groups) {
    // MLP is not yet owned by a shared_ptr while its constructor runs, so
    // traverse the already-constructed child modules instead of calling
    // Module::Parameters() on this object.
    for (const auto &[_, module] : modules_) {
      for (const auto &parameter : module->Parameters()) {
        parameter->set_is_expert_parallel(true);
      }
    }
  }
}

std::vector<std::shared_ptr<infini_train::Tensor>> MLP::Forward(
    const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
  if (modules_.contains(kSwiGLULayerName)) {
    // (B, T, C) -> ColumnParallelLinear(C, 2*H) -> (B, T, 2*H_local)
    auto packed = (*modules_[kCFcLayerName])(x)[0];
    // (B, T, 2*H_local) [gate, up] -> SwiGLU -> (B, T, H_local)
    auto activated = (*modules_[kSwiGLULayerName])({packed});
    // (B, T, H_local) -> RowParallelLinear(H, C) -> (B, T, C)
    return (*modules_[kCProjLayerName])(activated);
  }

  // GELU forward pass (standard)
  // (B, T, C) -> ColumnParallelLinear(C, 4*C) -> (B, T, 4*C_local)
  auto x1 = (*modules_[kCFcLayerName])(x);
  // (B, T, 4*C_local) -> GELU -> (B, T, 4*C_local)
  auto x2 = (*modules_[kGeluLayerName])(x1);
  // (B, T, 4*C_local) -> RowParallelLinear(4*C, C) -> (B, T, C)
  return (*modules_[kCProjLayerName])(x2);
}

ShardedStateDict MLP::BuildShardedStateDict(
    const ShardedStateDictBuildContext &context) const {
  auto state = Module::BuildShardedStateDict(context);
  if (!modules_.contains(kSwiGLULayerName)) {
    return state;
  }

  const auto c_fc_prefix = context.logical_prefix.empty()
                               ? kCFcLayerName
                               : context.logical_prefix + "." + kCFcLayerName;
  const auto weight_key =
      c_fc_prefix + "." + parallel::ColumnParallelLinear::kParamWeightName;
  const auto bias_key =
      c_fc_prefix + "." + parallel::ColumnParallelLinear::kParamBiasName;
  const auto lora_b_key =
      c_fc_prefix + "." + lora::LoRAColumnParallelLinear::kParamLoraBName;
  const auto weight_it = state.tensors.find(weight_key);
  CHECK(weight_it != state.tensors.end())
      << "Missing packed SwiGLU weight metadata: " << weight_key;

  const auto &weight = weight_it->second;
  CHECK(!weight.global_shape.empty() && !weight.local_shape.empty());
  CHECK_EQ(weight.global_shape[0] % 2, 0);
  CHECK_EQ(weight.local_shape[0] % 2, 0);
  const int64_t global_rows_per_projection = weight.global_shape[0] / 2;
  const int64_t local_rows_per_projection = weight.local_shape[0] / 2;
  const int rank = parallel::tp_rank;

  for (auto &[key, tensor] : state.tensors) {
    if (key != weight_key && key != bias_key && key != lora_b_key) {
      continue;
    }
    CHECK(!tensor.global_shape.empty() && !tensor.local_shape.empty());
    CHECK_EQ(tensor.global_shape[0], weight.global_shape[0]);
    CHECK_EQ(tensor.local_shape[0], weight.local_shape[0]);
    tensor.global_offset.assign(tensor.global_shape.size(), 0);
    tensor.segments = {
        {.global_offset = rank * local_rows_per_projection,
         .local_offset = 0,
         .length = local_rows_per_projection},
        {.global_offset =
             global_rows_per_projection + rank * local_rows_per_projection,
         .local_offset = local_rows_per_projection,
         .length = local_rows_per_projection},
    };
  }
  return state;
}

}  // namespace infini_train::nn
