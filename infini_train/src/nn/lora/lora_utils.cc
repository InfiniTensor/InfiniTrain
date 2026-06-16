#include "infini_train/include/nn/lora/lora_utils.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/lora/lora_linear.h"
#include "infini_train/include/nn/lora/lora_parallel_linear.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::lora {

namespace {

enum class LoRATensorSharding {
    // The tensor is identical on every TP rank, so save/load can copy it as-is.
    kReplicated,
    // LoRA-B for a ColumnParallelLinear: local shape is [out/tp, rank].
    kColumnParallelDim0,
    // Attention QKV LoRA-B: dim0-sharded like ColumnParallel, but each local
    // shard is packed as [Qi | Ki | Vi] instead of a simple contiguous slice.
    kPackedQKVColumnParallelDim0,
    // LoRA-A for a RowParallelLinear: local shape is [rank, in/tp].
    kRowParallelDim1,
};

bool EndsWith(const std::string &value, const std::string &suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string LoraANameForLoraB(const std::string &name) {
    const std::string lora_b = LoRAColumnParallelLinear::kParamLoraBName;
    CHECK(EndsWith(name, lora_b)) << "Expected LoRA B parameter name, got " << name;
    return name.substr(0, name.size() - lora_b.size()) + LoRAColumnParallelLinear::kParamLoraAName;
}

std::string QualifiedParamName(const std::string &module_name, const std::string &param_name) {
    return module_name.empty() ? param_name : module_name + "." + param_name;
}

std::vector<std::string>
SortedLoRAStateDictNames(const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) {
    std::vector<std::string> names;
    names.reserve(state_dict.size());
    for (const auto &[name, _] : state_dict) { names.push_back(name); }
    std::sort(names.begin(), names.end());
    return names;
}

std::unordered_map<std::string, LoRATensorSharding> BuildLoRATensorShardings(const std::shared_ptr<Module> &model) {
    std::unordered_map<std::string, LoRATensorSharding> shardings;

    // Sharding metadata is keyed by the same qualified parameter names used by
    // LoRAStateDict(), so save can decide whether a local tensor must be gathered.
    auto named_modules = model->NamedModules(/*memory=*/nullptr, /*prefix=*/"", /*remove_duplicate=*/false);
    for (const auto &[module_name, module] : named_modules) {
        if (dynamic_cast<LoRAColumnParallelLinear *>(module.get())) {
            shardings[QualifiedParamName(module_name, LoRAColumnParallelLinear::kParamLoraAName)]
                = LoRATensorSharding::kReplicated;
            shardings[QualifiedParamName(module_name, LoRAColumnParallelLinear::kParamLoraBName)]
                = LoRATensorSharding::kColumnParallelDim0;
        } else if (dynamic_cast<LoRARowParallelLinear *>(module.get())) {
            shardings[QualifiedParamName(module_name, LoRARowParallelLinear::kParamLoraAName)]
                = LoRATensorSharding::kRowParallelDim1;
            shardings[QualifiedParamName(module_name, LoRARowParallelLinear::kParamLoraBName)]
                = LoRATensorSharding::kReplicated;
        } else if (dynamic_cast<LoRALinear *>(module.get())) {
            shardings[QualifiedParamName(module_name, LoRALinear::kParamLoraAName)] = LoRATensorSharding::kReplicated;
            shardings[QualifiedParamName(module_name, LoRALinear::kParamLoraBName)] = LoRATensorSharding::kReplicated;
        }
    }

    // Packed QKV is a property of the attention module topology, not the
    // parameter name. Mark the LoRA-B of the attention QKV projection explicitly.
    for (const auto &[module_name, module] : named_modules) {
        if (!dynamic_cast<CausalSelfAttention *>(module.get())) {
            continue;
        }

        auto qkv_projection = module->mutable_module(CausalSelfAttention::kCAttnLayerName);
        if (!dynamic_cast<LoRAColumnParallelLinear *>(qkv_projection.get())) {
            continue;
        }

        const auto qkv_module_name = QualifiedParamName(module_name, CausalSelfAttention::kCAttnLayerName);
        shardings[QualifiedParamName(qkv_module_name, LoRAColumnParallelLinear::kParamLoraBName)]
            = LoRATensorSharding::kPackedQKVColumnParallelDim0;
    }

    return shardings;
}

LoRATensorSharding GetLoRATensorSharding(const std::unordered_map<std::string, LoRATensorSharding> &shardings,
                                         const std::string &name) {
    auto it = shardings.find(name);
    return it == shardings.end() ? LoRATensorSharding::kReplicated : it->second;
}

std::shared_ptr<Tensor> GatherTensorParallelShard(const std::shared_ptr<Tensor> &tensor, int64_t dim) {
    const int tp_size = parallel::global::GetTensorParallelSize();
    CHECK_GT(tp_size, 0);
    if (tp_size == 1) {
        return tensor;
    }

    if (dim < 0) {
        dim += static_cast<int64_t>(tensor->Dims().size());
    }
    CHECK_GE(dim, 0);
    CHECK_LT(dim, static_cast<int64_t>(tensor->Dims().size()));

    auto device = tensor->GetDevice();
    auto *tp_group = parallel::ProcessGroupFactory::Instance(device.type())
                         ->Get(parallel::GetTensorParallelProcessGroupName(device.Rank().GlobalRank()));

    std::vector<int64_t> gathered_dims = tensor->Dims();
    gathered_dims[0] *= tp_size;
    auto gathered = std::make_shared<Tensor>(gathered_dims, tensor->Dtype(), device);
    tp_group->AllGather(gathered, tensor, false);

    if (dim == 0) {
        return gathered;
    }

    // AllGather stacks shards along dim 0. For tensors sharded on another dim,
    // split rank-major rows back into per-rank tensors and concatenate on dim.
    auto rank_major_shards = gathered->Split(tensor->Dims()[0], 0);
    return nn::function::Concat(rank_major_shards, dim)->Contiguous();
}

bool IsPrimarySaveTPGroupRank() {
    int dp = 0;
    int tp = 0;
    int pp = 0;
    parallel::global::GetCoordOf(parallel::global::thread_global_rank, dp, tp, pp);
    return dp == 0 && pp == 0;
}

bool IsSaveWriterRank() {
    int dp = 0;
    int tp = 0;
    int pp = 0;
    parallel::global::GetCoordOf(parallel::global::thread_global_rank, dp, tp, pp);
    return dp == 0 && tp == 0 && pp == 0;
}

std::shared_ptr<Tensor>
ExportLoRATensorForSave(const std::string &name, const std::shared_ptr<Tensor> &tensor,
                        const std::unordered_map<std::string, std::shared_ptr<Tensor>> &local_state_dict,
                        const std::unordered_map<std::string, LoRATensorSharding> &shardings) {
    const auto sharding = GetLoRATensorSharding(shardings, name);
    switch (sharding) {
    case LoRATensorSharding::kReplicated:
        return tensor;
    case LoRATensorSharding::kPackedQKVColumnParallelDim0:
        // Packed QKV is still dim0-sharded like ColumnParallel; it only needs
        // an extra Q/K/V reorder after the common gather below.
    case LoRATensorSharding::kColumnParallelDim0: {
        auto gathered = GatherTensorParallelShard(tensor, 0);
        if (sharding != LoRATensorSharding::kPackedQKVColumnParallelDim0) {
            return gathered;
        }

        const auto lora_a_name = LoraANameForLoraB(name);
        auto lora_a_it = local_state_dict.find(lora_a_name);
        CHECK(lora_a_it != local_state_dict.end())
            << "SaveLoRAWeights: cannot infer packed QKV shape without " << lora_a_name;
        const auto &lora_a_dims = lora_a_it->second->Dims();
        CHECK_EQ(lora_a_dims.size(), 2) << "SaveLoRAWeights: unexpected lora_A shape for " << lora_a_name;
        // Packed QKV LoRA-B is stored locally as [Qi | Ki | Vi], but adapter files
        // use the full packed order [Q | K | V] to match base QKV checkpoints.
        return detail::RestorePackedQKVRowsFromTensorParallel(gathered, /*q_rows=*/lora_a_dims[1],
                                                              parallel::global::GetTensorParallelSize());
    }
    case LoRATensorSharding::kRowParallelDim1:
        return GatherTensorParallelShard(tensor, 1);
    }
    LOG(FATAL) << "Unknown LoRA tensor sharding";
    return tensor;
}

void LoadLoRATensorIntoModel(const std::string &name, const std::shared_ptr<Tensor> &src,
                             const std::unordered_map<std::string, std::shared_ptr<Tensor>> &model_state_dict,
                             const std::unordered_map<std::string, LoRATensorSharding> &shardings) {
    auto it = model_state_dict.find(name);
    if (it == model_state_dict.end()) {
        LOG(WARNING) << "LoRA parameter not found in model: " << name;
        return;
    }

    auto &dst = it->second;
    const auto &src_dims = src->Dims();
    const auto &dst_dims = dst->Dims();
    if (dst_dims == src_dims) {
        dst->CopyFrom(src);
        return;
    }

    const int tp_size = parallel::global::GetTensorParallelSize();
    const auto sharding = GetLoRATensorSharding(shardings, name);
    if (sharding == LoRATensorSharding::kPackedQKVColumnParallelDim0) {
        const auto lora_a_name = LoraANameForLoraB(name);
        auto lora_a_it = model_state_dict.find(lora_a_name);
        CHECK(lora_a_it != model_state_dict.end())
            << "LoadLoRATensorIntoModel: cannot infer packed QKV shape without " << lora_a_name;
        const auto &lora_a_dims = lora_a_it->second->Dims();
        CHECK_EQ(lora_a_dims.size(), 2) << "LoadLoRATensorIntoModel: unexpected lora_A shape for " << lora_a_name;

        // Full adapter files store packed QKV LoRA-B as [Q | K | V]. Each TP rank
        // needs the matching local packed layout [Qi | Ki | Vi].
        auto sliced
            = detail::SlicePackedQKVRowsForTensorParallel(src, /*q_rows=*/lora_a_dims[1], parallel::tp_rank, tp_size);
        CHECK(sliced->Dims() == dst_dims) << "LoadLoRATensorIntoModel: packed QKV shard shape mismatch for " << name;
        dst->CopyFrom(sliced);
        return;
    }

    CHECK_EQ(src_dims.size(), dst_dims.size()) << "LoadLoRATensorIntoModel: rank mismatch for " << name;
    int shard_dim = -1;
    for (int d = 0; d < static_cast<int>(src_dims.size()); ++d) {
        if (dst_dims[d] != src_dims[d]) {
            shard_dim = d;
            break;
        }
    }
    CHECK(shard_dim >= 0) << "LoadLoRATensorIntoModel: shape mismatch for " << name << " but no differing dim found";
    CHECK_EQ(src_dims[shard_dim] % tp_size, 0)
        << "LoadLoRATensorIntoModel: sharded dimension is not divisible by TP size for " << name;

    const int64_t shard_size = src_dims[shard_dim] / tp_size;
    const int64_t start = static_cast<int64_t>(parallel::tp_rank) * shard_size;
    auto sliced = src->Slice(shard_dim, start, start + shard_size);
    CHECK(sliced->Dims() == dst_dims) << "LoadLoRATensorIntoModel: shard shape mismatch for " << name;
    dst->CopyFrom(sliced);
}

} // namespace

namespace detail {

std::shared_ptr<Tensor> SlicePackedQKVRowsForTensorParallel(const std::shared_ptr<Tensor> &full_tensor, int64_t q_rows,
                                                            int tp_rank, int tp_size) {
    CHECK(full_tensor != nullptr);

    const auto &dims = full_tensor->Dims();
    CHECK_GE(dims.size(), 1);
    CHECK_GT(tp_size, 0);
    CHECK_GE(tp_rank, 0);
    CHECK_LT(tp_rank, tp_size);
    CHECK_GT(q_rows, 0);

    const int64_t total_rows = dims[0];
    CHECK_GT(total_rows, q_rows) << "Packed QKV tensor must contain Q, K, and V rows";
    CHECK_EQ((total_rows - q_rows) % 2, 0) << "Packed QKV K/V rows must be balanced";

    const int64_t kv_rows = (total_rows - q_rows) / 2;
    CHECK_GT(kv_rows, 0);
    CHECK_EQ(q_rows % tp_size, 0) << "Q rows must be divisible by TP size";
    CHECK_EQ(kv_rows % tp_size, 0) << "K/V rows must be divisible by TP size";

    const int64_t q_local_rows = q_rows / tp_size;
    const int64_t kv_local_rows = kv_rows / tp_size;
    CHECK_GT(q_local_rows, 0);
    CHECK_GT(kv_local_rows, 0);

    auto q_shard = full_tensor->Slice(0, static_cast<int64_t>(tp_rank) * q_local_rows,
                                      static_cast<int64_t>(tp_rank + 1) * q_local_rows);
    auto k_shard = full_tensor->Slice(0, q_rows + static_cast<int64_t>(tp_rank) * kv_local_rows,
                                      q_rows + static_cast<int64_t>(tp_rank + 1) * kv_local_rows);
    auto v_shard = full_tensor->Slice(0, q_rows + kv_rows + static_cast<int64_t>(tp_rank) * kv_local_rows,
                                      q_rows + kv_rows + static_cast<int64_t>(tp_rank + 1) * kv_local_rows);

    return infini_train::nn::function::Concat({q_shard, k_shard, v_shard}, 0);
}

std::shared_ptr<Tensor> RestorePackedQKVRowsFromTensorParallel(const std::shared_ptr<Tensor> &gathered_tensor,
                                                               int64_t q_rows, int tp_size) {
    CHECK(gathered_tensor != nullptr);

    const auto &dims = gathered_tensor->Dims();
    CHECK_GE(dims.size(), 1);
    CHECK_GT(tp_size, 0);
    CHECK_GT(q_rows, 0);
    CHECK_EQ(dims[0] % tp_size, 0) << "Gathered packed QKV rows must be divisible by TP size";

    const int64_t local_rows = dims[0] / tp_size;
    CHECK_EQ(q_rows % tp_size, 0) << "Q rows must be divisible by TP size";
    const int64_t q_local_rows = q_rows / tp_size;
    CHECK_GT(local_rows, q_local_rows) << "Gathered packed QKV tensor must contain local Q, K, and V rows";
    CHECK_EQ((local_rows - q_local_rows) % 2, 0) << "Local packed QKV K/V rows must be balanced";
    const int64_t kv_local_rows = (local_rows - q_local_rows) / 2;
    CHECK_GT(kv_local_rows, 0);

    std::vector<std::shared_ptr<Tensor>> reordered_shards;
    reordered_shards.reserve(static_cast<size_t>(tp_size) * 3);
    for (int rank = 0; rank < tp_size; ++rank) {
        const int64_t base = static_cast<int64_t>(rank) * local_rows;
        reordered_shards.push_back(gathered_tensor->Slice(0, base, base + q_local_rows));
    }
    for (int rank = 0; rank < tp_size; ++rank) {
        const int64_t base = static_cast<int64_t>(rank) * local_rows;
        reordered_shards.push_back(gathered_tensor->Slice(0, base + q_local_rows, base + q_local_rows + kv_local_rows));
    }
    for (int rank = 0; rank < tp_size; ++rank) {
        const int64_t base = static_cast<int64_t>(rank) * local_rows;
        reordered_shards.push_back(
            gathered_tensor->Slice(0, base + q_local_rows + kv_local_rows, base + q_local_rows + 2 * kv_local_rows));
    }

    return nn::function::Concat(reordered_shards, 0);
}

} // namespace detail

std::shared_ptr<Module> GetLoRAModel(std::shared_ptr<Module> model, const LoRAConfig &config) {
    // In-place injection: modify the module tree directly
    // No wrapper, no base_model_, no extra layer
    model = InjectLoRALayers(model, config);

    // Freeze all non-LoRA parameters in the model
    // This is needed to freeze modules that are NOT LoRA targets (e.g., embedding, LayerNorm)
    // Uses StateDict() to get ALL parameters (not just trainable ones like Parameters())
    FreezeBaseModel(model);

    LOG(INFO) << "GetLoRAModel: Applied LoRA in-place, rank=" << config.rank << ", alpha=" << config.alpha;
    return model;
}

void ReplaceModuleByPath(std::shared_ptr<Module> model, const std::string &path, std::shared_ptr<Module> new_module) {
    // Parse the path (e.g., "transformer.h.0.attn.c_attn" -> ["transformer", "h", "0", "attn", "c_attn"])
    std::vector<std::string> parts;
    std::string remaining = path;
    size_t pos = 0;
    while ((pos = remaining.find('.')) != std::string::npos) {
        parts.push_back(remaining.substr(0, pos));
        remaining = remaining.substr(pos + 1);
    }
    parts.push_back(remaining);

    // Navigate to parent module
    std::shared_ptr<Module> current = model;
    for (size_t i = 0; i < parts.size() - 1; ++i) {
        current = current->mutable_module(parts[i]);
        if (!current) {
            LOG(ERROR) << "ReplaceModuleByPath: Failed to find path: " << path;
            return;
        }
    }

    // Replace the module
    const std::string &module_name = parts.back();
    current->mutable_module(module_name) = new_module;
}

std::shared_ptr<Module> InjectLoRALayers(std::shared_ptr<Module> model, const LoRAConfig &config) {
    // Use NamedModules() to automatically traverse the entire model hierarchy
    auto named_modules = model->NamedModules();

    int lora_layers_applied = 0;
    std::shared_ptr<Module> result = model;

    for (const auto &[name, module] : named_modules) {
        // Get module type
        auto type = module->type();

        // Check if this module should have LoRA applied
        // For root module (name.empty()), check if its type matches target_modules
        bool should_apply = name.empty() ? config.ShouldApplyLoRA(type) : config.ShouldApplyLoRA(name);
        if (!should_apply) {
            continue;
        }

        if (type == Linear::kType) {
            if (dynamic_cast<LoRALinear *>(module.get())) {
                continue;
            }
            if (name.empty()) {
                // Root module is Linear - create new LoRA module and return it
                result = std::make_shared<LoRALinear>(module, config);
            } else {
                auto lora_module = std::make_shared<LoRALinear>(module, config);
                ReplaceModuleByPath(model, name, lora_module);
            }
            lora_layers_applied++;
        } else if (type == parallel::ColumnParallelLinear::kType) {
            if (dynamic_cast<LoRAColumnParallelLinear *>(module.get())) {
                continue;
            }

            auto column_module = std::dynamic_pointer_cast<parallel::ColumnParallelLinear>(module);
            CHECK(column_module != nullptr) << "Failed to cast module to ColumnParallelLinear: " << name;

            if (name.empty()) {
                result = std::make_shared<LoRAColumnParallelLinear>(column_module, config);
            } else {
                auto lora_module = std::make_shared<LoRAColumnParallelLinear>(column_module, config);
                ReplaceModuleByPath(model, name, lora_module);
            }
            lora_layers_applied++;
        } else if (type == parallel::RowParallelLinear::kType) {
            if (dynamic_cast<LoRARowParallelLinear *>(module.get())) {
                continue;
            }

            auto row_module = std::dynamic_pointer_cast<parallel::RowParallelLinear>(module);
            CHECK(row_module != nullptr) << "Failed to cast module to RowParallelLinear: " << name;

            if (name.empty()) {
                result = std::make_shared<LoRARowParallelLinear>(row_module, config);
            } else {
                auto lora_module = std::make_shared<LoRARowParallelLinear>(row_module, config);
                ReplaceModuleByPath(model, name, lora_module);
            }
            lora_layers_applied++;
        }
    }

    LOG(INFO) << "InjectLoRALayers: Applied LoRA to " << lora_layers_applied << " layers "
              << "(rank=" << config.rank << ", alpha=" << config.alpha << ")";
    return result;
}

void FreezeBaseModel(std::shared_ptr<Module> model) {
    // Freeze all parameters by setting requires_grad to false
    model->Apply([](Module *m) {
        for (auto &[name, param] : m->StateDict()) { param->set_requires_grad(false); }
    });

    // Re-enable training for LoRA parameters (lora_A and lora_B)
    // This ensures LoRA weights remain trainable after freezing all parameters
    auto lora_params = GetLoRAParameters(model);
    for (auto &param : lora_params) { param->set_requires_grad(true); }
}

void UnfreezeModel(std::shared_ptr<Module> model) {
    model->Apply([](Module *m) {
        for (auto &[name, param] : m->StateDict()) { param->set_requires_grad(true); }
    });
}

std::vector<std::shared_ptr<Tensor>> GetLoRAParameters(const std::shared_ptr<Module> &model) {
    std::vector<std::shared_ptr<Tensor>> lora_params;
    std::unordered_set<const Tensor *> visited;

    auto collect_lora
        = [&lora_params, &visited](auto *lora_module) {
              CHECK(!lora_module->IsMerged())
                  << "GetLoRAParameters() called on merged LoRA module. Call UnmergeLoRAWeights() first.";
              for (auto &param : lora_module->LoRAParameters()) {
                  if (visited.insert(param.get()).second) {
                      lora_params.push_back(param);
                  }
              }
          };

    model->Apply([&collect_lora](Module *m) {
        if (auto lora_module = dynamic_cast<LoRALinear *>(m)) {
            collect_lora(lora_module);
        } else if (auto lora_module = dynamic_cast<LoRAColumnParallelLinear *>(m)) {
            collect_lora(lora_module);
        } else if (auto lora_module = dynamic_cast<LoRARowParallelLinear *>(m)) {
            collect_lora(lora_module);
        }
    });

    return lora_params;
}

std::vector<std::shared_ptr<Tensor>> GetBaseParameters(const std::shared_ptr<Module> &model) {
    std::vector<std::shared_ptr<Tensor>> base_params;

    for (auto &[name, param] : model->StateDict()) {
        // Skip LoRA parameters
        if (name.find(LoRALinear::kParamLoraAName) != std::string::npos
            || name.find(LoRALinear::kParamLoraBName) != std::string::npos) {
            continue;
        }
        base_params.push_back(param);
    }

    return base_params;
}

void MergeLoRAWeights(std::shared_ptr<Module> model) {
    model->Apply([](Module *m) {
        if (auto lora = dynamic_cast<LoRALinear *>(m)) {
            lora->MergeWeights();
        } else if (auto lora = dynamic_cast<LoRAColumnParallelLinear *>(m)) {
            lora->MergeWeights();
        } else if (auto lora = dynamic_cast<LoRARowParallelLinear *>(m)) {
            lora->MergeWeights();
        }
    });
}

void UnmergeLoRAWeights(std::shared_ptr<Module> model) {
    model->Apply([](Module *m) {
        if (auto lora = dynamic_cast<LoRALinear *>(m)) {
            lora->UnmergeWeights();
        } else if (auto lora = dynamic_cast<LoRAColumnParallelLinear *>(m)) {
            lora->UnmergeWeights();
        } else if (auto lora = dynamic_cast<LoRARowParallelLinear *>(m)) {
            lora->UnmergeWeights();
        }
    });
}

std::shared_ptr<Module> MergeAndUnload(std::shared_ptr<Module> model) {
    // First merge all LoRA weights
    MergeLoRAWeights(model);

    // Traverse model and replace LoRA modules with base modules
    auto named_modules = model->NamedModules();
    std::shared_ptr<Module> result = model;

    for (const auto &[name, module] : named_modules) {
        if (auto *lora = dynamic_cast<LoRALinear *>(module.get())) {
            // Create base Linear with same dimensions
            auto base = std::make_shared<nn::Linear>(lora->in_features(), lora->out_features(), lora->has_bias(),
                                                     lora->parameter(nn::Linear::kParamWeightName)->GetDevice());
            // Share the merged weight
            *base->mutable_parameter(nn::Linear::kParamWeightName) = lora->parameter(nn::Linear::kParamWeightName);
            if (lora->has_bias()) {
                *base->mutable_parameter(nn::Linear::kParamBiasName) = lora->parameter(nn::Linear::kParamBiasName);
            }

            if (name.empty()) {
                result = base;
            } else {
                ReplaceModuleByPath(model, name, base);
            }
        } else if (auto *lora = dynamic_cast<LoRAColumnParallelLinear *>(module.get())) {
            auto base = std::make_shared<parallel::ColumnParallelLinear>(
                lora->in_features(), lora->out_features(), lora->bias(), lora->gather_output(),
                lora->input_is_parallel(), lora->skip_bias_add(), lora->sequence_parallel());
            *base->mutable_parameter(parallel::ColumnParallelLinear::kParamWeightName)
                = lora->parameter(parallel::ColumnParallelLinear::kParamWeightName);
            if (lora->bias()) {
                *base->mutable_parameter(parallel::ColumnParallelLinear::kParamBiasName)
                    = lora->parameter(parallel::ColumnParallelLinear::kParamBiasName);
            }

            if (name.empty()) {
                result = base;
            } else {
                ReplaceModuleByPath(model, name, base);
            }
        } else if (auto *lora = dynamic_cast<LoRARowParallelLinear *>(module.get())) {
            auto base = std::make_shared<parallel::RowParallelLinear>(
                lora->in_features(), lora->out_features(), lora->bias(), lora->reduce_output(),
                lora->input_is_parallel(), lora->skip_bias_add(), lora->sequence_parallel());
            *base->mutable_parameter(parallel::RowParallelLinear::kParamWeightName)
                = lora->parameter(parallel::RowParallelLinear::kParamWeightName);
            if (lora->bias()) {
                *base->mutable_parameter(parallel::RowParallelLinear::kParamBiasName)
                    = lora->parameter(parallel::RowParallelLinear::kParamBiasName);
            }

            if (name.empty()) {
                result = base;
            } else {
                ReplaceModuleByPath(model, name, base);
            }
        }
    }

    // Unfreeze all parameters so the model is fully usable
    UnfreezeModel(result);

    LOG(INFO) << "MergeAndUnload: Merged LoRA weights and removed LoRA modules";
    return result;
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> LoRAStateDict(const std::shared_ptr<Module> &model) {
    std::unordered_map<std::string, std::shared_ptr<Tensor>> lora_state_dict;

    // This is intentionally the local model state. Under TP, sharded LoRA
    // parameters remain local shards; SaveLoRAWeights() exports portable tensors.
    for (auto &[name, param] : model->StateDict()) {
        // Only include LoRA parameters
        if (name.find(LoRALinear::kParamLoraAName) != std::string::npos
            || name.find(LoRALinear::kParamLoraBName) != std::string::npos) {
            lora_state_dict[name] = param;
        }
    }

    return lora_state_dict;
}

void LoadLoRAStateDict(std::shared_ptr<Module> model,
                       const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) {
    auto model_state_dict = model->StateDict();
    auto shardings = BuildLoRATensorShardings(model);

    for (auto &[name, param] : state_dict) { LoadLoRATensorIntoModel(name, param, model_state_dict, shardings); }
}

void SaveLoRAWeights(const std::shared_ptr<Module> &model, const std::string &filepath) {
    auto lora_state_dict = LoRAStateDict(model);
    auto sorted_names = SortedLoRAStateDictNames(lora_state_dict);
    auto shardings = BuildLoRATensorShardings(model);

    // TP ranks in the primary DP/PP group must all run the gather loop below.
    // Only TP rank 0 writes the file after receiving full tensors.
    //
    // TODO: PP save needs a per-stage or aggregated adapter format. The current
    // path only writes from pp=0, so it is not a complete PP checkpoint.
    if (!IsPrimarySaveTPGroupRank()) {
        LOG(INFO) << "Skip saving LoRA weights on non-primary DP/PP rank.";
        return;
    }

    const bool is_writer = IsSaveWriterRank();
    std::vector<std::pair<std::string, std::shared_ptr<Tensor>>> exported_tensors;
    if (is_writer) {
        exported_tensors.reserve(sorted_names.size());
    }

    for (const auto &name : sorted_names) {
        auto exported = ExportLoRATensorForSave(name, lora_state_dict.at(name), lora_state_dict, shardings);
        if (is_writer) {
            exported_tensors.emplace_back(name, exported);
        }
    }

    if (!is_writer) {
        return;
    }

    std::ofstream file(filepath, std::ios::binary);
    CHECK(file.is_open()) << "Failed to open file for writing: " << filepath;

    // Write magic number
    uint32_t magic = 0x4C4F5241; // "LORA"
    file.write(reinterpret_cast<const char *>(&magic), sizeof(magic));

    // Write version
    uint32_t version = 1;
    file.write(reinterpret_cast<const char *>(&version), sizeof(version));

    // Write number of tensors
    uint32_t num_tensors = static_cast<uint32_t>(exported_tensors.size());
    file.write(reinterpret_cast<const char *>(&num_tensors), sizeof(num_tensors));

    // Write each tensor
    for (const auto &[name, tensor] : exported_tensors) {
        // Write name length and name
        uint32_t name_len = static_cast<uint32_t>(name.length());
        file.write(reinterpret_cast<const char *>(&name_len), sizeof(name_len));
        file.write(name.c_str(), name_len);

        // Write tensor dimensions
        const auto &dims = tensor->Dims();
        uint32_t num_dims = static_cast<uint32_t>(dims.size());
        file.write(reinterpret_cast<const char *>(&num_dims), sizeof(num_dims));
        for (auto dim : dims) {
            int64_t d = dim;
            file.write(reinterpret_cast<const char *>(&d), sizeof(d));
        }

        // Write tensor data (copy to CPU first if needed)
        int64_t num_elements = tensor->NumElements();
        Tensor cpu_tensor = tensor->To(Device(Device::DeviceType::kCPU, 0));
        file.write(reinterpret_cast<const char *>(cpu_tensor.DataPtr()), num_elements * sizeof(float));
    }

    file.close();
    LOG(INFO) << "Saved LoRA weights to " << filepath << " (" << num_tensors << " tensors)";
}

void LoadLoRAWeights(std::shared_ptr<Module> model, const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary);
    CHECK(file.is_open()) << "Failed to open file for reading: " << filepath;

    // Read and verify magic number
    uint32_t magic;
    file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    CHECK_EQ(magic, 0x4C4F5241) << "Invalid LoRA file format";

    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    CHECK_EQ(version, 1) << "Unsupported LoRA file version: " << version;

    // Read number of tensors
    uint32_t num_tensors;
    file.read(reinterpret_cast<char *>(&num_tensors), sizeof(num_tensors));

    auto model_state_dict = model->StateDict();
    auto shardings = BuildLoRATensorShardings(model);

    // Read each tensor
    for (uint32_t i = 0; i < num_tensors; ++i) {
        // Read name
        uint32_t name_len;
        file.read(reinterpret_cast<char *>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        file.read(&name[0], name_len);

        // Read dimensions
        uint32_t num_dims;
        file.read(reinterpret_cast<char *>(&num_dims), sizeof(num_dims));
        std::vector<int64_t> dims(num_dims);
        for (uint32_t j = 0; j < num_dims; ++j) { file.read(reinterpret_cast<char *>(&dims[j]), sizeof(int64_t)); }

        // Calculate number of elements
        int64_t num_elements = 1;
        for (auto dim : dims) { num_elements *= dim; }

        // Read tensor data into a temporary CPU tensor
        auto cpu_tensor = std::make_shared<Tensor>(dims, DataType::kFLOAT32, Device(Device::DeviceType::kCPU, 0));
        file.read(reinterpret_cast<char *>(cpu_tensor->DataPtr()), num_elements * sizeof(float));

        LoadLoRATensorIntoModel(name, cpu_tensor, model_state_dict, shardings);
    }

    file.close();
    LOG(INFO) << "Loaded LoRA weights from " << filepath << " (" << num_tensors << " tensors)";
}

int64_t CountTrainableParameters(const std::shared_ptr<Module> &model) {
    int64_t count = 0;
    for (auto &param : model->Parameters()) {
        if (param->requires_grad()) {
            count += param->NumElements();
        }
    }
    return count;
}

int64_t CountTotalParameters(const std::shared_ptr<Module> &model) {
    // Use Parameters() instead of StateDict() to avoid counting:
    // 1. Shared/duplicated tensors (weight tying)
    // 2. Buffers (which are not trainable parameters)
    int64_t count = 0;
    auto params = model->Parameters();
    for (auto &param : params) { count += param->NumElements(); }
    return count;
}

void PrintLoRASummary(const std::shared_ptr<Module> &model, int global_rank) {
    int64_t trainable = CountTrainableParameters(model);
    int64_t total = CountTotalParameters(model);
    int64_t frozen = total - trainable;

    double trainable_pct = 100.0 * trainable / total;

    std::string title
        = global_rank >= 0 ? " LoRA Model Summary (Rank " + std::to_string(global_rank) + ") " : " LoRA Model Summary ";
    size_t pad_left = (40 - title.size()) / 2;
    size_t pad_right = 40 - title.size() - pad_left;
    std::string header = std::string(pad_left, '=') + title + std::string(pad_right, '=');
    std::string separator(header.size(), '=');

    std::cout << header << std::endl;
    std::cout << "Total parameters:     " << total << std::endl;
    std::cout << "Trainable parameters: " << trainable << " (" << trainable_pct << "%)" << std::endl;
    std::cout << "Frozen parameters:    " << frozen << std::endl;
    std::cout << separator << std::endl;
}

std::unordered_set<std::string> ParseLoRATargetModules(const std::string &targets) {
    std::unordered_set<std::string> result;
    std::stringstream ss(targets);
    std::string module;
    while (std::getline(ss, module, ',')) {
        // Trim whitespace
        module.erase(module.find_last_not_of(" \t\r\n") + 1);
        module.erase(0, module.find_first_not_of(" \t\r\n"));
        if (!module.empty()) {
            result.insert(module);
        }
    }
    return result;
}

} // namespace infini_train::nn::lora
