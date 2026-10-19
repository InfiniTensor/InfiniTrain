#include "infini_train/include/nn/modules/transformer/transformer_config.h"

// std::pair structured bindings are provided by <utility>.
// IWYU pragma: no_include <tuple>

#include <stddef.h>

#include <array>
#include <ostream>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

namespace infini_train::nn {
namespace {
template <typename Enum, size_t N>
Enum ParseEnum(std::string_view value,
               const std::array<std::pair<std::string_view, Enum>, N> &entries,
               std::string_view name, std::string_view expected) {
  for (const auto &[key, parsed] : entries) {
    if (value == key) {
      return parsed;
    }
  }
  LOG(FATAL) << "Unknown " << name << ": " << value << ". Expected " << expected
             << ".";
  return entries[0].second;
}

ActivationRecomputeGranularity ParseActivationRecomputeGranularity(
    std::string_view value) {
  static constexpr std::array kEntries = {
      std::pair<std::string_view, ActivationRecomputeGranularity>{
          "none", ActivationRecomputeGranularity::kNone},
      std::pair<std::string_view, ActivationRecomputeGranularity>{
          "full", ActivationRecomputeGranularity::kFull},
      std::pair<std::string_view, ActivationRecomputeGranularity>{
          "selective", ActivationRecomputeGranularity::kSelective},
  };
  return ParseEnum(value, kEntries, "recompute_granularity",
                   "none|full|selective");
}

ActivationRecomputeMethod ParseActivationRecomputeMethod(
    std::string_view value) {
  static constexpr std::array kEntries = {
      std::pair<std::string_view, ActivationRecomputeMethod>{
          "none", ActivationRecomputeMethod::kNone},
      std::pair<std::string_view, ActivationRecomputeMethod>{
          "uniform", ActivationRecomputeMethod::kUniform},
      std::pair<std::string_view, ActivationRecomputeMethod>{
          "block", ActivationRecomputeMethod::kBlock},
  };
  return ParseEnum(value, kEntries, "recompute_method", "none|uniform|block");
}
}  // namespace

bool TransformerConfig::UseGQA() const { return n_kv_head < n_head; }

int TransformerConfig::GetChunkSize() const {
  auto stage_info = parallel::PipelineParallel::GetStageInfo(
      n_layer, parallel::global::GetPipelineParallelSize(), parallel::pp_rank,
      parallel::global::GetVirtualPipelineParallelSize());
  return stage_info.layer_ranges_per_chunk.size();
}

bool TransformerConfig::RecomputeEnabled() const {
  return recompute_granularity != ActivationRecomputeGranularity::kNone;
}

ActivationRecomputeOptions ParseActivationRecomputeOptions(
    bool enabled, std::string_view granularity, std::string_view method,
    int64_t num_layers) {
  ActivationRecomputeOptions options;
  if (!enabled) {
    return options;
  }
  options.granularity = ParseActivationRecomputeGranularity(granularity);
  if (options.granularity == ActivationRecomputeGranularity::kNone ||
      options.granularity == ActivationRecomputeGranularity::kSelective) {
    return options;
  }
  options.method = ParseActivationRecomputeMethod(method);
  if (options.method == ActivationRecomputeMethod::kNone) {
    return options;
  }
  CHECK_GE(num_layers, 1) << "recompute_num_layers must be >= 1 when "
                             "recompute_method is uniform or block.";
  options.num_layers = num_layers;
  return options;
}

void ApplyActivationRecomputeOptions(
    TransformerConfig *config, const ActivationRecomputeOptions &options) {
  CHECK(config);
  config->recompute_granularity = options.granularity;
  config->recompute_method = options.method;
  config->recompute_num_layers = options.num_layers;
}
}  // namespace infini_train::nn
