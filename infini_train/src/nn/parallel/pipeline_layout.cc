#include "infini_train/include/nn/parallel/pipeline_layout.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <functional>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <charconv>
#include <string_view>

namespace infini_train::nn::parallel {
namespace {

[[noreturn]] void Fail(const std::string& msg) {
    throw PipelineLayoutError(msg);
}

void CheckStageId(int stage_id, int num_stages, const std::string& field) {
    if (stage_id < 0 || stage_id >= num_stages) {
        Fail(std::format(
            "{}={} is outside valid stage range [0, {})",
            field,
            stage_id,
            num_stages));
    }
}

}  // namespace

PipelineLayoutError::PipelineLayoutError(const std::string& msg): std::runtime_error(msg) {}

int LayerRange::size() const  {return end - start;}

bool LayerRange::contains(int layer_id) const  {
    return layer_id < end && layer_id >= start;
}

PipelineLayout::PipelineLayout() : num_layers_(0), num_stages_(1), vpp_size_(1) {
    BuildIndexes();
}

PipelineLayout PipelineLayout::BuildDefault(
    int num_layers,
    int num_stages,
    int vpp_size,
    SpecialModulePlacement placement,
    PipelineLayoutPolicy policy) {
    if (num_layers <= 0) {
        Fail(std::format("num_layers must be positive, got {}", num_layers));
    }
    if (num_stages <= 0) {
        Fail(std::format("num_stages must be positive, got {}", num_stages));
    }
    if (vpp_size <= 0) {
        Fail(std::format("vpp_size must be positive, got {}", vpp_size));
    }

    const long long total_chunks_ll = 
        static_cast<long long>(num_stages) * static_cast<long long>(vpp_size);
    if (total_chunks_ll > std::numeric_limits<int>::max()) {
        Fail("num_stages * vpp_size exceeds supported integer range");
    }

    // Total number of transformer layers.
    const int total_chunks = static_cast<int>(total_chunks_ll);

    const int per_chunk_layers = num_layers / total_chunks;
    const int remainder_layers = num_layers % total_chunks;

    std::vector<ChunkLayout> chunks;
    chunks.reserve(total_chunks);
    int layer_cursor = 0;
    for(int global_chunk_id = 0; global_chunk_id < total_chunks; ++global_chunk_id) {
        // Distribute remainder layers across the first chunks.
        int chunk_size = per_chunk_layers + (global_chunk_id < remainder_layers ? 1 : 0);
        if(chunk_size == 0) {
            continue;
        }
        int stage_id = global_chunk_id % num_stages;
        int local_chunk_id = global_chunk_id / num_stages;
        chunks.push_back(ChunkLayout{
              .global_chunk_id = global_chunk_id,
              .stage_id = stage_id,
              .local_chunk_id = local_chunk_id,
              .layers = LayerRange{.start = layer_cursor, .end = layer_cursor + chunk_size},
          });
        layer_cursor += chunk_size;
    }
    if (num_layers < total_chunks) {
        policy.allow_empty_stages = true;
    }
    auto normalized_placement = BuildPlacement(placement, num_stages);
    return PipelineLayout(num_layers, num_stages, vpp_size,
                          std::move(chunks), normalized_placement, policy);
}

PipelineLayout PipelineLayout::BuildContiguous(
    const std::vector<int>& stage_layer_counts,
    SpecialModulePlacement placement,
    PipelineLayoutPolicy policy
) {
    if(stage_layer_counts.empty()) {
        Fail("stage layer counts must not be empty");
    }

    const int num_stages = static_cast<int>(stage_layer_counts.size());

    int num_layers = 0;
    for(int idx=0; idx<num_stages; ++idx) {
        if(stage_layer_counts[idx] < 0) {
            Fail(std::format("stage_layer_counts must be non-negative, got {}", stage_layer_counts[idx]));
        }
        if(stage_layer_counts[idx] == 0 && !policy.allow_empty_stages) {
            Fail(std::format("stage {} has zero layers; set allow_empty_stages=true to permit", idx));
        }
        num_layers += stage_layer_counts[idx];
    }

      const int vpp_size = 1;

      std::vector<ChunkLayout> chunks;
      int layer_cursor = 0;
      for (int stage_id = 0; stage_id < num_stages; ++stage_id) {
          int count = stage_layer_counts[stage_id];
          if (count == 0) {
              continue;
          }
          int global_chunk_id = stage_id;  // With vpp_size=1, the IDs equal stage IDs.
          chunks.push_back(ChunkLayout{
              .global_chunk_id = global_chunk_id,
              .stage_id = stage_id,
              .local_chunk_id = 0,
              .layers = LayerRange{.start = layer_cursor, .end = layer_cursor + count},
          });
          layer_cursor += count;
      }

      auto normalized_placement = BuildPlacement(placement, num_stages);
      return PipelineLayout(num_layers, num_stages, vpp_size,
                            std::move(chunks), normalized_placement, policy);
}

// By default, place norm and lm_head on the last stage and embedding on the first.
SpecialModulePlacement PipelineLayout::BuildPlacement(
    SpecialModulePlacement placement,
    int num_stages
) {
    if (placement.embedding_stage < 0) {
        placement.embedding_stage = 0;
    }
    if (placement.final_norm_stage < 0) {
        placement.final_norm_stage = num_stages - 1;
    }
    if (placement.lm_head_stage < 0) {
        placement.lm_head_stage = num_stages - 1;
    }
    return placement;
}

PipelineLayout::PipelineLayout(
    int num_layers,
    int num_stages,
    int vpp_size,
    std::vector<ChunkLayout> chunks,
    SpecialModulePlacement placement,
    PipelineLayoutPolicy policy)
    : num_layers_(num_layers),
    num_stages_(num_stages),
    vpp_size_(vpp_size),
    chunks_(std::move(chunks)),
    placement_(std::move(placement)),
    policy_(std::move(policy)) {
    Validate();
    BuildIndexes();
}

void PipelineLayout::BuildIndexes() {
    // Build the layer_id -> stage_id index.
    layer_locations_.resize(num_layers_);

    for(const auto& c: chunks_) {
        for(int layer_id = c.layers.start; layer_id < c.layers.end; ++layer_id) {
            auto& loc = layer_locations_[layer_id];
            loc.stage_id = c.stage_id;
            loc.global_chunk_id = c.global_chunk_id;
            loc.local_chunk_id = c.local_chunk_id;
            loc.index_in_chunk = layer_id - c.layers.start;
        }
    }

    // Compute contiguous local indices across chunks within each stage.
    std::vector<int> stage_counter(num_stages_, 0);
    for(int layer_id = 0; layer_id < num_layers_; ++layer_id) {
        auto& loc = layer_locations_[layer_id];
        if(loc.stage_id >= 0) {
            loc.flat_local_layer_index = stage_counter[loc.stage_id]++;
        }
    }

    // Build stages grouped by stage_id and collect global_chunk_ids.
    stages_.clear();
    stages_.resize(num_stages_);
    for(int i=0; i<num_stages_; ++i) {
        stages_[i].stage_id = i;
    }
    for(const auto& c: chunks_) {
        stages_[c.stage_id].global_chunk_ids.push_back(c.global_chunk_id);
    }
}

  int PipelineLayout::num_layers() const { return num_layers_; }
  int PipelineLayout::num_stages() const { return num_stages_; }
  int PipelineLayout::vpp_size() const { return vpp_size_; }

  const StageLayout& PipelineLayout::stage(int stage_id) const {
      if (stage_id < 0 || stage_id >= num_stages_) {
          Fail(std::format("stage_id={} out of range [0, {})", stage_id, num_stages_));
      }
      return stages_[stage_id];
  }

  const ChunkLayout& PipelineLayout::chunk(int global_chunk_id) const {
      for (const auto& c : chunks_) {
          if (c.global_chunk_id == global_chunk_id) {
              return c;
          }
      }
      Fail(std::format("global_chunk_id={} not found in layout", global_chunk_id));
  }

  const std::vector<ChunkLayout>& PipelineLayout::chunks() const { return chunks_; }

  const LayerLocation& PipelineLayout::locate_layer(int layer_id) const {
      if (layer_id < 0 || layer_id >= num_layers_) {
          Fail(std::format("layer_id={} out of range [0, {})", layer_id, num_layers_));
      }
      return layer_locations_[layer_id];
  }

  int PipelineLayout::stage_of_layer(int layer_id) const {
      return locate_layer(layer_id).stage_id;
  }

  int PipelineLayout::local_layer_index(int stage_id, int layer_id) const {
      if (stage_id < 0 || stage_id >= num_stages_) {
          Fail(std::format("stage_id={} out of range [0, {})", stage_id, num_stages_));
      }
      auto& loc = locate_layer(layer_id);
      if (loc.stage_id != stage_id) {
          Fail(std::format("layer {} belongs to stage {}, not {}", layer_id, loc.stage_id,
          stage_id));
      }
      return loc.flat_local_layer_index;
  }

  const ChunkLayout& PipelineLayout::chunk_of_layer(int layer_id) const {
      return chunk(locate_layer(layer_id).global_chunk_id);
  }

  bool PipelineLayout::owns(SpecialModule module, int stage_id) const {
      switch (module) {
      case SpecialModule::kEmbedding:
          return placement_.embedding_stage == stage_id;
      case SpecialModule::kFinalNorm:
          return placement_.final_norm_stage == stage_id;
      case SpecialModule::kLMHead:
          return placement_.lm_head_stage == stage_id;
      }
      return false;
  }

  const SpecialModulePlacement& PipelineLayout::special_modules() const { return placement_; }
  const PipelineLayoutPolicy& PipelineLayout::policy() const { return policy_; }

  void PipelineLayout::Validate() const {
      // 1. Layer ranges must cover [0, num_layers_) exactly once.
      std::vector<bool> seen(num_layers_, false);
      for (const auto& c : chunks_) {
          if (c.layers.start < 0 || c.layers.end < c.layers.start || c.layers.end > num_layers_) {
              Fail(std::format("chunk {} layer range [{}, {}) is outside [0, {})",
                               c.global_chunk_id, c.layers.start, c.layers.end, num_layers_));
          }
          for (int l = c.layers.start; l < c.layers.end; ++l) {
              if (seen[l]) {
                  Fail(std::format("layer {} appears in more than one chunk", l));
              }
              seen[l] = true;
          }
      }
      for (int l = 0; l < num_layers_; ++l) {
          if (!seen[l]) {
              Fail(std::format("layer {} is not assigned to any chunk", l));
          }
      }

      // 2. All chunk and stage indices must be valid.
      for (const auto& c : chunks_) {
          if (c.stage_id < 0 || c.stage_id >= num_stages_) {
              Fail(std::format("chunk {} stage_id={} out of range [0, {})",
                               c.global_chunk_id, c.stage_id, num_stages_));
          }
          if (c.global_chunk_id < 0) {
              Fail(std::format("chunk has negative global_chunk_id={}", c.global_chunk_id));
          }
      }

      std::unordered_set<int> chunk_ids;
      for (const auto& c : chunks_) {
          if (!chunk_ids.insert(c.global_chunk_id).second) {
              Fail(std::format("duplicate global_chunk_id={}", c.global_chunk_id));
          }
          if (c.local_chunk_id < 0) {
              Fail(std::format("chunk {} has negative local_chunk_id={}",
                               c.global_chunk_id, c.local_chunk_id));
          }
      }

      // 3. Special-module stages must be valid.
      CheckStageId(placement_.embedding_stage, num_stages_, "embedding_stage");
      CheckStageId(placement_.final_norm_stage, num_stages_, "final_norm_stage");
      CheckStageId(placement_.lm_head_stage, num_stages_, "lm_head_stage");

      if (policy_.require_boundary_special_modules) {
          if (placement_.embedding_stage != 0) {
              Fail(std::format("embedding_stage={} must be boundary stage 0",
                               placement_.embedding_stage));
          }
          if (placement_.final_norm_stage != num_stages_ - 1) {
              Fail(std::format("final_norm_stage={} must be last stage {}",
                               placement_.final_norm_stage, num_stages_ - 1));
          }
          if (placement_.lm_head_stage != num_stages_ - 1) {
              Fail(std::format("lm_head_stage={} must be last stage {}",
                               placement_.lm_head_stage, num_stages_ - 1));
          }
      }

      // 4. The default policy disallows empty stages.
      if (!policy_.allow_empty_stages) {
          std::vector<bool> stage_has_layers(num_stages_, false);
          for (const auto& c : chunks_) {
              if (c.layers.size() > 0) {
                  stage_has_layers[c.stage_id] = true;
              }
          }
          for (int i = 0; i < num_stages_; ++i) {
              if (!stage_has_layers[i]) {
                  Fail(std::format("stage {} has no layers; set allow_empty_stages=true to permit",
                  i));
              }
          }
      }

      // 5. The default policy requires contiguous execution order.
      if (policy_.require_contiguous_execution) {
          // Sort by layers.start and verify adjacent ranges are contiguous.
          auto sorted = chunks_;
          std::sort(sorted.begin(), sorted.end(),
                    [](const ChunkLayout& a, const ChunkLayout& b) {
                        return a.layers.start < b.layers.start;
                    });
          for (size_t i = 1; i < sorted.size(); ++i) {
              if (sorted[i].layers.start != sorted[i - 1].layers.end) {
                  Fail(std::format("layer ranges are not contiguous: gap between [{}, {}) and [{}, {})",
                    sorted[i - 1].layers.start, sorted[i - 1].layers.end,
                    sorted[i].layers.start, sorted[i].layers.end));
              }
          }
      }
  }

void PipelineLayout::ValidateForCurrentPipelineTransport() const {
      Validate();
      // Send/recv currently connects adjacent stages in 0..num_stages-1 order,
      // so every stage must have at least one executable chunk.
      for (int i = 0; i < num_stages_; ++i) {
          if (stages_[i].global_chunk_ids.empty()) {
              Fail(std::format(
                  "stage {} has no chunks; the current pipeline transport requires "
                  "every stage to have at least one chunk", i));
          }
      }
  }

  std::string PipelineLayout::ToString() const {
      std::ostringstream out;
      out << std::format("PipelineLayout:\n");
      out << std::format("  num_layers: {}\n", num_layers_);
      out << std::format("  num_stages: {}\n", num_stages_);
      out << std::format("  vpp_size: {}\n", vpp_size_);
      out << std::format("  special_modules: embedding={}, final_norm={}, lm_head={}\n",
                         placement_.embedding_stage,
                         placement_.final_norm_stage,
                         placement_.lm_head_stage);

      for (int sid = 0; sid < num_stages_; ++sid) {
          out << std::format("  stage {}:", sid);
          const auto& stage = stages_[sid];
          if (stage.global_chunk_ids.empty()) {
              out << " (empty)";
          }
          for (int gid : stage.global_chunk_ids) {
              const auto& c = chunk(gid);
              out << std::format(" [layers {}-{}]", c.layers.start, c.layers.end - 1);
          }
          // Record special-module ownership.
          std::vector<std::string> specials;
          if (owns(SpecialModule::kEmbedding, sid)) specials.push_back("embedding");
          if (owns(SpecialModule::kFinalNorm, sid)) specials.push_back("final_norm");
          if (owns(SpecialModule::kLMHead, sid)) specials.push_back("lm_head");
          if (!specials.empty()) {
              out << " {";
              for (size_t i = 0; i < specials.size(); ++i) {
                  if (i > 0) out << ", ";
                  out << specials[i];
              }
              out << "}";
          }
          out << "\n";
      }
      return out.str();
  }

std::vector<int> PipelineLayout::ParseLayerPartition(const std::string &value) {
    if (value.empty()) {
        Fail("pipeline_layer_partition must not be empty");
    }

    std::vector<int> result;
    size_t begin = 0;

    while (begin <= value.size()) {
        const size_t comma = value.find(',', begin);
        const size_t end = (comma == std::string::npos) ? value.size() : comma;

        if (end == begin) {
            Fail(std::format("pipeline_layer_partition contains an empty token near offset {}", begin));
        }

        const std::string_view token(value.data() + begin, end - begin);

      // Only non-negative integer tokens are accepted; whitespace, signs and letters are rejected.
        for (const char c : token) {
            if (c < '0' || c > '9') {
                Fail(std::format("pipeline_layer_partition contains invalid character '{}'", c));
            }
        }

        int number = 0;
        const char *const first = token.data();
        const char *const last = token.data() + token.size();
        const auto [ptr, ec] = std::from_chars(first, last, number);

        if (ec != std::errc() || ptr != last) {
            Fail(std::format("pipeline_layer_partition token '{}' is not a valid non-negative int", token));
        }

        result.push_back(number);

        if (comma == std::string::npos) {
            break;
        }

        begin = comma + 1;
    }

    return result;
}

PipelineLayout PipelineLayout::ParseMegatronStyleLayout(
    const std::string& value,
    int num_layers,
    int pp_size,
    int vpp_size,
    SpecialModulePlacement placement,
    PipelineLayoutPolicy policy) {
    if (value.empty()) Fail("pipeline_layout must not be empty");
    if (num_layers <= 0 || pp_size <= 0 || vpp_size <= 0) {
        Fail(std::format("num_layers, pp_size and vpp_size must be positive (got {}, {}, {})",
                         num_layers, pp_size, vpp_size));
    }

    // Expand parenthesized repetitions while preserving stage separators.
    std::function<std::string(std::string_view)> expand = [&](std::string_view input) {
        std::string out;
        for (size_t i = 0; i < input.size();) {
            if (input[i] != '(') {
                if (input[i] == ')' || input[i] == '*') {
                    Fail(std::format("pipeline_layout has unexpected '{}' at offset {}", input[i], i));
                }
                out.push_back(input[i++]);
                continue;
            }
            size_t depth = 1, j = i + 1;
            for (; j < input.size() && depth != 0; ++j) {
                if (input[j] == '(') ++depth;
                else if (input[j] == ')') --depth;
            }
            if (depth != 0) Fail("pipeline_layout has unmatched '('");
            const std::string inner = expand(input.substr(i + 1, j - i - 2));
            size_t repeat = 1;
            if (j < input.size() && input[j] == '*') {
                size_t k = j + 1;
                if (k == input.size() || input[k] < '0' || input[k] > '9') {
                    Fail("pipeline_layout repetition must use *N");
                }
                repeat = 0;
                while (k < input.size() && input[k] >= '0' && input[k] <= '9') {
                    repeat = repeat * 10 + static_cast<size_t>(input[k++] - '0');
                    if (repeat > 100000) Fail("pipeline_layout repetition is too large");
                }
                if (repeat == 0) Fail("pipeline_layout repetition must be positive");
                j = k;
            }
            for (size_t n = 0; n < repeat; ++n) out += inner;
            i = j;
        }
        return out;
    };

    const std::string expanded = expand(value);
    std::vector<std::string> stage_exprs;
    size_t begin = 0;
    while (true) {
        const size_t sep = expanded.find('|', begin);
        stage_exprs.emplace_back(expanded.substr(begin, sep == std::string::npos
                                                          ? std::string::npos : sep - begin));
        if (sep == std::string::npos) break;
        begin = sep + 1;
    }
    if (static_cast<int>(stage_exprs.size()) != pp_size) {
        Fail(std::format("pipeline_layout stage count mismatch: pp_size={}, stages={}",
                         pp_size, stage_exprs.size()));
    }

    std::vector<std::vector<int>> layer_counts(pp_size);
    bool saw_embedding = false, saw_final_norm = false, saw_lm_head = false;
    for (int stage_id = 0; stage_id < pp_size; ++stage_id) {
        const auto& expr = stage_exprs[stage_id];
        size_t chunk_begin = 0;
        while (true) {
            const size_t comma = expr.find(',', chunk_begin);
            const std::string token = expr.substr(chunk_begin,
                comma == std::string::npos ? std::string::npos : comma - chunk_begin);
            int t_count = 0;
            for (char c : token) {
                switch (c) {
                case 't': case 'T': ++t_count; break;
                case 'E':
                    if (saw_embedding) Fail("pipeline_layout contains multiple E tokens");
                    placement.embedding_stage = stage_id; saw_embedding = true; break;
                case 'F':
                    if (saw_final_norm) Fail("pipeline_layout contains multiple F tokens");
                    placement.final_norm_stage = stage_id; saw_final_norm = true; break;
                case 'H': case 'L':
                    if (saw_lm_head) Fail("pipeline_layout contains multiple H/L tokens");
                    placement.lm_head_stage = stage_id; saw_lm_head = true; break;
                case ' ': case '\t': case '\n': case '\r':
                    Fail("pipeline_layout does not allow whitespace");
                default:
                    Fail(std::format("pipeline_layout contains unknown symbol '{}'", c));
                }
            }
            layer_counts[stage_id].push_back(t_count);
            if (comma == std::string::npos) break;
            chunk_begin = comma + 1;
        }
        if (static_cast<int>(layer_counts[stage_id].size()) != vpp_size) {
            Fail(std::format(
                "pipeline_layout chunk count mismatch at stage {}: vpp_size={}, chunks={}",
                stage_id, vpp_size, layer_counts[stage_id].size()));
        }
    }

    // Global execution order is chunk-major: all stages of local chunk 0,
    // followed by all stages of local chunk 1, matching the existing
    // interleaved scheduler's global chunk numbering.
    std::vector<ChunkLayout> chunks;
    chunks.reserve(static_cast<size_t>(pp_size) * static_cast<size_t>(vpp_size));
    int layer_cursor = 0;
    for (int local_chunk_id = 0; local_chunk_id < vpp_size; ++local_chunk_id) {
        for (int stage_id = 0; stage_id < pp_size; ++stage_id) {
            const int count = layer_counts[stage_id][local_chunk_id];
            const int global_chunk_id = local_chunk_id * pp_size + stage_id;
            chunks.push_back(ChunkLayout{.global_chunk_id = global_chunk_id,
                                         .stage_id = stage_id,
                                         .local_chunk_id = local_chunk_id,
                                         .layers = LayerRange{layer_cursor, layer_cursor + count}});
            layer_cursor += count;
        }
    }
    if (layer_cursor != num_layers) {
        Fail(std::format("pipeline_layout layer count mismatch: num_layers={}, parsed_layers={}",
                         num_layers, layer_cursor));
    }
    if (!saw_embedding) placement.embedding_stage = 0;
    if (!saw_final_norm) placement.final_norm_stage = pp_size - 1;
    if (!saw_lm_head) placement.lm_head_stage = pp_size - 1;
    // Explicit layouts may intentionally contain empty stages; retain them for
    // inspection and let transport validation decide whether they are runnable.
    for (const auto& c : chunks) if (c.layers.size() == 0) policy.allow_empty_stages = true;
    return PipelineLayout(num_layers, pp_size, vpp_size, std::move(chunks),
                          BuildPlacement(placement, pp_size), policy);
}

std::vector<int> PipelineLayout::SuggestBalancedPartition(
    int num_layers, int pp_size, const std::vector<double>& layer_costs) {
    if (num_layers <= 0 || pp_size <= 0) {
        Fail(std::format("num_layers and pp_size must be positive (got {}, {})",
                         num_layers, pp_size));
    }
    if (num_layers < pp_size) {
        Fail(std::format("num_layers={} is smaller than pp_size={}", num_layers, pp_size));
    }
    std::vector<double> costs = layer_costs;
    if (costs.empty()) costs.assign(static_cast<size_t>(num_layers), 1.0);
    if (static_cast<int>(costs.size()) != num_layers) {
        Fail(std::format("layer_costs length mismatch: num_layers={}, costs={}",
                         num_layers, costs.size()));
    }
    for (int i = 0; i < num_layers; ++i) {
        if (!std::isfinite(costs[i]) || costs[i] < 0.0) {
            Fail(std::format("layer_costs[{}] must be finite and non-negative", i));
        }
    }
    double total = 0.0;
    for (double c : costs) total += c;
    std::vector<int> result;
    result.reserve(pp_size);
    int cursor = 0;
    double remaining = total;
    for (int stage = 0; stage < pp_size - 1; ++stage) {
        const int layers_left = num_layers - cursor;
        const int stages_left = pp_size - stage;
        const double target = remaining / static_cast<double>(stages_left);
        int take = 1;
        double accum = costs[cursor];
        while (take < layers_left - (stages_left - 1) &&
               accum + costs[cursor + take] <= target) {
            accum += costs[cursor + take++];
        }
        // If the next layer gets us closer to the target, include it; a heavy
        // layer is consequently isolated instead of being paired blindly.
        if (take < layers_left - (stages_left - 1) &&
            std::abs(accum + costs[cursor + take] - target) <
                std::abs(accum - target)) {
            accum += costs[cursor + take++];
        }
        result.push_back(take);
        cursor += take;
        remaining -= accum;
    }
    result.push_back(num_layers - cursor);
    return result;
}

PipelineLayout PipelineLayout::BuildPipelineLayout(
    int num_layers,
    int pp_size,
    int vpp_size,
    const std::string &layer_partition,
    SpecialModulePlacement placement,
    PipelineLayoutPolicy policy) {
        if(layer_partition.empty()) {
            return BuildDefault(num_layers, pp_size, vpp_size, placement, policy);
        }
        std::vector<int> pp = ParseLayerPartition(layer_partition);

        if (static_cast<int>(pp.size()) != pp_size) {
            Fail(std::format(
                "stage count mismatch: pipeline_parallel={}, partition_entries={}",
                pp_size,
                pp.size()));
        }

        long long layer_sum = 0;
        for (int layer_count : pp) {
            layer_sum += layer_count;
        }

        if (layer_sum != num_layers) {
            Fail(std::format(
                "layer sum mismatch: num_layers={}, partition_sum={}",
                num_layers,
                layer_sum));
        }

        if (vpp_size != 1) {
            Fail(std::format(
                "custom layer partition is not supported with vpp_size={}",
                vpp_size));
        }

        return BuildContiguous(pp, placement, policy);
    }

}  // namespace infini_train::nn::parallel
