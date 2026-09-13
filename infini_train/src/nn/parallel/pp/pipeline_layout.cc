#include "infini_train/include/nn/parallel/pp/pipeline_layout.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>

#include "glog/logging.h"

namespace infini_train::nn::parallel {

namespace {
int ParseLayerCount(const std::string &token, const std::string &whole) {
    if (token.empty()) {
        LOG(FATAL) << "Invalid pipeline_layer_partition '" << whole << "': empty entry";
    }
    int value = 0;
    for (char c : token) {
        if (c < '0' || c > '9') {
            LOG(FATAL) << "Invalid pipeline_layer_partition '" << whole << "': '" << token
                       << "' is not a positive integer";
        }
        value = value * 10 + (c - '0');
    }
    if (value <= 0) {
        LOG(FATAL) << "Invalid pipeline_layer_partition '" << whole << "': layer count must be positive, got " << value;
    }
    return value;
}
} // namespace

std::vector<int> ParsePipelineLayerPartition(const std::string &str) {
    std::vector<int> partition;
    if (str.empty()) {
        return partition;
    }
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        partition.push_back(ParseLayerCount(token, str));
    }
    return partition;
}

std::vector<double> ParsePipelineLayerCosts(const std::string &str) {
    std::vector<double> costs;
    if (str.empty()) {
        return costs;
    }
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            LOG(FATAL) << "Invalid pipeline_layer_costs '" << str << "': empty entry";
        }
        size_t parsed = 0;
        double value = 0.0;
        try {
            value = std::stod(token, &parsed);
        } catch (const std::exception &) {
            LOG(FATAL) << "Invalid pipeline_layer_costs '" << str << "': '" << token << "' is not a number";
        }
        if (parsed != token.size()) {
            LOG(FATAL) << "Invalid pipeline_layer_costs '" << str << "': '" << token << "' is not a number";
        }
        if (!std::isfinite(value) || value < 0.0) {
            LOG(FATAL) << "Invalid pipeline_layer_costs '" << str << "': '" << token
                       << "' is not a finite non-negative number";
        }
        costs.push_back(value);
    }
    return costs;
}

std::vector<int> SuggestBalancedPartition(int total_layers, int num_stages, const std::vector<double> &layer_costs) {
    CHECK_GT(total_layers, 0) << "total_layers must be positive";
    CHECK_GT(num_stages, 0) << "num_stages must be positive";
    CHECK_GE(total_layers, num_stages) << "cannot assign fewer layers than stages";

    std::vector<double> costs(total_layers, 1.0);
    if (!layer_costs.empty()) {
        CHECK_EQ(layer_costs.size(), static_cast<size_t>(total_layers))
            << "layer_costs has " << layer_costs.size() << " entries but total_layers is " << total_layers;
        for (int i = 0; i < total_layers; ++i) {
            CHECK_GE(layer_costs[i], 0.0) << "layer_costs must be non-negative, layer " << i << " has "
                                          << layer_costs[i];
            costs[i] = layer_costs[i];
        }
    }

    // prefix[t] = sum of costs[0 .. t-1].
    std::vector<double> prefix(total_layers + 1, 0.0);
    for (int i = 0; i < total_layers; ++i) {
        prefix[i + 1] = prefix[i] + costs[i];
    }

    // dp[i][j] is the minimal achievable maximum per-segment cost when the first j layers
    // are split into i contiguous segments; split[i][j] records the boundary that reaches it
    // (segments 1..i-1 cover layers [0, split), segment i covers [split, j)).
    constexpr double kInf = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> dp(num_stages + 1, std::vector<double>(total_layers + 1, kInf));
    std::vector<std::vector<int>> split(num_stages + 1, std::vector<int>(total_layers + 1, 0));

    for (int j = 0; j <= total_layers; ++j) {
        dp[1][j] = prefix[j];
    }
    for (int i = 2; i <= num_stages; ++i) {
        for (int j = i; j <= total_layers; ++j) {
            for (int p = i - 1; p <= j - 1; ++p) {
                const double bottleneck = std::max(dp[i - 1][p], prefix[j] - prefix[p]);
                if (bottleneck < dp[i][j]) {
                    dp[i][j] = bottleneck;
                    split[i][j] = p;
                }
            }
        }
    }

    // Reconstruct the per-stage layer counts from the last segment back to the first.
    std::vector<int> partition(num_stages, 0);
    int j = total_layers;
    for (int i = num_stages; i >= 1; --i) {
        const int p = split[i][j];
        partition[i - 1] = j - p;
        j = p;
    }

    return partition;
}

PipelineLoadStats ComputePipelineLoadAnalysis(int total_layers, int num_stages, const std::vector<int> &partition,
                                              const std::vector<double> &layer_costs, int num_micro_batches) {
    CHECK_GT(total_layers, 0) << "total_layers must be positive";
    CHECK_GT(num_stages, 0) << "num_stages must be positive";
    CHECK_GE(total_layers, num_stages) << "cannot assign fewer layers than stages";
    CHECK_GT(num_micro_batches, 0) << "num_micro_batches must be positive";

    std::vector<double> costs(total_layers, 1.0);
    if (!layer_costs.empty()) {
        CHECK_EQ(layer_costs.size(), static_cast<size_t>(total_layers))
            << "layer_costs has " << layer_costs.size() << " entries but total_layers is " << total_layers;
        for (int i = 0; i < total_layers; ++i) {
            CHECK_GE(layer_costs[i], 0.0) << "layer_costs must be non-negative, layer " << i << " has "
                                          << layer_costs[i];
            costs[i] = layer_costs[i];
        }
    }

    std::vector<int> part = partition;
    if (part.empty()) {
        part = SuggestBalancedPartition(total_layers, num_stages, {}); // default uniform partition
    } else {
        CHECK_EQ(part.size(), static_cast<size_t>(num_stages))
            << "partition has " << part.size() << " entries but num_stages is " << num_stages;
    }

    PipelineLoadStats stats;
    stats.num_stages = num_stages;
    stats.num_micro_batches = num_micro_batches;
    stats.stage_loads.assign(num_stages, 0.0);

    int cursor = 0;
    for (int stage = 0; stage < num_stages; ++stage) {
        double load = 0.0;
        for (int k = 0; k < part[stage]; ++k) {
            CHECK_LT(cursor, total_layers) << "partition sums to more than " << total_layers << " layers";
            load += costs[cursor++];
        }
        stats.stage_loads[stage] = load;
    }
    CHECK_EQ(cursor, total_layers) << "partition sums to " << cursor << " layers but the model has " << total_layers;

    stats.bottleneck = *std::max_element(stats.stage_loads.begin(), stats.stage_loads.end());
    stats.average = std::accumulate(stats.stage_loads.begin(), stats.stage_loads.end(), 0.0) / num_stages;
    stats.efficiency = stats.bottleneck > 0.0 ? stats.average / stats.bottleneck : 0.0;
    stats.imbalance_bubble = 1.0 - stats.efficiency;
    stats.structural_bubble = static_cast<double>(num_stages - 1) / static_cast<double>(num_stages - 1 + num_micro_batches);
    return stats;
}

PipelineLayout PipelineLayout::Create(int total_layers, int num_stages, int vpp_size,
                                      const std::vector<int> &partition) {
    CHECK_GT(total_layers, 0) << "total_layers must be positive";
    CHECK_GT(num_stages, 0) << "num_stages must be positive";
    CHECK_GT(vpp_size, 0) << "vpp_size must be positive";

    PipelineLayout layout;
    layout.num_stages_ = num_stages;
    layout.total_layers_ = total_layers;
    layout.vpp_size_ = vpp_size;
    layout.first_stage_idx_ = 0;
    layout.last_stage_idx_ = num_stages - 1;
    layout.stage_layer_ranges_.assign(num_stages, {});

    if (partition.empty()) {
        // Default: uniform contiguous partition, optionally interleaved across virtual chunks.
        const int layers_per_chunk = total_layers / (num_stages * vpp_size);
        const int remainder = total_layers % (num_stages * vpp_size);
        for (int stage = 0; stage < num_stages; ++stage) {
            for (int local_chunk = 0; local_chunk < vpp_size; ++local_chunk) {
                const int global_chunk = local_chunk * num_stages + stage;
                if (global_chunk * layers_per_chunk >= total_layers) {
                    break;
                }
                int start = global_chunk * layers_per_chunk;
                int end = start + layers_per_chunk;
                if (global_chunk < remainder) {
                    start = global_chunk * (layers_per_chunk + 1);
                    end = start + (layers_per_chunk + 1);
                } else {
                    start = remainder * (layers_per_chunk + 1) + (global_chunk - remainder) * layers_per_chunk;
                    end = start + layers_per_chunk;
                }
                end = std::min(end, total_layers);
                if (start < end) {
                    layout.stage_layer_ranges_[stage].push_back({start, end});
                }
            }
        }
    } else {
        // Custom non-uniform contiguous partition; incompatible with virtual pipeline.
        CHECK_EQ(partition.size(), static_cast<size_t>(num_stages))
            << "pipeline_layer_partition has " << partition.size() << " entries but pipeline_parallel is "
            << num_stages;
        CHECK_EQ(vpp_size, 1)
            << "Custom pipeline_layer_partition is incompatible with virtual_pipeline_parallel > 1";
        int cursor = 0;
        for (int stage = 0; stage < num_stages; ++stage) {
            const int count = partition[stage];
            CHECK_GT(count, 0) << "pipeline_layer_partition entry must be positive, stage " << stage << " has "
                               << count;
            layout.stage_layer_ranges_[stage].push_back({cursor, cursor + count});
            cursor += count;
        }
        CHECK_EQ(cursor, total_layers) << "pipeline_layer_partition sums to " << cursor
                                       << " layers but the model has " << total_layers;
    }

    // Build the layer -> stage lookup and verify layers are neither missing nor duplicated.
    layout.layer_to_stage_.assign(total_layers, -1);
    for (int stage = 0; stage < num_stages; ++stage) {
        for (const auto &[start, end] : layout.stage_layer_ranges_[stage]) {
            for (int layer = start; layer < end; ++layer) {
                CHECK_EQ(layout.layer_to_stage_[layer], -1)
                    << "layer " << layer << " is assigned to more than one pipeline stage";
                layout.layer_to_stage_[layer] = stage;
            }
        }
    }
    for (int layer = 0; layer < total_layers; ++layer) {
        CHECK_NE(layout.layer_to_stage_[layer], -1) << "layer " << layer << " is not assigned to any pipeline stage";
    }

    return layout;
}

StageInfo PipelineLayout::GetStageInfo(int stage_id) const {
    CHECK_GE(stage_id, 0);
    CHECK_LT(stage_id, num_stages_);
    StageInfo info;
    info.is_first_stage = (stage_id == first_stage_idx_);
    info.is_last_stage = (stage_id == last_stage_idx_);
    info.layer_ranges_per_chunk = stage_layer_ranges_[stage_id];
    return info;
}

int PipelineLayout::StageOfLayer(int layer_id) const {
    CHECK_GE(layer_id, 0);
    CHECK_LT(layer_id, total_layers_);
    return layer_to_stage_[layer_id];
}

bool PipelineLayout::OwnsLayer(int stage_id, int layer_id) const { return StageOfLayer(layer_id) == stage_id; }

int PipelineLayout::StageOfChunk(int global_chunk_id, int num_stages) { return global_chunk_id % num_stages; }

int PipelineLayout::LocalChunkIndexOfChunk(int global_chunk_id, int num_stages) { return global_chunk_id / num_stages; }

std::string PipelineLayout::Describe() const {
    std::string s = "PipelineLayout: num_stages=" + std::to_string(num_stages_) +
                    ", total_layers=" + std::to_string(total_layers_) + ", vpp=" + std::to_string(vpp_size_) + "\n";
    for (int stage = 0; stage < num_stages_; ++stage) {
        s += "  stage " + std::to_string(stage) + ": ";
        for (size_t i = 0; i < stage_layer_ranges_[stage].size(); ++i) {
            const auto &[start, end] = stage_layer_ranges_[stage][i];
            if (i > 0) {
                s += ", ";
            }
            s += "[" + std::to_string(start) + ", " + std::to_string(end) + ")";
        }
        if (stage_layer_ranges_[stage].empty()) {
            s += "(no layers)";
        }
        if (stage == first_stage_idx_) {
            s += " + embedding";
        }
        if (stage == last_stage_idx_) {
            s += " + final_norm + lm_head";
        }
        s += "\n";
    }
    return s;
}

} // namespace infini_train::nn::parallel
