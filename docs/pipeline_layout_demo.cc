// Demonstration for excellent-standard #4: compare the default uniform pipeline
// layout against a custom (cost-balanced) layout on a load-imbalanced model, and
// show the resulting pipeline bubble, per-stage time and throughput.
//
// This is a pure-CPU analytical demo: it uses the same ComputePipelineLoadAnalysis /
// SuggestBalancedPartition functions the training binary uses, so it can run on any
// machine without a GPU. The real (measured) per-stage CUDA timings are produced by
// the gpt2 example on a multi-GPU machine (see the multi-GPU run command in
// docs/pipeline_layout_guide.md).
//
// Build & run (from the repo root, inside WSL):
//   g++ -std=c++20 -I. -Ithird_party/glog/src \
//       docs/pipeline_layout_demo.cc infini_train/src/nn/parallel/pp/pipeline_layout.cc \
//       -Lbuild/third_party/glog -lglog -pthread -o build/pipeline_layout_demo
//   ./build/pipeline_layout_demo

#include <cstdio>
#include <string>
#include <vector>

#include "infini_train/include/nn/parallel/pp/pipeline_layout.h"

namespace {
using infini_train::nn::parallel::ComputePipelineLoadAnalysis;
using infini_train::nn::parallel::PipelineLoadStats;
using infini_train::nn::parallel::SuggestBalancedPartition;

// A load-imbalanced 12-layer model: the first 4 layers are "light" (cost 1) and the
// last 8 layers are "heavy" (cost 2). This models, e.g., attention-vs-MLP heavy blocks
// or a mixture-of-experts tail whose per-layer compute is no longer uniform.
std::vector<double> ImbalancedCosts() {
    std::vector<double> costs(12, 2.0);
    for (int i = 0; i < 4; ++i) { costs[i] = 1.0; }
    return costs;
}

std::string PartitionStr(const std::vector<int> &p) {
    std::string s;
    for (size_t i = 0; i < p.size(); ++i) {
        if (i) {
            s += ",";
        }
        s += std::to_string(p[i]);
    }
    return s;
}

// Steady-state throughput in micro-batches per unit time, from the GPipe make-span
// formula: T = bottleneck * (S - 1 + n), so throughput = n / T.
double Throughput(const PipelineLoadStats &s) {
    return s.num_micro_batches / (s.bottleneck * (s.num_stages - 1 + s.num_micro_batches));
}

void PrintComparison(const char *title, const std::vector<int> &uniform_partition,
                     const std::vector<int> &custom_partition, const std::vector<double> &costs, int n) {
    const int total_layers = static_cast<int>(costs.size());
    const int num_stages = static_cast<int>(uniform_partition.size());

    const PipelineLoadStats uniform
        = ComputePipelineLoadAnalysis(total_layers, num_stages, uniform_partition, costs, n);
    const PipelineLoadStats custom = ComputePipelineLoadAnalysis(total_layers, num_stages, custom_partition, costs, n);

    const std::string u_part = PartitionStr(uniform_partition);
    const std::string c_part = PartitionStr(custom_partition);

    std::printf("=== %s ===\n", title);
    std::printf("per-layer costs: [");
    for (size_t i = 0; i < costs.size(); ++i) { std::printf("%s%.0f", i ? "," : "", costs[i]); }
    std::printf("]   (S=%d stages, n=%d micro-batches)\n\n", num_stages, n);

    const std::string u_col = "uniform (" + u_part + ")";
    const std::string c_col = "custom (" + c_part + ")";
    std::printf("%-22s | %-14s | %-14s\n", "metric", u_col.c_str(), c_col.c_str());
    std::printf("%-22s-+-%s-+-%s\n", "----------------------", "---------------", "---------------");

    for (int s = 0; s < num_stages; ++s) {
        char line[32];
        std::snprintf(line, sizeof(line), "stage %d time (load)", s);
        std::printf("%-22s | %14.3f | %14.3f\n", line, uniform.stage_loads[s], custom.stage_loads[s]);
    }
    std::printf("%-22s | %14.3f | %14.3f\n", "bottleneck (max)", uniform.bottleneck, custom.bottleneck);
    std::printf("%-22s | %14.3f | %14.3f\n", "average", uniform.average, custom.average);
    std::printf("%-22s | %13.1f%% | %13.1f%%\n", "imbalance bubble", uniform.imbalance_bubble * 100.0,
                custom.imbalance_bubble * 100.0);
    std::printf("%-22s | %13.1f%% | %13.1f%%\n", "structural bubble", uniform.structural_bubble * 100.0,
                custom.structural_bubble * 100.0);
    std::printf("%-22s | %13.1f%% | %13.1f%%\n", "pipeline efficiency", uniform.efficiency * 100.0,
                custom.efficiency * 100.0);

    const double t_uniform = Throughput(uniform);
    const double t_custom = Throughput(custom);
    std::printf("%-22s | %14.4f | %14.4f\n", "throughput (mb/t)", t_uniform, t_custom);
    std::printf("%-22s | %14s | %13.2fx\n", "throughput speedup", "-", t_custom / t_uniform);
    std::printf("\n");
}
} // namespace

int main() {
    const int n = 8; // micro-batches per step

    // 1) Load-imbalanced model: default uniform {6,6} vs custom balanced {7,5}.
    const std::vector<double> costs = ImbalancedCosts();
    const std::vector<int> uniform{6, 6};
    const std::vector<int> custom = SuggestBalancedPartition(12, 2, costs);
    PrintComparison("Load-imbalanced model (4 light + 8 heavy layers)", uniform, custom, costs, n);

    // 2) Uniform-cost model: both layouts are equivalent, showing the feature correctly
    //    reports zero imbalance bubble for the balanced default.
    const std::vector<double> unit_costs(12, 1.0);
    const std::vector<int> unit_custom = SuggestBalancedPartition(12, 2, unit_costs);
    PrintComparison("Uniform-cost model (all layers equal)", uniform, unit_custom, unit_costs, n);

    return 0;
}
