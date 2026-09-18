#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "infini_train/include/nn/parallel/pipeline_layout.h"

using infini_train::nn::parallel::PipelineLayout;

namespace {
int IntArg(const std::string &arg, const char *name) {
    const std::string prefix = std::string(name) + "=";
    if (arg.rfind(prefix, 0) != 0) {
        return -1;
    }
    return std::stoi(arg.substr(prefix.size()));
}
std::vector<double> CostsArg(const std::string &arg) {
    const std::string prefix = "--layer_cost=";
    if (arg.rfind(prefix, 0) != 0) {
        return {};
    }
    std::vector<double> out;
    std::stringstream ss(arg.substr(prefix.size()));
    std::string token;
    while (std::getline(ss, token, ',')) { out.push_back(std::stod(token)); }
    return out;
}

std::vector<double> ReadCostsFile(const std::string &path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open layer cost file: " + path);
    }
    std::vector<double> costs;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        // Accept either one cost per line or CSV rows of `layer_id,cost`.
        std::stringstream ss(line);
        std::string first, second;
        if (!std::getline(ss, first, ',')) {
            continue;
        }
        if (std::getline(ss, second, ',')) {
            try {
                costs.push_back(std::stod(second));
            } catch (const std::exception &) {
                // Permit a conventional CSV header such as `layer,cost`.
                if (costs.empty()) {
                    continue;
                }
                throw;
            }
        } else {
            try {
                costs.push_back(std::stod(first));
            } catch (const std::exception &) {
                if (costs.empty()) {
                    continue;
                }
                throw;
            }
        }
    }
    return costs;
}

double MaxPartitionCost(const std::vector<double> &costs, const std::vector<int> &partition) {
    double max_cost = 0.0;
    size_t cursor = 0;
    for (int count : partition) {
        double stage_cost = 0.0;
        for (int i = 0; i < count; ++i) {
            stage_cost += costs[cursor++];
        }
        max_cost = std::max(max_cost, stage_cost);
    }
    return max_cost;
}
} // namespace

int main(int argc, char **argv) {
    int num_layers = -1, pp_size = -1;
    std::vector<double> costs;
    std::string cost_file;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            std::cout << "Usage: pipeline_layout_suggest --num_layers=N --pp_size=N "
                         "[--layer_cost=c1,c2,...] [--layer_cost_file=path]\n";
            return 0;
        }
        const int layers = IntArg(arg, "--num_layers");
        const int stages = IntArg(arg, "--pp_size");
        if (layers >= 0) {
            num_layers = layers;
        }
        if (stages >= 0) {
            pp_size = stages;
        }
        auto parsed = CostsArg(arg);
        if (!parsed.empty()) {
            costs = std::move(parsed);
        }
        const std::string cost_prefix = "--layer_cost_file=";
        if (arg.rfind(cost_prefix, 0) == 0) {
            cost_file = arg.substr(cost_prefix.size());
        }
    }
    try {
        if (!cost_file.empty()) {
            costs = ReadCostsFile(cost_file);
        }
        const auto partition = PipelineLayout::SuggestBalancedPartition(num_layers, pp_size, costs);
        std::cout << "pipeline_layer_partition=";
        for (size_t i = 0; i < partition.size(); ++i) {
            if (i) {
                std::cout << ',';
            }
            std::cout << partition[i];
        }
        std::cout << '\n';
        if (!costs.empty()) {
            const auto uniform = PipelineLayout::SuggestBalancedPartition(num_layers, pp_size);
            const double uniform_max = MaxPartitionCost(costs, uniform);
            const double suggested_max = MaxPartitionCost(costs, partition);
            const double reduction = uniform_max > 0.0 ? (uniform_max - suggested_max) / uniform_max * 100.0 : 0.0;
            std::cout << "uniform_partition=";
            for (size_t i = 0; i < uniform.size(); ++i) {
                if (i) {
                    std::cout << ',';
                }
                std::cout << uniform[i];
            }
            std::cout << '\n';
            std::cout << "uniform_max_stage_cost=" << uniform_max << '\n';
            std::cout << "suggested_max_stage_cost=" << suggested_max << '\n';
            std::cout << "predicted_max_stage_cost_reduction_pct=" << reduction << '\n';
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "PipelineLayoutError: " << e.what() << '\n';
        return 2;
    }
}
