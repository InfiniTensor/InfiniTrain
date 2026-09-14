#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "infini_train/include/nn/parallel/pipeline_layout.h"

using infini_train::nn::parallel::PipelineLayout;

namespace {
int IntArg(const std::string& arg, const char* name) {
    const std::string prefix = std::string(name) + "=";
    if (arg.rfind(prefix, 0) != 0) return -1;
    return std::stoi(arg.substr(prefix.size()));
}
std::vector<double> CostsArg(const std::string& arg) {
    const std::string prefix = "--layer_cost=";
    if (arg.rfind(prefix, 0) != 0) return {};
    std::vector<double> out;
    std::stringstream ss(arg.substr(prefix.size()));
    std::string token;
    while (std::getline(ss, token, ',')) out.push_back(std::stod(token));
    return out;
}
}  // namespace

int main(int argc, char** argv) {
    int num_layers = -1, pp_size = -1;
    std::vector<double> costs;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            std::cout << "Usage: pipeline_layout_suggest --num_layers=N --pp_size=N "
                         "[--layer_cost=c1,c2,...]\n";
            return 0;
        }
        const int layers = IntArg(arg, "--num_layers");
        const int stages = IntArg(arg, "--pp_size");
        if (layers >= 0) num_layers = layers;
        if (stages >= 0) pp_size = stages;
        auto parsed = CostsArg(arg);
        if (!parsed.empty()) costs = std::move(parsed);
    }
    try {
        const auto partition = PipelineLayout::SuggestBalancedPartition(num_layers, pp_size, costs);
        std::cout << "pipeline_layer_partition=";
        for (size_t i = 0; i < partition.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << partition[i];
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PipelineLayoutError: " << e.what() << '\n';
        return 2;
    }
}
