#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>
#include "gtest/gtest.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/transformer/moe/experts.h"
#include "infini_train/include/nn/modules/transformer/moe/moe_layer.h"
#include "infini_train/include/nn/modules/transformer/moe/moe_utils.h"
#include "infini_train/include/nn/modules/transformer/moe/router.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

#ifndef INFINI_TRAIN_TEST_EP
#error "INFINI_TRAIN_TEST_EP must be defined"
#endif

namespace {

using infini_train::DataType;
using infini_train::Device;
using infini_train::Tensor;
namespace core = infini_train::core;
namespace nn = infini_train::nn;
namespace global = infini_train::nn::parallel::global;
namespace parallel = infini_train::nn::parallel;

constexpr int kExpertParallelSize = INFINI_TRAIN_TEST_EP;
constexpr int64_t kNumLocalExperts = 2;
constexpr int64_t kNumExperts = kExpertParallelSize * kNumLocalExperts;
constexpr int64_t kHiddenSize = 8;
constexpr int64_t kTokensPerRank = 2;
constexpr int64_t kFfnHiddenSize = 16;
constexpr float kAtol = 2e-4f;
constexpr float kRtol = 2e-4f;

const std::vector<std::string> kExpertParameterSuffixes = {
    "c_fc.weight",
    "c_fc.bias",
    "c_proj.weight",
    "c_proj.bias",
};

using StateDict = std::unordered_map<std::string, std::shared_ptr<Tensor>>;
using HostState = std::unordered_map<std::string, std::vector<float>>;

class DenseMoEReference final : public nn::Module {
public:
    explicit DenseMoEReference(const nn::TransformerConfig &config) : Module("DenseMoEReference") {
        modules_[nn::moe::MoELayer::kRouterLayerName] = std::make_shared<nn::moe::TopKRouter>(config);
        modules_[nn::moe::MoELayer::kExpertsLayerName]
            = std::make_shared<nn::moe::SequentialMLP>(kNumExperts, config);
    }

    std::vector<std::shared_ptr<Tensor>>
    Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        if (input_tensors.size() != 1) { throw std::runtime_error("DenseMoEReference expects one input"); }
        auto hidden_states = input_tensors[0];
        const auto hidden_dims = hidden_states->Dims();
        const int64_t num_tokens = hidden_states->NumElements() / kHiddenSize;
        auto hidden_states_2d = hidden_states->View({num_tokens, kHiddenSize});

        auto router_output = (*modules_.at(nn::moe::MoELayer::kRouterLayerName))({hidden_states});
        auto probs = router_output[0]->View({num_tokens, kNumExperts});
        auto routing_map = router_output[1]->View({num_tokens, kNumExperts});
        const auto &dispatch = nn::moe::Permute(hidden_states_2d, probs, routing_map);
        auto expert_output = (*modules_.at(nn::moe::MoELayer::kExpertsLayerName))(
            {dispatch.permuted_input, dispatch.metadata.tokens_per_expert, dispatch.permuted_probs})[0];
        return {nn::moe::Unpermute(expert_output, dispatch.metadata, {num_tokens, kHiddenSize})
                    ->View(hidden_dims)};
    }
};

nn::TransformerConfig MakeConfig() {
    nn::TransformerConfig config;
    config.n_embd = kHiddenSize;
    config.activation_type = nn::MLPType::kGELU;
    config.add_bias_linear = true;
    config.ffn_type = nn::FFNType::kMoE;
    config.moe_config = nn::MoEConfig{};
    config.moe_config->num_experts = kNumExperts;
    config.moe_config->router_topk = 2;
    config.moe_config->router_pre_softmax = true;
    config.moe_config->moe_ffn_hidden_size = kFfnHiddenSize;
    return config;
}

std::shared_ptr<Tensor> MakeCpuTensor(const std::vector<int64_t> &dims, const std::vector<float> &values) {
    auto tensor = std::make_shared<Tensor>(dims, DataType::kFLOAT32, Device());
    if (tensor->NumElements() != values.size()) { throw std::runtime_error("CPU tensor value count mismatch"); }
    std::memcpy(tensor->DataPtr(), values.data(), tensor->SizeInBytes());
    return tensor;
}

std::shared_ptr<Tensor> CopyToDevice(const std::shared_ptr<Tensor> &cpu_tensor, Device device, bool requires_grad) {
    auto tensor = std::make_shared<Tensor>(cpu_tensor->To(device));
    if (requires_grad) { tensor->RequiresGrad(); }
    return tensor;
}

std::vector<float> CopyToHost(const std::shared_ptr<Tensor> &tensor) {
    if (tensor == nullptr) { throw std::runtime_error("Cannot copy a null tensor"); }
    if (tensor->Dtype() != DataType::kFLOAT32) { throw std::runtime_error("Correctness gate expects FP32 tensors"); }
    auto cpu_tensor = tensor->To(Device());
    core::GetDeviceGuardImpl(tensor->GetDevice().type())->SynchronizeDevice(tensor->GetDevice());
    std::vector<float> values(tensor->NumElements());
    std::memcpy(values.data(), cpu_tensor.DataPtr(), cpu_tensor.SizeInBytes());
    return values;
}

HostState CopyGradientsToHost(const StateDict &state) {
    HostState gradients;
    for (const auto &[name, parameter] : state) {
        if (parameter->grad() != nullptr) { gradients.emplace(name, CopyToHost(parameter->grad())); }
    }
    return gradients;
}

std::vector<float> MakeGlobalInput() {
    std::vector<float> input(kNumExperts * kHiddenSize, 0.0f);
    for (int64_t token = 0; token < kNumExperts; ++token) { input[token * kHiddenSize + token] = 1.0f; }
    return input;
}

std::vector<float> MakeGlobalOutputGrad() {
    std::vector<float> grad(kNumExperts * kHiddenSize);
    for (size_t idx = 0; idx < grad.size(); ++idx) { grad[idx] = 0.25f + 0.001f * static_cast<float>(idx); }
    return grad;
}

std::vector<float> RankChunk(const std::vector<float> &global_values, int rank) {
    const size_t chunk_size = kTokensPerRank * kHiddenSize;
    const auto begin = global_values.begin() + rank * chunk_size;
    return {begin, begin + chunk_size};
}

void InitializeMasterState(const StateDict &state, bool empty_last_expert_rank) {
    auto router_weight = state.at("router.weight");
    auto router_bias = state.at("router.bias");
    router_weight->Fill(-4.0f);
    router_bias->Fill(0.0f);
    auto *router_weight_ptr = static_cast<float *>(router_weight->DataPtr());
    const int64_t active_experts = empty_last_expert_rank ? kNumExperts - kNumLocalExperts : kNumExperts;
    for (int64_t token = 0; token < kNumExperts; ++token) {
        const int64_t primary = empty_last_expert_rank ? token % active_experts : token;
        const int64_t secondary
            = empty_last_expert_rank ? (primary + 1) % active_experts
                                     : (primary + kNumLocalExperts) % kNumExperts;
        router_weight_ptr[primary * kHiddenSize + token] = 4.0f;
        router_weight_ptr[secondary * kHiddenSize + token] = 2.0f;
    }

    for (int64_t expert = 0; expert < kNumExperts; ++expert) {
        const float scale = static_cast<float>(expert + 1);
        state.at("experts.expert_" + std::to_string(expert) + ".c_fc.weight")->Fill(0.010f * scale);
        state.at("experts.expert_" + std::to_string(expert) + ".c_fc.bias")->Fill(0.001f * scale);
        state.at("experts.expert_" + std::to_string(expert) + ".c_proj.weight")->Fill(0.015f * scale);
        state.at("experts.expert_" + std::to_string(expert) + ".c_proj.bias")->Fill(0.002f * scale);
    }
}

void LoadLocalState(nn::moe::MoELayer &layer, const StateDict &master_state, int expert_rank) {
    StateDict local_state;
    local_state.emplace("router.weight", master_state.at("router.weight"));
    local_state.emplace("router.bias", master_state.at("router.bias"));
    for (int64_t local_expert = 0; local_expert < kNumLocalExperts; ++local_expert) {
        const int64_t global_expert = expert_rank * kNumLocalExperts + local_expert;
        for (const auto &suffix : kExpertParameterSuffixes) {
            local_state.emplace("experts.expert_" + std::to_string(local_expert) + "." + suffix,
                                master_state.at("experts.expert_" + std::to_string(global_expert) + "." + suffix));
        }
    }
    layer.LoadStateDict(local_state);
}

struct RunResult {
    std::vector<float> output;
    std::vector<float> input_grad;
    HostState parameter_grads;
    std::string error;
};

RunResult RunReference(const std::shared_ptr<DenseMoEReference> &reference,
                       const std::shared_ptr<Tensor> &input_cpu,
                       const std::shared_ptr<Tensor> &output_grad_cpu) {
    RunResult result;
    const Device device(Device::DeviceType::kCUDA, 0);
    global::thread_global_rank = 0;
    core::DeviceGuard guard(device);
    reference->To(device);
    auto input = CopyToDevice(input_cpu, device, true);
    auto output = (*reference)({input})[0];
    auto output_grad = CopyToDevice(output_grad_cpu, device, false);
    output->Backward(output_grad);
    result.output = CopyToHost(output);
    result.input_grad = CopyToHost(input->grad());
    result.parameter_grads = CopyGradientsToHost(reference->StateDict());
    return result;
}

RunResult RunExpertRank(const std::shared_ptr<nn::moe::MoELayer> &layer,
                        const std::shared_ptr<Tensor> &input_cpu,
                        const std::shared_ptr<Tensor> &output_grad_cpu, int rank) {
    RunResult result;
    try {
        global::thread_global_rank = rank;
        const Device device(Device::DeviceType::kCUDA, rank);
        core::DeviceGuard guard(device);
        layer->To(device);
        auto input = CopyToDevice(input_cpu, device, true);
        auto output = (*layer)({input})[0];
        auto output_grad = CopyToDevice(output_grad_cpu, device, false);
        output->Backward(output_grad);
        result.output = CopyToHost(output);
        result.input_grad = CopyToHost(input->grad());
        result.parameter_grads = CopyGradientsToHost(layer->StateDict());
    } catch (const std::exception &error) {
        result.error = error.what();
    }
    return result;
}

void ExpectVectorsNear(const std::vector<float> &actual, const std::vector<float> &expected,
                       const std::string &description) {
    ASSERT_EQ(actual.size(), expected.size()) << description;
    for (size_t idx = 0; idx < actual.size(); ++idx) {
        const float tolerance = kAtol + kRtol * std::abs(expected[idx]);
        EXPECT_NEAR(actual[idx], expected[idx], tolerance) << description << " at element " << idx;
    }
}

float MaxAbs(const std::vector<float> &values) {
    float result = 0.0f;
    for (float value : values) { result = std::max(result, std::abs(value)); }
    return result;
}

void RunCorrectnessCase(bool empty_last_expert_rank) {
    int available_gpus = 0;
    const auto cuda_status = cudaGetDeviceCount(&available_gpus);
    if (cuda_status != cudaSuccess || available_gpus < kExpertParallelSize) {
        GTEST_SKIP() << "requires " << kExpertParallelSize << " GPUs (found " << available_gpus << ")";
    }

    auto reference = std::make_shared<DenseMoEReference>(MakeConfig());
    InitializeMasterState(reference->StateDict(), empty_last_expert_rank);
    const auto master_state = reference->StateDict();

    std::vector<std::shared_ptr<nn::moe::MoELayer>> expert_layers;
    expert_layers.reserve(kExpertParallelSize);
    for (int rank = 0; rank < kExpertParallelSize; ++rank) {
        global::thread_global_rank = rank;
        auto layer = std::make_shared<nn::moe::MoELayer>(MakeConfig());
        LoadLocalState(*layer, master_state, rank);
        expert_layers.push_back(std::move(layer));
    }

    const auto global_input_values = MakeGlobalInput();
    const auto global_output_grad_values = MakeGlobalOutputGrad();
    auto global_input = MakeCpuTensor({kNumExperts, kHiddenSize}, global_input_values);
    auto global_output_grad = MakeCpuTensor({kNumExperts, kHiddenSize}, global_output_grad_values);
    const auto reference_result = RunReference(reference, global_input, global_output_grad);

    global::thread_global_rank = 0;
    auto *process_group_factory = parallel::ProcessGroupFactory::Instance(Device::DeviceType::kCUDA);
    process_group_factory->GetOrCreate(parallel::GetExpertTensorAndExpertParallelProcessGroupName(0),
                                       parallel::GetExpertTensorAndExpertParallelGroupRanks(0));

    std::vector<RunResult> expert_results(kExpertParallelSize);
    std::vector<std::thread> workers;
    workers.reserve(kExpertParallelSize);
    for (int rank = 0; rank < kExpertParallelSize; ++rank) {
        auto input = MakeCpuTensor({kTokensPerRank, kHiddenSize}, RankChunk(global_input_values, rank));
        auto output_grad
            = MakeCpuTensor({kTokensPerRank, kHiddenSize}, RankChunk(global_output_grad_values, rank));
        workers.emplace_back([&, rank, input, output_grad] {
            expert_results[rank] = RunExpertRank(expert_layers[rank], input, output_grad, rank);
        });
    }
    for (auto &worker : workers) { worker.join(); }

    for (int rank = 0; rank < kExpertParallelSize; ++rank) {
        ASSERT_TRUE(expert_results[rank].error.empty()) << "rank " << rank << ": " << expert_results[rank].error;
        ExpectVectorsNear(expert_results[rank].output, RankChunk(reference_result.output, rank),
                          "output rank " + std::to_string(rank));
        ExpectVectorsNear(expert_results[rank].input_grad, RankChunk(reference_result.input_grad, rank),
                          "input grad rank " + std::to_string(rank));
    }

    for (const auto &router_name : {std::string("router.weight"), std::string("router.bias")}) {
        ASSERT_TRUE(reference_result.parameter_grads.contains(router_name));
        std::vector<float> summed_router_grad(reference_result.parameter_grads.at(router_name).size(), 0.0f);
        for (int rank = 0; rank < kExpertParallelSize; ++rank) {
            ASSERT_TRUE(expert_results[rank].parameter_grads.contains(router_name));
            const auto &rank_grad = expert_results[rank].parameter_grads.at(router_name);
            ASSERT_EQ(rank_grad.size(), summed_router_grad.size());
            for (size_t idx = 0; idx < rank_grad.size(); ++idx) { summed_router_grad[idx] += rank_grad[idx]; }
        }
        ExpectVectorsNear(summed_router_grad, reference_result.parameter_grads.at(router_name), router_name);
    }
    EXPECT_GT(MaxAbs(reference_result.parameter_grads.at("router.weight")), 0.0f);

    const int64_t active_experts = empty_last_expert_rank ? kNumExperts - kNumLocalExperts : kNumExperts;
    for (int64_t global_expert = 0; global_expert < kNumExperts; ++global_expert) {
        const int owner = global_expert / kNumLocalExperts;
        const int64_t local_expert = global_expert % kNumLocalExperts;
        for (const auto &suffix : kExpertParameterSuffixes) {
            const auto reference_name = "experts.expert_" + std::to_string(global_expert) + "." + suffix;
            const auto local_name = "experts.expert_" + std::to_string(local_expert) + "." + suffix;
            if (global_expert >= active_experts) {
                EXPECT_FALSE(reference_result.parameter_grads.contains(reference_name));
                EXPECT_FALSE(expert_results[owner].parameter_grads.contains(local_name));
                continue;
            }
            ASSERT_TRUE(reference_result.parameter_grads.contains(reference_name)) << reference_name;
            ASSERT_TRUE(expert_results[owner].parameter_grads.contains(local_name)) << local_name;
            ExpectVectorsNear(expert_results[owner].parameter_grads.at(local_name),
                              reference_result.parameter_grads.at(reference_name), reference_name);
        }
    }
    EXPECT_GT(MaxAbs(reference_result.input_grad), 0.0f);
}

TEST(MoEExpertParallelCorrectnessTest, MatchesEp1OutputAndAllGradients) { RunCorrectnessCase(false); }

TEST(MoEExpertParallelCorrectnessTest, EmptyExpertRankKeepsCollectiveOrderAndParity) { RunCorrectnessCase(true); }

} // namespace
