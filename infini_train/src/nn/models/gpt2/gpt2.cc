#include "infini_train/include/models/gpt2/gpt2.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "example/common/utils.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/config.h"
#include "infini_train/include/nn/modules/transformer/spec.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;
namespace nn = infini_train::nn;

namespace {
constexpr int kRandomSeed = 42;

// TODO(dcj): make this rng generator compatible with torch later
static std::mt19937 gen{kRandomSeed};
} // namespace

std::vector<std::shared_ptr<infini_train::Tensor>>
NewGELU::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto &input = x[0];
    return {0.5 * input
            * (1.0 + nn::function::Tanh(std::sqrt(2.0 / M_PI) * (input + 0.044715 * nn::function::Pow(input, 3.0))))};
}

CausalSelfAttention::CausalSelfAttention(const nn::TransformerConfig &config)
    : config_(config), n_head_(config.n_head), n_embd_(config.n_embd) {
    auto tp_world_size = nn::parallel::global::GetTensorParallelSize();
    CHECK_EQ(config.n_embd % config.n_head, 0);
    CHECK_EQ(n_head_ % tp_world_size, 0) << "n_head must be divisible by TP world size";
    local_n_head_ = n_head_ / tp_world_size;

    // qkv: ColumnParallel (do not gather output) -> each tp_rank gets 3 * (n_embd / tp_world) channels
    modules_[kCAttnLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
        /*in_features=*/n_embd_,
        /*out_features=*/3 * n_embd_,
        /*bias=*/true,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    // proj: RowParallel (input is parallel and output is full)
    modules_[kCProjLayerName] = std::make_shared<nn::parallel::RowParallelLinear>(
        /*in_features=*/n_embd_,
        /*out_features=*/n_embd_,
        /*bias=*/true,
        /*reduce_output=*/true,
        /*input_is_parallel=*/true,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    // causal mask: (1, 1, block_size, block_size)
    buffers_[kParamBiasName] = nn::function::Tril(nn::function::Ones({config_.block_size, config_.block_size}))
                                   ->View({1, 1, config_.block_size, config_.block_size});
}

std::vector<std::shared_ptr<infini_train::Tensor>>
CausalSelfAttention::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto tp_world_size = nn::parallel::global::GetTensorParallelSize();

    const auto B = x[0]->Dims()[0];                  // bs
    const auto C = x[0]->Dims()[2];                  // n_embd
    const int64_t head_dim = n_embd_ / n_head_;      // per-head dim (global)
    const int64_t local_C = n_embd_ / tp_world_size; // per-rank hidden

    // (B, T, C) -> ColumnParallelLinear(C, 3*C) -> (B, T, 3 * local_C)
    // -> Split -> (3, B, T, local_C)
    auto qkv = modules_[kCAttnLayerName]->Forward(x)[0]->Split(local_C, 2);

    // (B, T, local_C)
    auto q = qkv[0];
    auto k = qkv[1];
    auto v = qkv[2];

    // NOTE(zbl): Acquire full T after AllGather is performed in ColumnParallelLinear
    const auto T = q->Dims()[1];

    // View to multi-head: local_n_head * head_dim == local_C
    // (B, T, local_C) -> (B, T, h_l, Dh) -> (B, h_l, T, Dh)
    k = k->View({B, T, local_n_head_, head_dim})->Transpose(1, 2);
    q = q->View({B, T, local_n_head_, head_dim})->Transpose(1, 2);
    v = v->View({B, T, local_n_head_, head_dim})->Transpose(1, 2);

    // (B, h_l, T, T)
    auto att = q->Matmul(k->Transpose(-2, -1)) * (1.0 / std::sqrt(head_dim));
    // (1, 1, T, T)
    auto mask = buffers_[kParamBiasName]->Slice({0, 0, 0, 0}, {1, 1, T, T}, {1, 1, 1, 1});
    // (1, 1, T, T) -> eq 0 -> (1, 1, T, T) -> masked_fill -> (B, h_l, T, T)
    att = att->MaskedFill(mask == 0, -std::numeric_limits<float>::infinity());
    // (B, h_l, T, T)
    att = nn::function::Softmax(att, -1);
    // (B, h_l, T, Dh)
    auto y = att->Matmul(v);
    // (B, h_l, T, Dh) -> (B, T, h_l, Dh) -> (B, T, local_C)
    y = y->Transpose(1, 2)->Contiguous()->View({B, T, local_C});

    // Get full tensor
    // (B, T, local_C) -> RowParallelLinear(n_embd, n_embd) -> (B, T, C)
    y = modules_[kCProjLayerName]->Forward({y})[0];
    // (B, T, C) == (bs, seq_len, n_embd)
    return {y};
}

MLP::MLP(const nn::TransformerConfig &config) {
    // c_fc: ColumnParallel (input full, output parallel)
    modules_[kCFcLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
        /*in_features=*/config.n_embd, /*out_features=*/4 * config.n_embd,
        /*bias=*/true,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    modules_[kGeluLayerName] = std::make_shared<NewGELU>();

    // c_proj: RowParallel (input parallel, output full)
    modules_[kCProjLayerName] = std::make_shared<nn::parallel::RowParallelLinear>(
        /*in_features=*/4 * config.n_embd, /*out_features=*/config.n_embd,
        /*bias=*/true,
        /*reduce_output=*/true,
        /*input_is_parallel=*/true,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MLP::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // (B, T, C) -> ColumnParallelLinear(C, 4 * C) -> (B, T, 4 * C_local)
    auto x1 = modules_[kCFcLayerName]->Forward(x);
    // (B, T, 4 * C_local) -> GELU -> (B, T, 4 * C_local)
    auto x2 = modules_[kGeluLayerName]->Forward(x1);
    // (B, T, 4 * C_local) -> RowParallelLinear(4 * C, C) -> (B, T, C)
    auto x3 = modules_[kCProjLayerName]->Forward(x2);
    // (B, T, C)
    return x3;
}

Block::Block(const nn::TransformerConfig &config) {
    modules_[kLn1LayerName] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.n_embd});
    modules_[kAttnLayerName] = std::make_shared<CausalSelfAttention>(config);
    modules_[kLn2LayerName] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.n_embd});
    modules_[kMlpLayerName] = std::make_shared<MLP>(config);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
Block::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> attention -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x1 = x[0] + modules_[kAttnLayerName]->Forward(modules_[kLn1LayerName]->Forward(x))[0];
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> MLP -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x2 = x1 + modules_[kMlpLayerName]->Forward(modules_[kLn2LayerName]->Forward({x1}))[0];
    // (bs, seq_len, n_embd)
    return {x2};
}

std::shared_ptr<nn::Module> GPT2Kernel::MakeBlock(const nn::TransformerConfig &config) {
    return std::make_shared<Block>(config);
}

std::shared_ptr<nn::Module> GPT2Kernel::MakeFinalNorm(const nn::TransformerConfig &config) {
    return std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.n_embd});
}

GPT2::GPT2(const nn::TransformerConfig &config)
    : config_(config), stage_info_(nn::parallel::PipelineParallel::GetStageInfo(
                           config_.n_layer, nn::parallel::global::GetPipelineParallelSize(), nn::parallel::pp_rank,
                           nn::parallel::global::GetVirtualPipelineParallelSize())) {
    auto tp_world_size = nn::parallel::global::GetTensorParallelSize();

    // NOTE(zbl): VocabParallelEmbedding requires vocab_size % tp_size == 0
    //            Megatron-LM has an optional argument `--make-vocab-size-divisible-by`, would do padding to vocab
    //            Here we introduce padding by default, might need modify Tokenizer correspondingly later
    CHECK_EQ(config.vocab_size % tp_world_size, 0) << "Vocab size should be divisible by TP world size";

    auto kernel = std::make_shared<GPT2Kernel>();

    std::unordered_map<std::string, std::shared_ptr<nn::Module>> transformer;
    if (stage_info_.is_first_stage) {
        modules_[kPPFirstStageName] = std::make_shared<nn::TransformerFirstStageABI>(config_, kernel);
        transformer[nn::TransformerFirstStageABI::kWTELayerName]
            = modules_[kPPFirstStageName]->mutable_module(nn::TransformerFirstStageABI::kWTELayerName);
        if (kernel->UseAbsolutePositionEmbedding()) {
            transformer[nn::TransformerFirstStageABI::kWPELayerName]
                = modules_[kPPFirstStageName]->mutable_module(nn::TransformerFirstStageABI::kWPELayerName);
        }
    }

    {
        std::map<int, std::pair<int, std::shared_ptr<nn::TransformerChunkABI>>> start_layer_to_layer_size_and_chunk;
        for (int chunk_idx = 0; chunk_idx < stage_info_.layer_ranges_per_chunk.size(); ++chunk_idx) {
            const auto [start_layer, end_layer] = stage_info_.layer_ranges_per_chunk[chunk_idx];
            auto chunk = std::make_shared<nn::TransformerChunkABI>(config_, start_layer, end_layer, kernel);
            start_layer_to_layer_size_and_chunk[start_layer] = std::make_pair(end_layer - start_layer, chunk);
        }
        std::vector<std::shared_ptr<nn::Module>> h;
        int chunk_idx = 0;
        for (auto &[start_layer, layer_size_and_chunk] : start_layer_to_layer_size_and_chunk) {
            auto [layer_size, chunk] = layer_size_and_chunk;
            for (int idx = 0; idx < layer_size; ++idx) {
                h.push_back(
                    chunk->mutable_module(nn::TransformerChunkABI::kHLayerName)->mutable_module(std::to_string(idx)));
            }
            modules_[kPPChunkNamePrefix + std::to_string(chunk_idx)] = std::move(chunk);
            ++chunk_idx;
        }
        transformer[nn::TransformerChunkABI::kHLayerName] = std::make_shared<nn::ModuleList>(std::move(h));
    }

    if (stage_info_.is_last_stage) {
        modules_[kPPLastStageName] = std::make_shared<nn::TransformerLastStageABI>(config_, kernel);
        transformer[nn::TransformerLastStageABI::kLnFLayerName]
            = modules_[kPPLastStageName]->mutable_module(nn::TransformerLastStageABI::kLnFLayerName);
        modules_[nn::TransformerLastStageABI::kLMHeadLayerName]
            = modules_[kPPLastStageName]->mutable_module(nn::TransformerLastStageABI::kLMHeadLayerName);
    }
    modules_[kTransformerLayerName] = std::make_shared<nn::ModuleDict>(std::move(transformer));

    // FIXME(jym): Assigning the parameter values of wte to LMHead, which is not real tying operation
    if (nn::parallel::global::GetPipelineParallelSize() == 1) {
        // https://paperswithcode.com/method/weight-tying
        *mutable_module(kTransformerLayerName)
             ->mutable_module(nn::TransformerFirstStageABI::kWTELayerName)
             ->mutable_parameter(nn::parallel::VocabParallelEmbedding::kParamWeightName)
            = module(nn::TransformerLastStageABI::kLMHeadLayerName)
                  .parameter(nn::parallel::ColumnParallelLinear::kParamWeightName);
    }
}

std::vector<std::shared_ptr<infini_train::Tensor>>
GPT2::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto x1 = modules_[kPPFirstStageName]->Forward(x);
    for (int chunk_idx = 0; chunk_idx < stage_info_.layer_ranges_per_chunk.size(); ++chunk_idx) {
        x1 = modules_[kPPChunkNamePrefix + std::to_string(chunk_idx)]->Forward(x1);
    }
    return modules_[kPPLastStageName]->Forward(x1);
}

std::shared_ptr<GPT2> GPT2::FromPretrained(ModelType model_type) {
    // TODO(dcj): implement this later
    LOG(FATAL) << "Not implemented yet";
    return nullptr;
}

int GPT2::GetChunkSize() const { return stage_info_.layer_ranges_per_chunk.size(); }
