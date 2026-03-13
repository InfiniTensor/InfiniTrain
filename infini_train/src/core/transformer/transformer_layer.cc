#include "infini_train/include/core/transformer/transformer_layer.h"

#include <cmath>
#include <map>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_block.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"
#include "third_party/glog/src/glog/logging.h"

using namespace infini_train;

namespace infini_train::nn {

TransformerFirstStage::TransformerFirstStage(const TransformerConfig &config, const ModuleSpec &spec)
    : CloneableModule(kType), config_(config), spec_(spec) {
    // Build token embedding (required for all models)
    modules_[kWTELayerName] = build_module(config, spec.submodules_.at(kWTELayerName));

    // Build position embedding only for models that use absolute position encoding
    // LLaMA3 use RoPE, so they don't need position embedding
    if (config_.attention_type == AttentionType::kStandard) {
        modules_[kWPELayerName] = build_module(config, spec.submodules_.at(kWPELayerName));
    }
}

std::vector<std::shared_ptr<Tensor>> TransformerFirstStage::Forward(const std::vector<std::shared_ptr<Tensor>> &input) {
    // (B, T)
    auto x1 = input[0];
    CHECK_LE(x1->Dims()[1], config_.block_size)
        << "Cannot forward sequence of length " << x1->Dims()[1] << ", block size is only " << config_.block_size;
    const auto device = x1->GetDevice();

    // (B, T) -> Embedding(V_local, C) -> (B, T, C)
    auto tok_emb = (*modules_[kWTELayerName])({x1});

    // Add position embedding only for models that use absolute position encoding
    if (config_.attention_type == AttentionType::kStandard) {
        // (T_local)
        // NOTE(zbl): Slice pos sequence when SP is enabled
        auto tp_world_size = nn::parallel::global::GetTensorParallelSize();
        auto sequence_parallel_enabled = nn::parallel::global::GetSequenceParallelEnabled();
        int tp_rank = 0;
        if (tp_world_size > 1) {
            auto tp_group = nn::parallel::ProcessGroupFactory::Instance()->Get(
                nn::parallel::GetTensorParallelProcessGroupName(device.Rank().GlobalRank()));
            tp_rank = tp_group->GetGroupRank(device.Rank().GlobalRank());
        }
        int64_t t_local = sequence_parallel_enabled ? x1->Dims()[1] / tp_world_size : x1->Dims()[1];
        int64_t start = sequence_parallel_enabled ? tp_rank * t_local : 0;
        auto pos = nn::init::Arange(start, start + t_local, infini_train::DataType::kINT64, device);

        // (T) -> Embedding(T_max, C) -> (T, C)
        auto pos_emb = (*modules_[kWPELayerName])({pos});
        // (B, T, C)
        return {tok_emb[0] + pos_emb[0]};
    } else {
        // For RoPE-based models (LLaMA3), no position embedding needed
        // (B, T, C)
        return tok_emb;
    }
}

TransformerChunk::TransformerChunk(const TransformerConfig &config, int start_layer, int end_layer,
                                   const ModuleSpec &spec)
    : CloneableModule(kType), config_(config), spec_(spec) {
    std::vector<std::shared_ptr<nn::Module>> h;
    for (int64_t i = start_layer; i < end_layer; ++i) {
        auto layer = std::make_shared<TransformerBlock>(config, spec);
        h.push_back(layer);
    }
    modules_[kHLayerName] = std::make_shared<nn::ModuleList>(std::move(h));
}

std::vector<std::shared_ptr<Tensor>> TransformerChunk::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    auto x1 = x[0];

    // Check if we need to pass RoPE parameters (for LLaMA3 style models)
    if (config_.attention_type == AttentionType::kRoPE) {
        // For RoPE models, we need to prepare freqs_cis and potentially other parameters
        const auto device = x1->GetDevice();

        // Init freqs_cis on device only once
        if (buffers_[kFreqsCisName] == nullptr) {
            int64_t head_dim = config_.n_embd / config_.n_head;
            buffers_[kFreqsCisName] = PrecomputeFreqsCis(head_dim, config_.block_size * 2, config_.rope_theta,
                                                         config_.use_scaled_rope, device);
        }

        const auto t = x1->Dims()[1] * nn::parallel::global::GetSequenceParallelSize(); // full_seq_len

        // Dynamic start_pos (set to 0 for now)
        int64_t start_pos = 0;
        auto freqs_view = buffers_[kFreqsCisName]->Slice(0, start_pos, start_pos + t, 1);

        // Create causal mask
        std::shared_ptr<Tensor> ones = std::make_shared<Tensor>(nn::function::Ones({t, t})->To(device));
        std::shared_ptr<Tensor> mask = nn::function::Triu(ones, 1)->View({1, 1, t, t});

        std::shared_ptr<Tensor> start_pos_ptr = nullptr;

        // Pass RoPE parameters to each transformer block
        for (auto &h : *std::dynamic_pointer_cast<nn::ModuleList>(modules_[kHLayerName])) {
            x1 = (*h)({x1, freqs_view, start_pos_ptr, mask})[0];
        }
    } else {
        // Standard attention (GPT2 style)
        for (auto &h : *std::dynamic_pointer_cast<nn::ModuleList>(modules_[kHLayerName])) { x1 = (*h)({x1})[0]; }
    }

    return {x1};
}

// Add RoPE helper method to TransformerChunk
std::shared_ptr<Tensor> TransformerChunk::PrecomputeFreqsCis(int64_t dim, int64_t end, float theta, bool use_scaled,
                                                             infini_train::Device device) {
    auto dtype = DataType::kFLOAT32;
    CHECK_GE(dim, 2) << "dim must be >= 2 for slicing";

    auto arange = nn::init::Arange(0, dim, dtype, device)->Slice(0, 0, dim, 2);
    auto freqs = 1.0f / nn::function::Pow(theta, arange / float(dim));
    // TODO(zbl): use_scaled
    // if (use_scaled) {
    //     freqs = ApplyScaling(freqs, 8192.0f);
    // }
    auto t = nn::init::Arange(0, end, dtype, device);
    // (end, dim / 2)
    auto freqs_outer = t->Outer(freqs);
    auto cos = nn::function::Cos(freqs_outer);
    auto sin = nn::function::Sin(freqs_outer);
    // NOTE(zbl): torch script uses cis expression, here use stack
    // (end, dim / 2, 2)
    auto freqs_cis = nn::function::Stack(std::vector<std::shared_ptr<Tensor>>{cos, sin}, -1)->Contiguous();

    return freqs_cis;
}

TransformerLastStage::TransformerLastStage(const TransformerConfig &config, const ModuleSpec &spec)
    : CloneableModule(kType), config_(config), spec_(spec) {
    CHECK(spec.submodules_.contains(kLnFLayerName)) << "TransformerLastStage spec missing submodule: " << kLnFLayerName;
    CHECK(spec.submodules_.contains(kLMHeadLayerName))
        << "TransformerLastStage spec missing submodule: " << kLMHeadLayerName;
    modules_[kLnFLayerName] = build_module(config, spec.submodules_.at(kLnFLayerName));
    modules_[kLMHeadLayerName] = build_module(config, spec.submodules_.at(kLMHeadLayerName));
}

std::vector<std::shared_ptr<Tensor>> TransformerLastStage::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    // (B, T, C) -> Layernorm -> (B, T, C)
    auto x1 = (*modules_[kLnFLayerName])(x);

    // TODO(dcj): add inference-time mini-optimization
    // (B, T, C) -> Linear(C, V) -> (B, T, V)
    return (*modules_[kLMHeadLayerName])(x1);
}

TransformerLayer::TransformerLayer(const TransformerConfig config, const ModuleSpec &spec
                                   /*, const std::unordered_map<std::string, std::any> &params*/)
    : CloneableModule(kType), config_(config), spec_(spec),
      stage_info_(nn::parallel::PipelineParallel::GetStageInfo(
          config_.n_layer, nn::parallel::global::GetPipelineParallelSize(), nn::parallel::pp_rank,
          nn::parallel::global::GetVirtualPipelineParallelSize())) {

    auto tp_world_size = nn::parallel::global::GetTensorParallelSize();

    // NOTE(zbl): VocabParallelEmbedding requires vocab_size % tp_size == 0
    //            Megatron-LM has an optional argument `--make-vocab-size-divisible-by`, would do padding to vocab
    //            Here we introduce padding by default, might need modify Tokenizer correspondingly later
    CHECK_EQ(config.vocab_size % tp_world_size, 0) << "Vocab size should be divisible by TP world size";

    std::unordered_map<std::string, std::shared_ptr<nn::Module>> transformer;
    if (stage_info_.is_first_stage) {
        modules_[kPPFirstStageName]
            = std::make_shared<TransformerFirstStage>(config_, spec_.submodules_.at("first_stage"));
        transformer[TransformerFirstStage::kWTELayerName]
            = modules_[kPPFirstStageName]->mutable_module(TransformerFirstStage::kWTELayerName);
        if (config_.attention_type == AttentionType::kStandard) {
            transformer[TransformerFirstStage::kWPELayerName]
                = modules_[kPPFirstStageName]->mutable_module(TransformerFirstStage::kWPELayerName);
        }
    }

    {
        std::map<int, std::pair<int, std::shared_ptr<TransformerChunk>>> start_layer_to_layer_size_and_chunk;
        for (int chunk_idx = 0; chunk_idx < stage_info_.layer_ranges_per_chunk.size(); ++chunk_idx) {
            const auto [start_layer, end_layer] = stage_info_.layer_ranges_per_chunk[chunk_idx];
            auto chunk
                = std::make_shared<TransformerChunk>(config_, start_layer, end_layer, spec_.submodules_.at("block"));
            start_layer_to_layer_size_and_chunk[start_layer] = std::make_pair(end_layer - start_layer, chunk);
        }
        std::vector<std::shared_ptr<nn::Module>> h;
        int chunk_idx = 0;
        for (auto &[start_layer, layer_size_and_chunk] : start_layer_to_layer_size_and_chunk) {
            auto [layer_size, chunk] = layer_size_and_chunk;
            for (int idx = 0; idx < layer_size; ++idx) {
                h.push_back(chunk->mutable_module(TransformerChunk::kHLayerName)->mutable_module(std::to_string(idx)));
            }
            modules_[kPPChunkNamePrefix + std::to_string(chunk_idx)] = std::move(chunk);
            ++chunk_idx;
        }
        transformer[TransformerChunk::kHLayerName] = std::make_shared<nn::ModuleList>(std::move(h));
    }

    if (stage_info_.is_last_stage) {
        modules_[kPPLastStageName]
            = std::make_shared<TransformerLastStage>(config_, spec_.submodules_.at("last_stage"));
        transformer[TransformerLastStage::kLnFLayerName]
            = modules_[kPPLastStageName]->mutable_module(TransformerLastStage::kLnFLayerName);
        modules_[TransformerLastStage::kLMHeadLayerName]
            = modules_[kPPLastStageName]->mutable_module(TransformerLastStage::kLMHeadLayerName);
    }
    modules_[kTransformerLayerName] = std::make_shared<nn::ModuleDict>(std::move(transformer));

    // Weight tying based on config (only for models that support it, e.g., GPT2)
    // https://paperswithcode.com/method/weight-tying
    if (config_.tie_weights && nn::parallel::global::GetPipelineParallelSize() == 1) {
        *mutable_module(kTransformerLayerName)
             ->mutable_module(TransformerFirstStage::kWTELayerName)
             ->mutable_parameter(nn::parallel::VocabParallelEmbedding::kParamWeightName)
            = module(TransformerLastStage::kLMHeadLayerName)
                  .parameter(nn::parallel::ColumnParallelLinear::kParamWeightName);
    }
}

std::vector<std::shared_ptr<Tensor>> TransformerLayer::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    auto x1 = (*modules_[kPPFirstStageName])(x);
    for (int chunk_idx = 0; chunk_idx < stage_info_.layer_ranges_per_chunk.size(); ++chunk_idx) {
        x1 = (*modules_[kPPChunkNamePrefix + std::to_string(chunk_idx)])(x1);
    }
    return (*modules_[kPPLastStageName])(x1);
}

} // namespace infini_train::nn
