#include "infini_train/include/nn/modules/transformer/transformer.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"
#include "infini_train/include/nn/modules/transformer/mlp.h"
#include "infini_train/include/nn/modules/transformer/moe/moe_layer.h"
#include "infini_train/include/nn/modules/transformer/utils.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
namespace {
parallel::StageInfo ConvertLayoutToStageInfo(parallel::LayerIndex num_layers,
                                             const std::shared_ptr<const parallel::PipelineLayout> &pipeline_layout,
                                             int stage_id) {
    if (pipeline_layout == nullptr) {
        throw std::invalid_argument("pipeline layout must not be null");
    }
    if (pipeline_layout->GetNumLayers() != num_layers) {
        throw std::invalid_argument("pipeline_layout layer count does not match TransformerConfig::n_layer");
    }

    const parallel::PipelineStageLayout &stage = pipeline_layout->GetStage(stage_id);
    if (stage.has_final_norm != stage.has_lm_head) {
        throw std::invalid_argument("final norm and LM head must belong to the same pipeline stage");
    }

    std::vector<std::pair<int, int>> layer_ranges_per_chunk;
    layer_ranges_per_chunk.reserve(stage.chunks.size());
    for (std::size_t chunk_idx = 0; chunk_idx < stage.chunks.size(); ++chunk_idx) {
        const parallel::PipelineChunkLayout &chunk = stage.chunks[chunk_idx];
        if (chunk.stage_id != stage.stage_id || chunk.global_chunk_id < 0) {
            throw std::invalid_argument("pipeline chunk identity is inconsistent with its stage");
        }
        if (chunk.local_chunk_idx != static_cast<int>(chunk_idx)) {
            throw std::invalid_argument("pipeline local chunk indices must be contiguous and start at zero");
        }
        if (chunk.layer_range.begin < 0 || chunk.layer_range.end < chunk.layer_range.begin
            || chunk.layer_range.end > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("pipeline layer range cannot be represented by TransformerChunk");
        }
        layer_ranges_per_chunk.emplace_back(static_cast<int>(chunk.layer_range.begin),
                                            static_cast<int>(chunk.layer_range.end));
    }
    return parallel::StageInfo{
        .is_first_stage = stage.has_embedding,
        .is_last_stage = stage.has_lm_head,
        .layer_ranges_per_chunk = std::move(layer_ranges_per_chunk),
    };
}
} // namespace

TransformerFirstStage::TransformerFirstStage(const TransformerConfig &config)
    : CloneableModule(kType), config_(config) {
    modules_[kWTELayerName] = std::make_shared<parallel::VocabParallelEmbedding>(
        config_.vocab_size, config_.n_embd, parallel::global::GetSequenceParallelEnabled());

    // Only learned absolute position embedding uses a trainable WPE table.
    if (config_.position_embedding_type == PositionEmbeddingType::kLearnedAbsolute) {
        modules_[kWPELayerName] = std::make_shared<Embedding>(config_.block_size, config_.n_embd);
    } else if (config_.position_embedding_type != PositionEmbeddingType::kRoPE) {
        LOG(FATAL) << "Unsupported position embedding type";
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

    // Add position embedding only for models that use learned absolute position encoding.
    if (modules_.contains(kWPELayerName)) {
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
        // For RoPE-based models (LLaMA3), no absolute position embedding is needed.
        // (B, T, C)
        return tok_emb;
    }
}

TransformerLayer::TransformerLayer(const nn::TransformerConfig &config) : CloneableModule(kType) {
    switch (config.norm_type) {
    case NormType::kLayerNorm:
        modules_[kLn1LayerName] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.n_embd});
        modules_[kLn2LayerName] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.n_embd});
        break;
    case NormType::kRMSNorm:
        modules_[kLn1LayerName] = std::make_shared<RMSNorm>(config.n_embd, config.norm_eps);
        modules_[kLn2LayerName] = std::make_shared<RMSNorm>(config.n_embd, config.norm_eps);
        break;
    default:
        LOG(FATAL) << "Unsupported norm type";
    }

    modules_[kAttnLayerName] = std::make_shared<CausalSelfAttention>(config);
    if (config.ffn_type == FFNType::kMoE) {
        modules_[kMlpLayerName] = std::make_shared<moe::MoELayer>(config);
    } else {
        modules_[kMlpLayerName] = std::make_shared<MLP>(config);
    }
}

std::vector<std::shared_ptr<Tensor>> TransformerLayer::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd)
    auto ln1_out = (*modules_[kLn1LayerName])({x[0]})[0];

    std::vector<std::shared_ptr<Tensor>> attn_input = {ln1_out};
    if (x.size() > 1) {
        attn_input.push_back(x[1]); // freqs_cis
    }
    if (x.size() > 2) {
        attn_input.push_back(x[2]); // start_pos
    }
    if (x.size() > 3) {
        attn_input.push_back(x[3]); // mask
    }

    auto attn_out = (*modules_[kAttnLayerName])(attn_input)[0];
    auto x1 = x[0] + attn_out;

    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> MLP -> (bs, seq_len, n_embd) -> Add -> (bs,
    // seq_len, n_embd)
    auto x2 = x1 + (*modules_[kMlpLayerName])((*modules_[kLn2LayerName])({x1}))[0];

    // (bs, seq_len, n_embd)
    return {x2};
}

TransformerChunk::TransformerChunk(const TransformerConfig &config, int start_layer, int end_layer)
    : CloneableModule(kType), config_(config) {
    std::vector<std::shared_ptr<nn::Module>> h;
    for (int64_t i = start_layer; i < end_layer; ++i) {
        auto layer = std::make_shared<TransformerLayer>(config);
        h.push_back(layer);
    }
    modules_[kHLayerName] = std::make_shared<nn::ModuleList>(std::move(h));
}

std::vector<std::shared_ptr<Tensor>> TransformerChunk::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    auto x1 = x[0];

    // Check if we need to pass RoPE parameters (for LLaMA3 style models).
    if (config_.position_embedding_type == PositionEmbeddingType::kRoPE) {
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
    } else if (config_.position_embedding_type == PositionEmbeddingType::kLearnedAbsolute) {
        // Learned absolute position embedding models (GPT-2 style).
        for (auto &h : *std::dynamic_pointer_cast<nn::ModuleList>(modules_[kHLayerName])) { x1 = (*h)({x1})[0]; }
    } else {
        LOG(FATAL) << "Unsupported position embedding type";
    }

    return {x1};
}

TransformerLastStage::TransformerLastStage(const TransformerConfig &config) : CloneableModule(kType), config_(config) {
    switch (config.norm_type) {
    case NormType::kLayerNorm:
        modules_[kLnFLayerName] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config_.n_embd});
        break;
    case NormType::kRMSNorm:
        modules_[kLnFLayerName] = std::make_shared<RMSNorm>(config.n_embd, config.norm_eps);
        break;
    default:
        LOG(FATAL) << "Unsupported norm type";
    }
    // NOTE(zbl): weight-tying is possible but torch script did not do so
    modules_[kLMHeadLayerName] = std::make_shared<parallel::ColumnParallelLinear>(
        /*in_features=*/config_.n_embd, /*out_features=*/config_.vocab_size,
        /*bias=*/config_.add_bias_lm_head,
        // NOTE(zbl): each rank would get sharded [B, T, V_local] as logits
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());
}

std::vector<std::shared_ptr<Tensor>> TransformerLastStage::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    // (B, T, C) -> Layernorm -> (B, T, C)
    auto x1 = (*modules_[kLnFLayerName])(x);

    // TODO(dcj): add inference-time mini-optimization
    // (B, T, C) -> Linear(C, V) -> (B, T, V)
    return (*modules_[kLMHeadLayerName])(x1);
}

TransformerModel::TransformerModel(TransformerConfig config)
    : CloneableModule(kType), config_(config), pipeline_stage_id_(parallel::pp_rank),
      stage_info_(nn::parallel::PipelineParallel::GetStageInfo(
          config_.n_layer, nn::parallel::global::GetPipelineParallelSize(), pipeline_stage_id_,
          nn::parallel::global::GetVirtualPipelineParallelSize())) {
    BuildModules();
}

TransformerModel::TransformerModel(TransformerConfig config,
                                   std::shared_ptr<const parallel::PipelineLayout> pipeline_layout,
                                   int pipeline_stage_id)
    : CloneableModule(kType), config_(std::move(config)), pipeline_layout_(std::move(pipeline_layout)),
      pipeline_stage_id_(pipeline_stage_id),
      stage_info_(ConvertLayoutToStageInfo(config_.n_layer, pipeline_layout_, pipeline_stage_id_)) {
    BuildModules();
}

void TransformerModel::BuildModules() {
    auto tp_world_size = nn::parallel::global::GetTensorParallelSize();

    // NOTE(zbl): VocabParallelEmbedding requires vocab_size % tp_size == 0
    //            Megatron-LM has an optional argument `--make-vocab-size-divisible-by`, would do padding to vocab
    //            Here we introduce padding by default, might need modify Tokenizer correspondingly later
    CHECK_EQ(config_.vocab_size % tp_world_size, 0) << "Vocab size should be divisible by TP world size";

    std::unordered_map<std::string, std::shared_ptr<nn::Module>> transformer;
    if (stage_info_.is_first_stage) {
        modules_[kPPFirstStageName] = std::make_shared<TransformerFirstStage>(config_);
        transformer[TransformerFirstStage::kWTELayerName]
            = modules_[kPPFirstStageName]->mutable_module(TransformerFirstStage::kWTELayerName);
        if (config_.position_embedding_type == PositionEmbeddingType::kLearnedAbsolute) {
            transformer[TransformerFirstStage::kWPELayerName]
                = modules_[kPPFirstStageName]->mutable_module(TransformerFirstStage::kWPELayerName);
        }
    }

    {
        std::vector<std::shared_ptr<nn::Module>> h;
        int chunk_idx = 0;
        for (const auto &[start_layer, end_layer] : stage_info_.layer_ranges_per_chunk) {
            auto chunk = std::make_shared<TransformerChunk>(config_, start_layer, end_layer);
            for (int idx = 0; idx < end_layer - start_layer; ++idx) {
                h.push_back(chunk->mutable_module(TransformerChunk::kHLayerName)->mutable_module(std::to_string(idx)));
            }
            modules_[kPPChunkNamePrefix + std::to_string(chunk_idx)] = std::move(chunk);
            ++chunk_idx;
        }
        transformer[TransformerChunk::kHLayerName] = std::make_shared<nn::ModuleList>(std::move(h));
    }

    if (stage_info_.is_last_stage) {
        modules_[kPPLastStageName] = std::make_shared<TransformerLastStage>(config_);
        transformer[TransformerLastStage::kLnFLayerName]
            = modules_[kPPLastStageName]->mutable_module(TransformerLastStage::kLnFLayerName);
        modules_[TransformerLastStage::kLMHeadLayerName]
            = modules_[kPPLastStageName]->mutable_module(TransformerLastStage::kLMHeadLayerName);
    }
    modules_[kTransformerModelName] = std::make_shared<nn::ModuleDict>(std::move(transformer));

    // FIXME(jym): Assigning the parameter values of wte to LMHead, which is not real tying operation
    // TODO: Implement real GPT-2 weight tying: make lm_head.weight share the exact same Parameter/Tensor (same
    // shared_ptr/storage) as transformer.wte.weight (pointer aliasing, not value copy), and ensure the tie is
    // applied after loading weights so it won't be overwritten. Also fix GPT2::FromLLMC() loading logic to respect
    // weight tying (do not create/load a separate lm_head.weight tensor; load once into the tied weight) so
    // parameter counting matches PyTorch/PEFT.
    if (config_.tie_weights && stage_info_.is_first_stage && stage_info_.is_last_stage) {
        // https://paperswithcode.com/method/weight-tying
        *mutable_module(kTransformerModelName)
             ->mutable_module(TransformerFirstStage::kWTELayerName)
             ->mutable_parameter(nn::parallel::VocabParallelEmbedding::kParamWeightName)
            = module(TransformerLastStage::kLMHeadLayerName)
                  .parameter(nn::parallel::ColumnParallelLinear::kParamWeightName);
    }
}

std::vector<std::shared_ptr<Tensor>> TransformerModel::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    auto x1 = (*modules_[kPPFirstStageName])(x);
    for (int chunk_idx = 0; chunk_idx < stage_info_.layer_ranges_per_chunk.size(); ++chunk_idx) {
        x1 = (*modules_[kPPChunkNamePrefix + std::to_string(chunk_idx)])(x1);
    }

    auto res = (*modules_[kPPLastStageName])(x1);
    return res;
}

} // namespace infini_train::nn
