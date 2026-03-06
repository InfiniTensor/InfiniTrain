#include "infini_train/include/core/transformer/transformer_layer.h"

#include <map>

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_block.h"
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

namespace infini_train::nn {

TransformerFirstStage::TransformerFirstStage(const TransformerConfig &config, const ModuleSpec &spec)
    : CloneableModule(kType), config_(config), spec_(spec) {
    modules_[kWTELayerName] = build_module(config, spec.submodules_.at(kWTELayerName));
    modules_[kWPELayerName] = build_module(config, spec.submodules_.at(kWPELayerName));
}

std::vector<std::shared_ptr<infini_train::Tensor>>
TransformerFirstStage::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &input) {
    // (B, T)
    auto x1 = input[0];
    CHECK_LE(x1->Dims()[1], config_.block_size)
        << "Cannot forward sequence of length " << x1->Dims()[1] << ", block size is only " << config_.block_size;
    const auto device = x1->GetDevice();

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

    // (B, T) -> Embedding(V_local, C) -> (B, T, C)
    auto tok_emb = (*modules_[kWTELayerName])({x1})[0];

    // (T) -> Embedding(T_max, C) -> (T, C)
    auto pos_emb = (*modules_[kWPELayerName])({pos})[0];
    // (B, T, C)
    return {tok_emb + pos_emb};
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

std::vector<std::shared_ptr<infini_train::Tensor>>
TransformerChunk::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto x1 = x[0];
    // (bs, seq_len, n_embd) -> transformer -> (bs, seq_len, n_embd)
    for (auto &h : *std::dynamic_pointer_cast<nn::ModuleList>(modules_[kHLayerName])) { x1 = (*h)({x1})[0]; }
    return {x1};
}

TransformerLastStage::TransformerLastStage(const TransformerConfig &config, const ModuleSpec &spec)
    : CloneableModule(kType), config_(config), spec_(spec) {
    modules_[kLnFLayerName] = build_module(config, spec.submodules_.at(kLnFLayerName));
    modules_[kLMHeadLayerName] = build_module(config, spec.submodules_.at(kLMHeadLayerName));
}

std::vector<std::shared_ptr<infini_train::Tensor>>
TransformerLastStage::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
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
        transformer[TransformerFirstStage::kWPELayerName]
            = modules_[kPPFirstStageName]->mutable_module(TransformerFirstStage::kWPELayerName);
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

    // FIXME(jym): Assigning the parameter values of wte to LMHead, which is not real tying operation
    if (nn::parallel::global::GetPipelineParallelSize() == 1) {
        // https://paperswithcode.com/method/weight-tying
        *mutable_module(kTransformerLayerName)
             ->mutable_module(TransformerFirstStage::kWTELayerName)
             ->mutable_parameter(nn::parallel::VocabParallelEmbedding::kParamWeightName)
            = module(TransformerLastStage::kLMHeadLayerName)
                  .parameter(nn::parallel::ColumnParallelLinear::kParamWeightName);
    }
}

std::vector<std::shared_ptr<infini_train::Tensor>>
TransformerLayer::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto x1 = (*modules_[kPPFirstStageName])(x);
    for (int chunk_idx = 0; chunk_idx < stage_info_.layer_ranges_per_chunk.size(); ++chunk_idx) {
        x1 = (*modules_[kPPChunkNamePrefix + std::to_string(chunk_idx)])(x1);
    }
    return (*modules_[kPPLastStageName])(x1);
}

static bool wte_registered = []() {
    ModuleRegistry::Instance().Register(
        typeid(nn::parallel::VocabParallelEmbedding),
        [](const TransformerConfig &config, const ModuleSpec &) -> std::shared_ptr<Module> {
            return std::make_shared<nn::parallel::VocabParallelEmbedding>(
                config.vocab_size, config.n_embd, nn::parallel::global::GetSequenceParallelEnabled());
        });
    return true;
}();

static bool wpe_registered = []() {
    ModuleRegistry::Instance().Register(
        typeid(nn::Embedding), [](const TransformerConfig &config, const ModuleSpec &) -> std::shared_ptr<Module> {
            return std::make_shared<nn::Embedding>(config.block_size, config.n_embd);
        });
    return true;
}();

static bool ln_f_registered = []() {
    ModuleRegistry::Instance().Register(
        typeid(LayerNorm), [](const TransformerConfig &config, const ModuleSpec &) -> std::shared_ptr<Module> {
            return std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.n_embd});
        });
    return true;
}();

} // namespace infini_train::nn