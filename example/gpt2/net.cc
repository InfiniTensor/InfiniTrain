#include "example/gpt2/net.h"

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

CausalSelfAttention::CausalSelfAttention(const GPT2Config &config)
    : CloneableModule(kType), config_(config), n_head_(config.n_head), n_embd_(config.n_embd) {
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
    auto qkv = (*modules_[kCAttnLayerName])(x)[0]->Split(local_C, 2);

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
    y = (*modules_[kCProjLayerName])({y})[0];
    // (B, T, C) == (bs, seq_len, n_embd)
    return {y};
}

MLP::MLP(const GPT2Config &config) : CloneableModule(kType) {
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
    auto x1 = (*modules_[kCFcLayerName])(x);
    // (B, T, 4 * C_local) -> GELU -> (B, T, 4 * C_local)
    auto x2 = (*modules_[kGeluLayerName])(x1);
    // (B, T, 4 * C_local) -> RowParallelLinear(4 * C, C) -> (B, T, C)
    auto x3 = (*modules_[kCProjLayerName])(x2);
    // (B, T, C)
    return x3;
}

Block::Block(const GPT2Config &config) : CloneableModule(kType) {
    modules_[kLn1LayerName] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.n_embd});
    modules_[kAttnLayerName] = std::make_shared<CausalSelfAttention>(config);
    modules_[kLn2LayerName] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.n_embd});
    modules_[kMlpLayerName] = std::make_shared<MLP>(config);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
Block::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> attention -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x1 = x[0] + (*modules_[kAttnLayerName])((*modules_[kLn1LayerName])(x))[0];
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> MLP -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x2 = x1 + (*modules_[kMlpLayerName])((*modules_[kLn2LayerName])({x1}))[0];
    // (bs, seq_len, n_embd)
    return {x2};
}

GPT2FirstStage::GPT2FirstStage(const GPT2Config &config) : CloneableModule(kType), config_(config) {
    modules_[kWTELayerName] = std::make_shared<nn::parallel::VocabParallelEmbedding>(
        config_.vocab_size, config_.n_embd, nn::parallel::global::GetSequenceParallelEnabled());
    modules_[kWPELayerName] = std::make_shared<nn::Embedding>(config_.block_size, config_.n_embd);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
GPT2FirstStage::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &input) {
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

GPT2Chunk::GPT2Chunk(const GPT2Config &config, int start_layer, int end_layer)
    : CloneableModule(kType), config_(config) {
    std::vector<std::shared_ptr<nn::Module>> h;
    for (int64_t i = start_layer; i < end_layer; ++i) {
        auto layer = std::make_shared<Block>(config);
        h.push_back(layer);
    }
    modules_[kHLayerName] = std::make_shared<nn::ModuleList>(std::move(h));
}

std::vector<std::shared_ptr<infini_train::Tensor>>
GPT2Chunk::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto x1 = x[0];
    // (bs, seq_len, n_embd) -> transformer -> (bs, seq_len, n_embd)
    for (auto &h : *std::dynamic_pointer_cast<nn::ModuleList>(modules_[kHLayerName])) { x1 = (*h)({x1})[0]; }
    return {x1};
}

GPT2LastStage::GPT2LastStage(const GPT2Config &config) : CloneableModule(kType), config_(config) {
    modules_[kLnFLayerName] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config_.n_embd});
    // don't init this one, we will tie weights
    modules_[kLMHeadLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
        /*in_features=*/config_.n_embd, /*out_features=*/config_.vocab_size,
        /*bias=*/false,
        // NOTE(zbl): each tp_rank would get sharded [B, T, V_local] as logits
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());
}

std::vector<std::shared_ptr<infini_train::Tensor>>
GPT2LastStage::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // (B, T, C) -> Layernorm -> (B, T, C)
    auto x1 = (*modules_[kLnFLayerName])(x);

    // TODO(dcj): add inference-time mini-optimization
    // (B, T, C) -> Linear(C, V) -> (B, T, V)
    return (*modules_[kLMHeadLayerName])(x1);
}

GPT2::GPT2(const GPT2Config &config)
    : CloneableModule(kType), config_(config),
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
        modules_[kPPFirstStageName] = std::make_shared<GPT2FirstStage>(config_);
        transformer[GPT2FirstStage::kWTELayerName]
            = modules_[kPPFirstStageName]->mutable_module(GPT2FirstStage::kWTELayerName);
        transformer[GPT2FirstStage::kWPELayerName]
            = modules_[kPPFirstStageName]->mutable_module(GPT2FirstStage::kWPELayerName);
    }

    {
        std::map<int, std::pair<int, std::shared_ptr<GPT2Chunk>>> start_layer_to_layer_size_and_chunk;
        for (int chunk_idx = 0; chunk_idx < stage_info_.layer_ranges_per_chunk.size(); ++chunk_idx) {
            const auto [start_layer, end_layer] = stage_info_.layer_ranges_per_chunk[chunk_idx];
            auto chunk = std::make_shared<GPT2Chunk>(config_, start_layer, end_layer);
            start_layer_to_layer_size_and_chunk[start_layer] = std::make_pair(end_layer - start_layer, chunk);
        }
        std::vector<std::shared_ptr<nn::Module>> h;
        int chunk_idx = 0;
        for (auto &[start_layer, layer_size_and_chunk] : start_layer_to_layer_size_and_chunk) {
            auto [layer_size, chunk] = layer_size_and_chunk;
            for (int idx = 0; idx < layer_size; ++idx) {
                h.push_back(chunk->mutable_module(GPT2Chunk::kHLayerName)->mutable_module(std::to_string(idx)));
            }
            modules_[kPPChunkNamePrefix + std::to_string(chunk_idx)] = std::move(chunk);
            ++chunk_idx;
        }
        transformer[GPT2Chunk::kHLayerName] = std::make_shared<nn::ModuleList>(std::move(h));
    }

    if (stage_info_.is_last_stage) {
        modules_[kPPLastStageName] = std::make_shared<GPT2LastStage>(config_);
        transformer[GPT2LastStage::kLnFLayerName]
            = modules_[kPPLastStageName]->mutable_module(GPT2LastStage::kLnFLayerName);
        modules_[GPT2LastStage::kLMHeadLayerName]
            = modules_[kPPLastStageName]->mutable_module(GPT2LastStage::kLMHeadLayerName);
    }
    modules_[kTransformerLayerName] = std::make_shared<nn::ModuleDict>(std::move(transformer));

    // FIXME(jym): Assigning the parameter values of wte to LMHead, which is not real tying operation
    if (nn::parallel::global::GetPipelineParallelSize() == 1) {
        // https://paperswithcode.com/method/weight-tying
        *mutable_module(kTransformerLayerName)
             ->mutable_module(GPT2FirstStage::kWTELayerName)
             ->mutable_parameter(nn::parallel::VocabParallelEmbedding::kParamWeightName)
            = module(GPT2LastStage::kLMHeadLayerName).parameter(nn::parallel::ColumnParallelLinear::kParamWeightName);
    }
}

std::vector<std::shared_ptr<infini_train::Tensor>>
GPT2::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto x1 = (*modules_[kPPFirstStageName])(x);
    for (int chunk_idx = 0; chunk_idx < stage_info_.layer_ranges_per_chunk.size(); ++chunk_idx) {
        x1 = (*modules_[kPPChunkNamePrefix + std::to_string(chunk_idx)])(x1);
    }
    return (*modules_[kPPLastStageName])(x1);
}

std::shared_ptr<GPT2> GPT2::FromPretrained(ModelType model_type) {
    // TODO(dcj): implement this later
    LOG(FATAL) << "Not implemented yet";
    return nullptr;
}

namespace {
constexpr int32_t kHeaderMagic = 20240326;
constexpr int32_t kHeaderFP32Version = 3;
constexpr int32_t kHeaderBF16Version = 5;

std::tuple<int32_t, infini_train::DataType> DetermineAndCheckVersion(const std::vector<uint8_t> &header,
                                                                     size_t offset) {
    const auto version = BytesToType<uint32_t>(header, offset);
    switch (version) {
    case kHeaderBF16Version:
        return {version, infini_train::DataType::kBFLOAT16};
    case kHeaderFP32Version:
        return {version, infini_train::DataType::kFLOAT32};
    default:
        LOG(FATAL) << "Unsupported version: " << version << " at " << __FILE__ << ":" << __LINE__;
        return {}; // Unreachable, but keeps compiler happy
    }
}
} // namespace

std::shared_ptr<GPT2> GPT2::FromLLMC(const std::string &filepath) {
    if (!std::filesystem::exists(filepath)) {
        LOG(FATAL) << "File not found: " << filepath;
    }

    std::ifstream ifs(filepath, std::ios::binary);
    const auto header = ReadSeveralBytesFromIfstream(256 * sizeof(int32_t), &ifs);

    const auto magic = BytesToType<uint32_t>(header, 0);
    CHECK_EQ(magic, kHeaderMagic);
    auto [version, dtype] = DetermineAndCheckVersion(header, 4);
    CHECK_EQ(version, kHeaderFP32Version);

    auto tp_size = nn::parallel::global::GetTensorParallelSize();

    const auto block_size = BytesToType<uint32_t>(header, 8);
    const auto vocab_size = BytesToType<uint32_t>(header, 12);
    const auto n_layer = BytesToType<uint32_t>(header, 16);
    const auto n_head = BytesToType<uint32_t>(header, 20);
    const auto n_embd = BytesToType<uint32_t>(header, 24);
    const auto padded_vocab_size = BytesToType<uint32_t>(header, 28);
    // NOTE(zbl): vocab_size needs to be padded to multiple of TP size
    const auto model_vocab_size = tp_size > 1 ? padded_vocab_size : vocab_size;
    auto local_gpt2 = std::make_shared<GPT2>(GPT2Config{.block_size = block_size,
                                                        .vocab_size = model_vocab_size,
                                                        .original_vocab_size = vocab_size,
                                                        .n_layer = n_layer,
                                                        .n_head = n_head,
                                                        .n_embd = n_embd});

    LOG(INFO) << "magic: " << magic << " version: " << version << " block_size: " << block_size
              << " vocab_size: " << vocab_size << " n_layer: " << n_layer << " n_head: " << n_head
              << " n_embd: " << n_embd << " padded_vocab_size: " << padded_vocab_size;

    CHECK_EQ(n_embd % tp_size, 0) << "n_embd must be divisible by TP world size.";
    CHECK_EQ(n_embd % n_head, 0) << "n_embd must be divisible by n_head.";
    CHECK_EQ(n_head % tp_size, 0) << "n_head must be divisible by TP world size.";

    // ========== pp_size：num_stages; vpp_size: num_chunks_per_stage ==========
    int pp_size = nn::parallel::global::GetPipelineParallelSize();
    int vpp_size = nn::parallel::global::GetVirtualPipelineParallelSize();
    auto pp_rank = nn::parallel::pp_rank;
    auto [is_first_stage, is_last_stage, layer_ranges_per_chunk]
        = nn::parallel::PipelineParallel::GetStageInfo(n_layer, pp_size, pp_rank, vpp_size);
    // ========== layer to chunk ==========
    std::vector<bool> owned_layers(n_layer, false);
    for (const auto &[start, end] : layer_ranges_per_chunk) {
        for (int i = start; i < end; ++i) { owned_layers[i] = true; }
    }

    auto tp_rank = nn::parallel::tp_rank;
    // calculate xx_size_per_partition
    const int64_t vpp = model_vocab_size / tp_size;
    const int64_t v_start = static_cast<int64_t>(tp_rank) * vpp;
    const int64_t v_end = v_start + vpp;

    const int64_t qkv_out = 3 * n_embd;
    const int64_t qkv_pp = qkv_out / tp_size;
    const int64_t qkv_start = static_cast<int64_t>(tp_rank) * qkv_pp;

    const int64_t fc_out = 4 * n_embd;
    const int64_t fc_pp = fc_out / tp_size;
    const int64_t fc_start = static_cast<int64_t>(tp_rank) * fc_pp;

    const int64_t in_pp = n_embd / tp_size;        // for c_proj (row-parallel, shard on input)
    const int64_t in4_pp = (4 * n_embd) / tp_size; // for mlp.c_proj (input shard)

    auto state_dict = local_gpt2->StateDict();

    // transformer.wte.weight (also transformer.lm_head.weight)
    // full: (model_vocab_size, n_embd)
    // local: (vocab_size_per_partition, n_embd)
    if (is_first_stage) {
        auto &transformer_wte_weight
            = state_dict[std::format("{}.{}.{}", GPT2::kTransformerLayerName, GPT2FirstStage::kWTELayerName,
                                     nn::parallel::VocabParallelEmbedding::kParamWeightName)];
        ReadMatrixRowShardFloat(ifs, static_cast<float *>(transformer_wte_weight->DataPtr()), model_vocab_size, n_embd,
                                v_start, vpp);
    } else if (pp_size > 1 && is_last_stage) {
        auto &lm_head_weight = state_dict[std::format("{}.{}", GPT2LastStage::kLMHeadLayerName,
                                                      nn::parallel::ColumnParallelLinear::kParamWeightName)];
        ReadMatrixRowShardFloat(ifs, static_cast<float *>(lm_head_weight->DataPtr()), model_vocab_size, n_embd, v_start,
                                vpp);
    } else {
        size_t wte_bytes = model_vocab_size * n_embd * sizeof(float);
        ifs.seekg(wte_bytes, std::ios::cur);
    }

    if (tp_size == 1) {
        // Skip padded vocab part when TP is not enabled
        ifs.ignore((padded_vocab_size - model_vocab_size) * n_embd * sizeof(float));
    }

    if (is_first_stage) {
        // transformer.wpe.weight
        auto &transformer_wpe_weight = state_dict[std::format(
            "{}.{}.{}", GPT2::kTransformerLayerName, GPT2FirstStage::kWPELayerName, nn::Embedding::kParamWeightName)];
        ReadMatrixAllFloat(ifs, static_cast<float *>(transformer_wpe_weight->DataPtr()), block_size, n_embd);
    } else {
        size_t wpe_bytes = block_size * n_embd * sizeof(float);
        ifs.seekg(wpe_bytes, std::ios::cur);
    }

    // transformer.h.{i}.ln_1.weight
    int local_layer_index = 0;
    for (int idx = 0; idx < n_layer; ++idx) {
        if (owned_layers[idx]) {
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2Chunk::kHLayerName,
                                                  std::to_string(local_layer_index), Block::kLn1LayerName,
                                                  nn::LayerNorm::kParamWeightName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
            ++local_layer_index;
        } else {
            size_t ln_1_w_bytes = n_embd * sizeof(float);
            ifs.seekg(ln_1_w_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.ln_1.bias
    local_layer_index = 0;
    for (int idx = 0; idx < n_layer; ++idx) {
        if (owned_layers[idx]) {
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2Chunk::kHLayerName,
                                                  std::to_string(local_layer_index), Block::kLn1LayerName,
                                                  nn::LayerNorm::kParamBiasName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
            ++local_layer_index;
        } else {
            size_t ln_1_b_bytes = n_embd * sizeof(float);
            ifs.seekg(ln_1_b_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.attn.c_attn.weight (ColumnParallelLinear, but actually applies on "rows")
    local_layer_index = 0;
    for (int idx = 0; idx < n_layer; ++idx) {
        if (owned_layers[idx]) {
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName,
                                                  GPT2Chunk::kHLayerName, std::to_string(local_layer_index),
                                                  Block::kAttnLayerName, CausalSelfAttention::kCAttnLayerName,
                                                  nn::parallel::ColumnParallelLinear::kParamWeightName)];
            // NOTE(zbl): In the .bin model file, Q/K/V is concated along last dim,
            //            i.e. [Q|K|V].T = [q1|q2|...|qn|k1|k2|...|kn|v1|v2|...|vn].T
            //            However, each tp_rank needs to get [q_i|k_i|v_i].T, so we need to jump and read them
            //            respectively
            float *dst = static_cast<float *>(tensor->DataPtr());
            const int64_t local_C = n_embd / tp_size;
            const int64_t rows_all = 3 * n_embd;
            const int64_t cols_all = n_embd;
            const std::streampos base_pos = ifs.tellg();
            // Read q_i -> write to dst rows of [0 : local_C)
            ifs.seekg(base_pos);
            ReadMatrixRowShardFloat(ifs,
                                    /*dst=*/dst + (0 * local_C) * cols_all,
                                    /*rows=*/rows_all, /*cols=*/cols_all,
                                    /*row_start=*/tp_rank * local_C, /*row_cnt=*/local_C);
            // Read k_i -> write to dst rows of [local_C : 2*local_C)
            ifs.seekg(base_pos);
            ReadMatrixRowShardFloat(ifs,
                                    /*dst=*/dst + (1 * local_C) * cols_all,
                                    /*rows=*/rows_all, /*cols=*/cols_all,
                                    /*row_start=*/n_embd + tp_rank * local_C, /*row_cnt=*/local_C);
            // Read v_i -> write to dst rows of [2*local_C : 3*local_C)
            ifs.seekg(base_pos);
            ReadMatrixRowShardFloat(ifs,
                                    /*dst=*/dst + (2 * local_C) * cols_all,
                                    /*rows=*/rows_all, /*cols=*/cols_all,
                                    /*row_start=*/2 * n_embd + tp_rank * local_C, /*row_cnt=*/local_C);

            ++local_layer_index;
        } else {
            size_t c_attn_w_bytes = qkv_out * n_embd * sizeof(float);
            ifs.seekg(c_attn_w_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.attn.c_attn.bias (ColumnParallelLinear)
    local_layer_index = 0;
    for (int idx = 0; idx < n_layer; ++idx) {
        if (owned_layers[idx]) {
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName,
                                                  GPT2Chunk::kHLayerName, std::to_string(local_layer_index),
                                                  Block::kAttnLayerName, CausalSelfAttention::kCAttnLayerName,
                                                  nn::parallel::ColumnParallelLinear::kParamBiasName)];
            // NOTE(zbl): Same as c_attn.weight, the bias for Q/K/V is concated
            //            i.e. [Q|K|V] = [q1|q2|...|qn|k1|k2|...|kn|v1|v2|...|vn]
            //            However, each tp_rank needs to get [q_i|k_i|v_i], so we need to jump and read them
            //            respectively
            float *dst = static_cast<float *>(tensor->DataPtr());
            const int64_t local_C = n_embd / tp_size;
            const int64_t len_all = 3 * n_embd;
            const std::streampos base_pos = ifs.tellg();
            // Read q_i
            ifs.seekg(base_pos);
            ReadVectorShardFloat(ifs,
                                 /*dst=*/dst + (0 * local_C),
                                 /*len=*/len_all,
                                 /*start=*/tp_rank * local_C, /*cnt=*/local_C);
            // Read k_i
            ifs.seekg(base_pos);
            ReadVectorShardFloat(ifs,
                                 /*dst=*/dst + (1 * local_C),
                                 /*len=*/len_all,
                                 /*start=*/n_embd + tp_rank * local_C, /*cnt=*/local_C);
            // Read v_i
            ifs.seekg(base_pos);
            ReadVectorShardFloat(ifs,
                                 /*dst=*/dst + (2 * local_C),
                                 /*len=*/len_all,
                                 /*start=*/2 * n_embd + tp_rank * local_C, /*cnt=*/local_C);

            ++local_layer_index;
        } else {
            size_t c_attn_b_bytes = qkv_out * sizeof(float);
            ifs.seekg(c_attn_b_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.attn.c_proj.weight (RowParallelLinear, but actually applies on "columns")
    local_layer_index = 0;
    for (int idx = 0; idx < n_layer; ++idx) {
        if (owned_layers[idx]) {
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName,
                                                  GPT2Chunk::kHLayerName, std::to_string(local_layer_index),
                                                  Block::kAttnLayerName, CausalSelfAttention::kCProjLayerName,
                                                  nn::parallel::RowParallelLinear::kParamWeightName)];
            ReadMatrixColShardFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd, n_embd, tp_rank * in_pp,
                                    in_pp);
            ++local_layer_index;
        } else {
            size_t c_proj_w_bytes = n_embd * n_embd * sizeof(float);
            ifs.seekg(c_proj_w_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.attn.c_proj.bias (RowParallelLinear, no shard on bias)
    local_layer_index = 0;
    for (int idx = 0; idx < n_layer; ++idx) {
        if (owned_layers[idx]) {
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName,
                                                  GPT2Chunk::kHLayerName, std::to_string(local_layer_index),
                                                  Block::kAttnLayerName, CausalSelfAttention::kCProjLayerName,
                                                  nn::parallel::RowParallelLinear::kParamBiasName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
            ++local_layer_index;
        } else {
            size_t c_proj_b_bytes = n_embd * sizeof(float);
            ifs.seekg(c_proj_b_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.ln_2.weight
    local_layer_index = 0;
    for (int idx = 0; idx < n_layer; ++idx) {
        if (owned_layers[idx]) {
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2Chunk::kHLayerName,
                                                  std::to_string(local_layer_index), Block::kLn2LayerName,
                                                  nn::LayerNorm::kParamWeightName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
            ++local_layer_index;
        } else {
            size_t ln_2_w_bytes = n_embd * sizeof(float);
            ifs.seekg(ln_2_w_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.ln_2.bias
    local_layer_index = 0;
    for (int idx = 0; idx < n_layer; ++idx) {
        if (owned_layers[idx]) {
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2Chunk::kHLayerName,
                                                  std::to_string(local_layer_index), Block::kLn2LayerName,
                                                  nn::LayerNorm::kParamBiasName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
            ++local_layer_index;
        } else {
            size_t ln_2_b_bytes = n_embd * sizeof(float);
            ifs.seekg(ln_2_b_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.mlp.c_fc.weight (ColumnParallelLinear, but actually applies on "rows")
    local_layer_index = 0;
    for (int idx = 0; idx < n_layer; ++idx) {
        if (owned_layers[idx]) {
            auto &tensor
                = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2Chunk::kHLayerName,
                                         std::to_string(local_layer_index), Block::kMlpLayerName, MLP::kCFcLayerName,
                                         nn::parallel::ColumnParallelLinear::kParamWeightName)];
            ReadMatrixRowShardFloat(ifs, static_cast<float *>(tensor->DataPtr()), fc_out, n_embd, fc_start, fc_pp);
            ++local_layer_index;
        } else {
            size_t c_fc_w_bytes = fc_out * n_embd * sizeof(float);
            ifs.seekg(c_fc_w_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.mlp.c_fc.bias (ColumnParallelLinear)
    local_layer_index = 0;
    for (int idx = 0; idx < n_layer; ++idx) {
        if (owned_layers[idx]) {
            auto &tensor
                = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2Chunk::kHLayerName,
                                         std::to_string(local_layer_index), Block::kMlpLayerName, MLP::kCFcLayerName,
                                         nn::parallel::ColumnParallelLinear::kParamBiasName)];
            ReadVectorShardFloat(ifs, static_cast<float *>(tensor->DataPtr()), fc_out, fc_start, fc_pp);
            ++local_layer_index;
        } else {
            size_t c_fc_b_bytes = fc_out * sizeof(float);
            ifs.seekg(c_fc_b_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.mlp.c_proj.weight (RowParallelLinear, but actually applies on "columns")
    local_layer_index = 0;
    for (int idx = 0; idx < n_layer; ++idx) {
        if (owned_layers[idx]) {
            auto &tensor
                = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2Chunk::kHLayerName,
                                         std::to_string(local_layer_index), Block::kMlpLayerName, MLP::kCProjLayerName,
                                         nn::parallel::RowParallelLinear::kParamWeightName)];
            ReadMatrixColShardFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd, fc_out, tp_rank * in4_pp,
                                    in4_pp);
            ++local_layer_index;
        } else {
            size_t c_proj_w_bytes = fc_out * n_embd * sizeof(float);
            ifs.seekg(c_proj_w_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.mlp.c_proj.bias (RowParallelLinear, no shard on bias)
    local_layer_index = 0;
    for (int idx = 0; idx < n_layer; ++idx) {
        if (owned_layers[idx]) {
            auto &tensor
                = state_dict[std::format("{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, GPT2Chunk::kHLayerName,
                                         std::to_string(local_layer_index), Block::kMlpLayerName, MLP::kCProjLayerName,
                                         nn::parallel::RowParallelLinear::kParamBiasName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
            ++local_layer_index;
        } else {
            size_t c_proj_b_bytes = n_embd * sizeof(float);
            ifs.seekg(c_proj_b_bytes, std::ios::cur);
        }
    }

    if (is_last_stage) {
        // transformer.ln_f.weight
        auto &transformer_ln_f_weight = state_dict[std::format(
            "{}.{}.{}", GPT2::kTransformerLayerName, GPT2LastStage::kLnFLayerName, nn::LayerNorm::kParamWeightName)];
        ReadVectorAllFloat(ifs, static_cast<float *>(transformer_ln_f_weight->DataPtr()), n_embd);
        // transformer.ln_f.bias
        auto &transformer_ln_f_bias = state_dict[std::format(
            "{}.{}.{}", GPT2::kTransformerLayerName, GPT2LastStage::kLnFLayerName, nn::LayerNorm::kParamBiasName)];
        ReadVectorAllFloat(ifs, static_cast<float *>(transformer_ln_f_bias->DataPtr()), n_embd);
    } else {
        size_t ln_f_w_bytes = n_embd * sizeof(float);
        size_t ln_f_b_bytes = n_embd * sizeof(float);
        ifs.seekg(ln_f_w_bytes + ln_f_b_bytes, std::ios::cur);
    }
    return local_gpt2;
}

int GPT2::GetChunkSize() const { return stage_info_.layer_ranges_per_chunk.size(); }
