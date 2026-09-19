#include "example/llama3/checkpoint_loader.h"
#include "example/common/parser.h"
#include <optional>
#include <span>
#include <utility>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"
#include "infini_train/include/nn/modules/transformer/mlp.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

#include "example/common/utils.h"
#include "example/llama3/config.h"

using namespace infini_train;
namespace nn = infini_train::nn;

namespace {
constexpr int kRandomSeed = 42;

// TODO(zbl): make this rng generator compatible with torch later
static std::mt19937 gen{kRandomSeed};
} // namespace

namespace {
constexpr int32_t kLLaMA3Magic = 20240803;
constexpr int32_t kLLaMA3FP32Version = 3;
} // namespace

namespace llama3 {
namespace {
using LayerIndex = nn::parallel::LayerIndex;
using PipelineLayout = nn::parallel::PipelineLayout;

std::shared_ptr<nn::TransformerModel>
LoadFromLLMCImpl(const std::string &filepath,
                 std::optional<infini_train::examples::PipelineLayoutRequest> layout_request) {
    if (!std::filesystem::exists(filepath)) {
        LOG(FATAL) << "File not found: " << filepath;
    }

    std::ifstream ifs(filepath, std::ios::binary);
    const auto header = ReadSeveralBytesFromIfstream(256 * sizeof(int32_t), &ifs);

    const auto magic = BytesToType<uint32_t>(header, 0);
    CHECK_EQ(magic, kLLaMA3Magic);
    const auto version = BytesToType<uint32_t>(header, 4);
    CHECK_EQ(version, kLLaMA3FP32Version);

    const auto block_size = BytesToType<uint32_t>(header, 8);
    const auto vocab_size = BytesToType<uint32_t>(header, 12);
    const auto n_layer = BytesToType<uint32_t>(header, 16);
    const auto n_head = BytesToType<uint32_t>(header, 20);
    const auto n_kv_head = BytesToType<uint32_t>(header, 24);
    const auto n_embd = BytesToType<uint32_t>(header, 28);
    const auto ffn_dim_multiplier = BytesToType<float>(header, 32);
    const auto multiple_of = BytesToType<uint32_t>(header, 36);
    const auto norm_eps = BytesToType<float>(header, 40);
    const auto rope_theta = BytesToType<float>(header, 44);
    const auto use_scaled_rope = BytesToType<int32_t>(header, 48);
    const auto max_gen_bs = BytesToType<int32_t>(header, 52);
    const auto version_major = BytesToType<int32_t>(header, 56);
    const auto version_minor = BytesToType<int32_t>(header, 60);

    nn::TransformerConfig llama3_config = llama3::LLaMA3Config();
    llama3_config.block_size = block_size;
    llama3_config.vocab_size = vocab_size;
    llama3_config.original_vocab_size = vocab_size;
    llama3_config.n_layer = n_layer;
    llama3_config.n_head = n_head;
    llama3_config.n_kv_head = n_kv_head;
    llama3_config.n_embd = n_embd;
    llama3_config.ffn_dim_multiplier = ffn_dim_multiplier;
    llama3_config.multiple_of = multiple_of;
    llama3_config.rope_theta = rope_theta;
    llama3_config.use_scaled_rope = static_cast<bool>(use_scaled_rope);
    llama3_config.norm_eps = norm_eps;
    llama3_config.max_gen_batch_size = max_gen_bs;
    llama3::SanitizeLLaMA3Config(llama3_config);
    const int pp_size = nn::parallel::global::GetPipelineParallelSize();
    const int vpp_size = nn::parallel::global::GetVirtualPipelineParallelSize();
    const int pp_rank = nn::parallel::pp_rank;

    std::shared_ptr<const PipelineLayout> pipeline_layout;
    std::shared_ptr<nn::TransformerModel> llama3;
    if (layout_request.has_value()) {
        pipeline_layout = infini_train::examples::ResolvePipelineLayout(static_cast<LayerIndex>(n_layer), pp_size,
                                                                        vpp_size, *layout_request);
        llama3 = pipeline_layout ? std::make_shared<nn::TransformerModel>(llama3_config, pipeline_layout, pp_rank)
                                 : std::make_shared<nn::TransformerModel>(llama3_config);
    } else {
        llama3 = std::make_shared<nn::TransformerModel>(llama3_config);
    }

    bool has_embedding = false;
    bool has_final_norm = false;
    bool has_lm_head = false;
    std::vector<bool> owned_layers(n_layer, false);
    std::vector<LayerIndex> local_indices(n_layer, -1);
    std::vector<std::pair<LayerIndex, LayerIndex>> stage_ranges;

    if (pipeline_layout) {
        const auto &stage = pipeline_layout->GetStage(pp_rank);
        has_embedding = stage.has_embedding;
        has_final_norm = stage.has_final_norm;
        has_lm_head = stage.has_lm_head;
        for (const auto &chunk : stage.chunks) {
            stage_ranges.emplace_back(chunk.layer_range.begin, chunk.layer_range.end);
            for (LayerIndex layer = chunk.layer_range.begin; layer < chunk.layer_range.end; ++layer) {
                const auto index = static_cast<std::size_t>(layer);
                owned_layers.at(index) = true;
                local_indices.at(index) = pipeline_layout->GetLocalLayerIndex(pp_rank, layer);
            }
        }
    } else {
        const auto legacy = nn::parallel::PipelineParallel::GetStageInfo(n_layer, pp_size, pp_rank, vpp_size);
        has_embedding = legacy.is_first_stage;
        has_final_norm = legacy.is_last_stage;
        has_lm_head = legacy.is_last_stage;
        for (const auto &[begin, end] : legacy.layer_ranges_per_chunk) {
            stage_ranges.emplace_back(begin, end);
            for (int layer = begin; layer < end; ++layer) { owned_layers.at(static_cast<std::size_t>(layer)) = true; }
        }
        LayerIndex local_index = 0;
        for (std::size_t layer = 0; layer < owned_layers.size(); ++layer) {
            if (owned_layers[layer]) {
                local_indices[layer] = local_index++;
            }
        }
    }

    const int tp_size = nn::parallel::global::GetTensorParallelSize();
    const int tp_rank = nn::parallel::tp_rank;

    CHECK_EQ(n_embd % tp_size, 0) << "n_embd must be divisible by TP world size.";
    CHECK_EQ(n_head % tp_size, 0) << "n_head must be divisible by TP world size.";
    CHECK_EQ(n_kv_head % tp_size, 0) << "n_kv_head must be divisible by TP world size.";
    CHECK_EQ(vocab_size % tp_size, 0) << "vocab_size must be divisible by TP world size.";

    if (tp_rank == 0) {
        LOG(INFO) << "Model Config:";
        LOG(INFO) << "  block_size         = " << block_size;
        LOG(INFO) << "  vocab_size         = " << vocab_size;
        LOG(INFO) << "  n_layer            = " << n_layer;
        LOG(INFO) << "  n_head             = " << n_head;
        LOG(INFO) << "  n_kv_head          = " << n_kv_head;
        LOG(INFO) << "  n_embd             = " << n_embd;
        LOG(INFO) << "  ffn_dim_multiplier = " << ffn_dim_multiplier;
        LOG(INFO) << "  multiple_of        = " << multiple_of;
        LOG(INFO) << "  norm_eps           = " << norm_eps;
        LOG(INFO) << "  rope_theta         = " << rope_theta;
        LOG(INFO) << "  use_scaled_rope    = " << use_scaled_rope;
        LOG(INFO) << "  max_gen_bs         = " << max_gen_bs;
        LOG(INFO) << "  version_major      = " << version_major;
        LOG(INFO) << "  version_minor      = " << version_minor;

        LOG(INFO) << "Pipeline Parallel Chunks:";
        for (size_t i = 0; i < stage_ranges.size(); ++i) {
            LOG(INFO) << "  Chunk " << i << ": layers " << stage_ranges[i].first << " to " << stage_ranges[i].second;
        }
    }

    const int64_t head_dim = static_cast<int64_t>(n_embd) / static_cast<int64_t>(n_head);

    // nn::MLP hidden dim calculation in LLaMA-3
    auto round_up_to = [](int64_t x, int64_t m) { return (x + m - 1) / m * m; };
    int64_t hidden_dim = 4LL * static_cast<int64_t>(n_embd);
    hidden_dim = (2LL * hidden_dim) / 3LL;
    if (ffn_dim_multiplier > 0.0f) {
        hidden_dim = static_cast<int64_t>(
            std::llround(static_cast<double>(ffn_dim_multiplier) * static_cast<double>(hidden_dim)));
    }

    int64_t ffn_hidden = round_up_to(hidden_dim, static_cast<int64_t>(multiple_of));

    // ===== Per-rank sizes / offsets =====
    // vocab parallel
    const int64_t vpp = static_cast<int64_t>(vocab_size) / tp_size;
    const int64_t v_start = static_cast<int64_t>(tp_rank) * vpp;

    // attention Q/K/V packed as rows: [Q | K | V]
    const int64_t q_out_rows = static_cast<int64_t>(n_embd);
    const int64_t kv_out_rows = static_cast<int64_t>(n_kv_head) * head_dim; // for K or V (each)
    const int64_t attn_rows_all = q_out_rows + 2 * kv_out_rows;
    const int64_t attn_cols = static_cast<int64_t>(n_embd);

    // local Q/K/V rows per tp_rank
    const int64_t q_local_rows = static_cast<int64_t>(n_embd) / tp_size; // = (n_head/world)*head_dim
    const int64_t kv_head_local = static_cast<int64_t>(n_kv_head) / tp_size;
    const int64_t kv_local_rows = kv_head_local * head_dim; // for K or V (each)

    // RowParallel (proj)
    const int64_t in_pp = static_cast<int64_t>(n_embd) / tp_size;
    // nn::MLP: c_fc/c_fc2（shard along row），c_proj（shard along col）
    const int64_t fc_out = ffn_hidden;
    const int64_t fc_pp = fc_out / tp_size;
    const int64_t in_fc_pp = ffn_hidden / tp_size;

    auto state_dict = llama3->StateDict();

    // ========== Read Sharded Params ==========
    // transformer.wte.weight : (vocab_size, n_embd) -> local tp_rank: rows of
    // [v_start : v_start+vpp)
    if (has_embedding) {
        auto &wte = state_dict[std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                           nn::TransformerFirstStage::kWTELayerName,
                                           nn::parallel::VocabParallelEmbedding::kParamWeightName)];
        ReadMatrixRowShardFloat(ifs, static_cast<float *>(wte->DataPtr()),
                                /*rows=*/vocab_size, /*cols=*/n_embd,
                                /*row_start=*/v_start, /*row_cnt=*/vpp);
    } else {
        size_t wte_bytes = static_cast<size_t>(vocab_size) * n_embd * sizeof(float);
        ifs.seekg(wte_bytes, std::ios::cur);
    }

    // transformer.h.{i}.ln_1.weight : Full version nn::RMSNorm
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers.at(static_cast<std::size_t>(i))) {
            const LayerIndex local_layer_index = local_indices.at(static_cast<std::size_t>(i));
            CHECK_GE(local_layer_index, 0);
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kLn1LayerName, nn::RMSNorm::kParamWeightName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
        } else {
            size_t ln_1_bytes = n_embd * sizeof(float);
            ifs.seekg(ln_1_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.attn.c_attn.weight : ColumnParallelLinear, but actually
    // applies on "rows" W-qkv should be [Q(=n_embd) | K(=n_kv_head*head_dim) |
    // V(=n_kv_head*head_dim)] × n_embd
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers.at(static_cast<std::size_t>(i))) {
            const LayerIndex local_layer_index = local_indices.at(static_cast<std::size_t>(i));
            CHECK_GE(local_layer_index, 0);
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName,
                std::to_string(local_layer_index), nn::TransformerLayer::kAttnLayerName,
                nn::CausalSelfAttention::kCAttnLayerName, nn::parallel::ColumnParallelLinear::kParamWeightName)];

            float *dst = static_cast<float *>(tensor->DataPtr());
            const std::streampos base_pos = ifs.tellg();

            // Q block -> [0 : q_local_rows)
            ifs.seekg(base_pos);
            ReadMatrixRowShardFloat(ifs,
                                    /*dst=*/dst + (0 * attn_cols),
                                    /*rows=*/attn_rows_all, /*cols=*/attn_cols,
                                    /*row_start=*/tp_rank * q_local_rows,
                                    /*row_cnt=*/q_local_rows);

            // K block -> [q_local_rows : q_local_rows + kv_local_rows)
            ifs.seekg(base_pos);
            ReadMatrixRowShardFloat(ifs,
                                    /*dst=*/dst + (q_local_rows * attn_cols),
                                    /*rows=*/attn_rows_all, /*cols=*/attn_cols,
                                    /*row_start=*/q_out_rows + tp_rank * kv_local_rows,
                                    /*row_cnt=*/kv_local_rows);

            // V block -> [q_local_rows + kv_local_rows : q_local_rows +
            // 2*kv_local_rows)
            ifs.seekg(base_pos);
            ReadMatrixRowShardFloat(ifs,
                                    /*dst=*/dst + ((q_local_rows + kv_local_rows) * attn_cols),
                                    /*rows=*/attn_rows_all, /*cols=*/attn_cols,
                                    /*row_start=*/q_out_rows + kv_out_rows + tp_rank * kv_local_rows,
                                    /*row_cnt=*/kv_local_rows);
        } else {
            size_t qkv_bytes = static_cast<size_t>(attn_rows_all) * attn_cols * sizeof(float);
            ifs.seekg(qkv_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.attn.c_proj.weight : RowParallelLinear, but actually
    // applies on "columns"
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers.at(static_cast<std::size_t>(i))) {
            const LayerIndex local_layer_index = local_indices.at(static_cast<std::size_t>(i));
            CHECK_GE(local_layer_index, 0);
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName,
                std::to_string(local_layer_index), nn::TransformerLayer::kAttnLayerName,
                nn::CausalSelfAttention::kCProjLayerName, nn::parallel::RowParallelLinear::kParamWeightName)];
            ReadMatrixColShardFloat(ifs, static_cast<float *>(tensor->DataPtr()),
                                    /*rows=*/n_embd, /*cols=*/n_embd,
                                    /*col_start=*/tp_rank * in_pp, /*col_cnt=*/in_pp);
        } else {
            size_t c_proj_bytes = static_cast<size_t>(n_embd) * n_embd * sizeof(float);
            ifs.seekg(c_proj_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.ln_2.weight : Full version RMSNorm
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers.at(static_cast<std::size_t>(i))) {
            const LayerIndex local_layer_index = local_indices.at(static_cast<std::size_t>(i));
            CHECK_GE(local_layer_index, 0);
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kLn2LayerName, nn::RMSNorm::kParamWeightName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
        } else {
            size_t ln_2_bytes = static_cast<size_t>(n_embd) * sizeof(float);
            ifs.seekg(ln_2_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.mlp.c_fc.weight : ColumnParallelLinear, but actually
    // applies on "rows"
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers.at(static_cast<std::size_t>(i))) {
            const LayerIndex local_layer_index = local_indices.at(static_cast<std::size_t>(i));
            CHECK_GE(local_layer_index, 0);
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kMlpLayerName, nn::MLP::kCFcLayerName,
                                                  nn::parallel::ColumnParallelLinear::kParamWeightName)];
            ReadMatrixRowShardFloat(ifs, static_cast<float *>(tensor->DataPtr()),
                                    /*rows=*/fc_out, /*cols=*/n_embd,
                                    /*row_start=*/tp_rank * fc_pp, /*row_cnt=*/fc_pp);
        } else {
            size_t fc_bytes = static_cast<size_t>(ffn_hidden) * n_embd * sizeof(float);
            ifs.seekg(fc_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.mlp.c_fc2.weight : ColumnParallelLinear, but actually
    // applies on "rows"
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers.at(static_cast<std::size_t>(i))) {
            const LayerIndex local_layer_index = local_indices.at(static_cast<std::size_t>(i));
            CHECK_GE(local_layer_index, 0);
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kMlpLayerName, nn::MLP::kCFc2LayerName,
                                                  nn::parallel::ColumnParallelLinear::kParamWeightName)];
            ReadMatrixRowShardFloat(ifs, static_cast<float *>(tensor->DataPtr()),
                                    /*rows=*/fc_out, /*cols=*/n_embd,
                                    /*row_start=*/tp_rank * fc_pp, /*row_cnt=*/fc_pp);
        } else {
            size_t fc2_bytes = static_cast<size_t>(ffn_hidden) * n_embd * sizeof(float);
            ifs.seekg(fc2_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.mlp.c_proj.weight : RowParallelLinear, but actually
    // applies on "columns"
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers.at(static_cast<std::size_t>(i))) {
            const LayerIndex local_layer_index = local_indices.at(static_cast<std::size_t>(i));
            CHECK_GE(local_layer_index, 0);
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kMlpLayerName, nn::MLP::kCProjLayerName,
                                                  nn::parallel::RowParallelLinear::kParamWeightName)];
            ReadMatrixColShardFloat(ifs, static_cast<float *>(tensor->DataPtr()),
                                    /*rows=*/n_embd, /*cols=*/fc_out,
                                    /*col_start=*/tp_rank * in_fc_pp,
                                    /*col_cnt=*/in_fc_pp);
        } else {
            size_t c_proj_bytes = static_cast<size_t>(n_embd) * ffn_hidden * sizeof(float);
            ifs.seekg(c_proj_bytes, std::ios::cur);
        }
    }

    // transformer.ln_f.weight : Full version nn::RMSNorm
    // lm_head.weight : (vocab_size, n_embd) -> ColumnParallelLinear, but actually
    // applies on "rows"
    CHECK_EQ(has_final_norm, has_lm_head) << "current combined output module requires final norm and LM head together";
    {
        if (has_final_norm && has_lm_head) {
            auto &ln_f
                = state_dict[std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                         nn::TransformerLastStage::kLnFLayerName, nn::RMSNorm::kParamWeightName)];
            auto &lm_head = state_dict[std::format("{}.{}", nn::TransformerLastStage::kLMHeadLayerName,
                                                   nn::parallel::ColumnParallelLinear::kParamWeightName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(ln_f->DataPtr()), n_embd);
            ReadMatrixRowShardFloat(ifs, static_cast<float *>(lm_head->DataPtr()),
                                    /*rows=*/vocab_size, /*cols=*/n_embd,
                                    /*row_start=*/v_start, /*row_cnt=*/vpp);
        } else {
            size_t ln_f_bytes = static_cast<size_t>(n_embd) * sizeof(float);
            size_t lm_head_bytes = static_cast<size_t>(vocab_size) * n_embd * sizeof(float);
            ifs.seekg(ln_f_bytes + lm_head_bytes, std::ios::cur);
        }
    }

    return llama3;
}
} // namespace

std::shared_ptr<nn::TransformerModel> LoadFromLLMC(const std::string &filepath) {
    return LoadFromLLMCImpl(filepath, std::nullopt);
}

std::shared_ptr<nn::TransformerModel>
LoadFromLLMC(const std::string &filepath, const infini_train::examples::PipelineLayoutRequest &layout_request) {
    return LoadFromLLMCImpl(filepath, layout_request);
}

} // namespace llama3
