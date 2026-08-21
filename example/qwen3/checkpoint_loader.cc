#include "example/qwen3/checkpoint_loader.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "glog/logging.h"

#include "example/common/utils.h"
#include "example/qwen3/config.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"
#include "infini_train/include/nn/modules/transformer/mlp.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;
namespace nn = infini_train::nn;

namespace {
constexpr int32_t kQwen3Magic = 20240804;
constexpr int32_t kQwen3FP32Version = 4;
constexpr size_t kQwen3HeaderBytes = 256 * sizeof(int32_t);
} // namespace

namespace qwen3 {

std::shared_ptr<nn::TransformerModel> LoadFromLLMC(const std::string &filepath) {
    if (!std::filesystem::exists(filepath)) {
        LOG(FATAL) << "File not found: " << filepath;
    }

    std::error_code file_size_error;
    const auto actual_file_size = std::filesystem::file_size(filepath, file_size_error);
    CHECK(!file_size_error) << "Failed to get file size for " << filepath << ": " << file_size_error.message();
    CHECK_GE(actual_file_size, kQwen3HeaderBytes) << "Qwen3 LLMC file is shorter than its header: " << filepath;

    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open Qwen3 LLMC file: " << filepath;
    const auto header = ReadSeveralBytesFromIfstream(kQwen3HeaderBytes, &ifs);
    CHECK(ifs.good()) << "Failed to read Qwen3 LLMC header: " << filepath;

    const auto magic = BytesToType<uint32_t>(header, 0);
    CHECK_EQ(magic, kQwen3Magic);
    const auto version = BytesToType<uint32_t>(header, 4);
    CHECK_EQ(version, kQwen3FP32Version);

    const auto block_size = BytesToType<uint32_t>(header, 8);
    const auto vocab_size = BytesToType<uint32_t>(header, 12);
    const auto n_layer = BytesToType<uint32_t>(header, 16);
    const auto n_head = BytesToType<uint32_t>(header, 20);
    const auto n_kv_head = BytesToType<uint32_t>(header, 24);
    const auto n_embd = BytesToType<uint32_t>(header, 28);
    const auto intermediate_size = BytesToType<uint32_t>(header, 32);
    const auto multiple_of = BytesToType<uint32_t>(header, 36);
    const auto norm_eps = BytesToType<float>(header, 40);
    const auto rope_theta = BytesToType<float>(header, 44);
    const auto use_scaled_rope = BytesToType<int32_t>(header, 48);
    const auto max_gen_bs = BytesToType<int32_t>(header, 52);
    const auto version_major = BytesToType<int32_t>(header, 56);
    const auto version_minor = BytesToType<int32_t>(header, 60);

    CHECK_GT(n_layer, 0);
    CHECK_GT(n_head, 0);
    CHECK_GT(n_kv_head, 0);
    CHECK_GT(n_embd, 0);
    CHECK_GT(intermediate_size, 0);
    CHECK_EQ(n_embd % n_head, 0) << "n_embd must be divisible by n_head.";
    CHECK_EQ(n_head % n_kv_head, 0) << "n_head must be divisible by n_kv_head.";

    const uintmax_t header_head_dim = n_embd / n_head;
    const uintmax_t embedding_elements = static_cast<uintmax_t>(vocab_size) * n_embd;
    const uintmax_t norm_elements = static_cast<uintmax_t>(n_layer) * n_embd;
    const uintmax_t qk_norm_elements = 2 * static_cast<uintmax_t>(n_layer) * header_head_dim;
    const uintmax_t qkv_elements
        = static_cast<uintmax_t>(n_layer) * (n_embd + 2 * static_cast<uintmax_t>(n_kv_head) * header_head_dim) * n_embd;
    const uintmax_t attention_output_elements = static_cast<uintmax_t>(n_layer) * n_embd * n_embd;
    const uintmax_t mlp_elements = 3 * static_cast<uintmax_t>(n_layer) * intermediate_size * n_embd;
    const uintmax_t final_norm_elements = n_embd;
    const uintmax_t expected_file_size
        = kQwen3HeaderBytes
        + sizeof(float)
              * (2 * embedding_elements + 2 * norm_elements + qk_norm_elements + qkv_elements
                 + attention_output_elements + mlp_elements + final_norm_elements);
    CHECK_EQ(actual_file_size, expected_file_size) << "Qwen3 LLMC size mismatch for " << filepath << ": expected "
                                                   << expected_file_size << " bytes, got " << actual_file_size;

    nn::TransformerConfig qwen3_config = qwen3::Qwen3Config();
    qwen3_config.block_size = block_size;
    qwen3_config.vocab_size = vocab_size;
    qwen3_config.n_layer = n_layer;
    qwen3_config.n_head = n_head;
    qwen3_config.n_kv_head = n_kv_head;
    qwen3_config.n_embd = n_embd;
    qwen3_config.ffn_expansion_ratio
        = 3.0f * static_cast<float>(intermediate_size) / (2.0f * static_cast<float>(n_embd));
    qwen3_config.multiple_of = multiple_of;
    qwen3_config.rope_theta = rope_theta;
    qwen3_config.use_scaled_rope = static_cast<bool>(use_scaled_rope);
    qwen3_config.norm_eps = norm_eps;
    qwen3_config.max_gen_batch_size = max_gen_bs;
    qwen3_config.use_qk_norm = true;
    qwen3_config.qk_norm_eps = norm_eps;
    auto qwen3 = std::make_shared<nn::TransformerModel>(qwen3_config);

    // ========== pp_size: num_stages; vpp_size: num_chunks_per_stage ==========
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

    const int tp_size = nn::parallel::global::GetTensorParallelSize();
    const int tp_rank = nn::parallel::tp_rank;

    CHECK_EQ(n_embd % tp_size, 0) << "n_embd must be divisible by TP world size.";
    CHECK_EQ(n_head % tp_size, 0) << "n_head must be divisible by TP world size.";
    CHECK_EQ(n_kv_head % tp_size, 0) << "n_kv_head must be divisible by TP world size.";
    CHECK_EQ(vocab_size % tp_size, 0) << "vocab_size must be divisible by TP world size.";
    CHECK_EQ(intermediate_size % tp_size, 0) << "intermediate_size must be divisible by TP world size.";

    if (tp_rank == 0) {
        LOG(INFO) << "Model Config:";
        LOG(INFO) << "  block_size         = " << block_size;
        LOG(INFO) << "  vocab_size         = " << vocab_size;
        LOG(INFO) << "  n_layer            = " << n_layer;
        LOG(INFO) << "  n_head             = " << n_head;
        LOG(INFO) << "  n_kv_head          = " << n_kv_head;
        LOG(INFO) << "  n_embd             = " << n_embd;
        LOG(INFO) << "  intermediate_size  = " << intermediate_size;
        LOG(INFO) << "  multiple_of        = " << multiple_of;
        LOG(INFO) << "  norm_eps           = " << norm_eps;
        LOG(INFO) << "  rope_theta         = " << rope_theta;
        LOG(INFO) << "  use_scaled_rope    = " << use_scaled_rope;
        LOG(INFO) << "  max_gen_bs         = " << max_gen_bs;
        LOG(INFO) << "  version_major      = " << version_major;
        LOG(INFO) << "  version_minor      = " << version_minor;

        LOG(INFO) << "Pipeline Parallel Chunks:";
        for (size_t i = 0; i < layer_ranges_per_chunk.size(); ++i) {
            LOG(INFO) << "  Chunk " << i << ": layers " << layer_ranges_per_chunk[i].first << " to "
                      << layer_ranges_per_chunk[i].second;
        }
    }

    const int64_t head_dim = static_cast<int64_t>(n_embd) / static_cast<int64_t>(n_head);

    const int64_t ffn_hidden = static_cast<int64_t>(intermediate_size);

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
    // nn::MLP: c_fc/c_fc2(shard along row), c_proj(shard along col)
    const int64_t fc_out = ffn_hidden;
    const int64_t fc_pp = fc_out / tp_size;
    const int64_t in_fc_pp = ffn_hidden / tp_size;

    auto state_dict = qwen3->StateDict();

    // ========== Read Sharded Params ==========
    // transformer.wte.weight : (vocab_size, n_embd) -> local tp_rank: rows of [v_start : v_start+vpp)
    if (is_first_stage) {
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
    int local_layer_index = 0;
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers[i]) {
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kLn1LayerName, nn::RMSNorm::kParamWeightName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
            ++local_layer_index;
        } else {
            size_t ln_1_bytes = n_embd * sizeof(float);
            ifs.seekg(ln_1_bytes, std::ios::cur);
        }
    }

    local_layer_index = 0;
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers[i]) {
            auto &q_norm_tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName,
                std::to_string(local_layer_index), nn::TransformerLayer::kAttnLayerName,
                nn::CausalSelfAttention::kQNormLayerName, nn::RMSNorm::kParamWeightName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(q_norm_tensor->DataPtr()), head_dim);

            auto &k_norm_tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName,
                std::to_string(local_layer_index), nn::TransformerLayer::kAttnLayerName,
                nn::CausalSelfAttention::kKNormLayerName, nn::RMSNorm::kParamWeightName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(k_norm_tensor->DataPtr()), head_dim);
            ++local_layer_index;
        } else {
            size_t qk_norm_bytes = 2 * head_dim * sizeof(float);
            ifs.seekg(qk_norm_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.attn.c_attn.weight : ColumnParallelLinear, but actually applies on "rows"
    // W-qkv should be [Q(=n_embd) | K(=n_kv_head*head_dim) | V(=n_kv_head*head_dim)] x n_embd
    local_layer_index = 0;
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers[i]) {
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
                                    /*row_start=*/tp_rank * q_local_rows, /*row_cnt=*/q_local_rows);

            // K block -> [q_local_rows : q_local_rows + kv_local_rows)
            ifs.seekg(base_pos);
            ReadMatrixRowShardFloat(ifs,
                                    /*dst=*/dst + (q_local_rows * attn_cols),
                                    /*rows=*/attn_rows_all, /*cols=*/attn_cols,
                                    /*row_start=*/q_out_rows + tp_rank * kv_local_rows, /*row_cnt=*/kv_local_rows);

            // V block -> [q_local_rows + kv_local_rows : q_local_rows + 2*kv_local_rows)
            ifs.seekg(base_pos);
            ReadMatrixRowShardFloat(ifs,
                                    /*dst=*/dst + ((q_local_rows + kv_local_rows) * attn_cols),
                                    /*rows=*/attn_rows_all, /*cols=*/attn_cols,
                                    /*row_start=*/q_out_rows + kv_out_rows + tp_rank * kv_local_rows,
                                    /*row_cnt=*/kv_local_rows);
            ++local_layer_index;
        } else {
            size_t qkv_bytes = static_cast<size_t>(attn_rows_all) * attn_cols * sizeof(float);
            ifs.seekg(qkv_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.attn.c_proj.weight : RowParallelLinear, but actually applies on "columns"
    local_layer_index = 0;
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers[i]) {
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName,
                std::to_string(local_layer_index), nn::TransformerLayer::kAttnLayerName,
                nn::CausalSelfAttention::kCProjLayerName, nn::parallel::RowParallelLinear::kParamWeightName)];
            ReadMatrixColShardFloat(ifs, static_cast<float *>(tensor->DataPtr()),
                                    /*rows=*/n_embd, /*cols=*/n_embd,
                                    /*col_start=*/tp_rank * in_pp, /*col_cnt=*/in_pp);
            ++local_layer_index;
        } else {
            size_t c_proj_bytes = static_cast<size_t>(n_embd) * n_embd * sizeof(float);
            ifs.seekg(c_proj_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.ln_2.weight : Full version RMSNorm
    local_layer_index = 0;
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers[i]) {
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kLn2LayerName, nn::RMSNorm::kParamWeightName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
            ++local_layer_index;
        } else {
            size_t ln_2_bytes = static_cast<size_t>(n_embd) * sizeof(float);
            ifs.seekg(ln_2_bytes, std::ios::cur);
        }
    }

    // gate_proj is loaded into c_fc2 because MLP applies SiLU to its second projection.
    local_layer_index = 0;
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers[i]) {
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kMlpLayerName, nn::MLP::kCFc2LayerName,
                                                  nn::parallel::ColumnParallelLinear::kParamWeightName)];
            ReadMatrixRowShardFloat(ifs, static_cast<float *>(tensor->DataPtr()),
                                    /*rows=*/fc_out, /*cols=*/n_embd,
                                    /*row_start=*/tp_rank * fc_pp, /*row_cnt=*/fc_pp);
            ++local_layer_index;
        } else {
            size_t fc_bytes = static_cast<size_t>(ffn_hidden) * n_embd * sizeof(float);
            ifs.seekg(fc_bytes, std::ios::cur);
        }
    }

    // up_proj is loaded into c_fc, producing SiLU(gate_proj(x)) * up_proj(x).
    local_layer_index = 0;
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers[i]) {
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kMlpLayerName, nn::MLP::kCFcLayerName,
                                                  nn::parallel::ColumnParallelLinear::kParamWeightName)];
            ReadMatrixRowShardFloat(ifs, static_cast<float *>(tensor->DataPtr()),
                                    /*rows=*/fc_out, /*cols=*/n_embd,
                                    /*row_start=*/tp_rank * fc_pp, /*row_cnt=*/fc_pp);
            ++local_layer_index;
        } else {
            size_t fc2_bytes = static_cast<size_t>(ffn_hidden) * n_embd * sizeof(float);
            ifs.seekg(fc2_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.mlp.c_proj.weight : RowParallelLinear, but actually applies on "columns"
    local_layer_index = 0;
    for (int i = 0; i < static_cast<int>(n_layer); ++i) {
        if (owned_layers[i]) {
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kMlpLayerName, nn::MLP::kCProjLayerName,
                                                  nn::parallel::RowParallelLinear::kParamWeightName)];
            ReadMatrixColShardFloat(ifs, static_cast<float *>(tensor->DataPtr()),
                                    /*rows=*/n_embd, /*cols=*/fc_out,
                                    /*col_start=*/tp_rank * in_fc_pp, /*col_cnt=*/in_fc_pp);
            ++local_layer_index;
        } else {
            size_t c_proj_bytes = static_cast<size_t>(n_embd) * ffn_hidden * sizeof(float);
            ifs.seekg(c_proj_bytes, std::ios::cur);
        }
    }

    // transformer.ln_f.weight : Full version nn::RMSNorm
    // lm_head.weight : (vocab_size, n_embd) -> ColumnParallelLinear, but actually applies on "rows"
    {
        if (is_last_stage) {
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

    return qwen3;
}
} // namespace qwen3
