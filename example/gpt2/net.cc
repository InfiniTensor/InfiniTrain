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
#include "infini_train/include/models/gpt2/gpt2.h"
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

    auto local_gpt2 = std::make_shared<GPT2>(nn::TransformerConfig{.block_size = block_size,
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
        auto &transformer_wte_weight = state_dict[std::format("{}.{}.{}", GPT2::kTransformerLayerName,
                                                              nn::TransformerFirstStageABI::kWTELayerName,
                                                              nn::parallel::VocabParallelEmbedding::kParamWeightName)];
        ReadMatrixRowShardFloat(ifs, static_cast<float *>(transformer_wte_weight->DataPtr()), model_vocab_size, n_embd,
                                v_start, vpp);
    } else if (pp_size > 1 && is_last_stage) {
        auto &lm_head_weight = state_dict[std::format("{}.{}", nn::TransformerLastStageABI::kLMHeadLayerName,
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
        auto &transformer_wpe_weight
            = state_dict[std::format("{}.{}.{}", GPT2::kTransformerLayerName,
                                     nn::TransformerFirstStageABI::kWPELayerName, nn::Embedding::kParamWeightName)];
        ReadMatrixAllFloat(ifs, static_cast<float *>(transformer_wpe_weight->DataPtr()), block_size, n_embd);
    } else {
        size_t wpe_bytes = block_size * n_embd * sizeof(float);
        ifs.seekg(wpe_bytes, std::ios::cur);
    }

    // transformer.h.{i}.ln_1.weight
    int local_layer_index = 0;
    for (int idx = 0; idx < n_layer; ++idx) {
        if (owned_layers[idx]) {
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, nn::TransformerChunkABI::kHLayerName,
                std::to_string(local_layer_index), Block::kLn1LayerName, nn::LayerNorm::kParamWeightName)];
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
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, nn::TransformerChunkABI::kHLayerName,
                std::to_string(local_layer_index), Block::kLn1LayerName, nn::LayerNorm::kParamBiasName)];
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
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, nn::TransformerChunkABI::kHLayerName,
                std::to_string(local_layer_index), Block::kAttnLayerName, CausalSelfAttention::kCAttnLayerName,
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
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, nn::TransformerChunkABI::kHLayerName,
                std::to_string(local_layer_index), Block::kAttnLayerName, CausalSelfAttention::kCAttnLayerName,
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
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, nn::TransformerChunkABI::kHLayerName,
                std::to_string(local_layer_index), Block::kAttnLayerName, CausalSelfAttention::kCProjLayerName,
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
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, nn::TransformerChunkABI::kHLayerName,
                std::to_string(local_layer_index), Block::kAttnLayerName, CausalSelfAttention::kCProjLayerName,
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
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, nn::TransformerChunkABI::kHLayerName,
                std::to_string(local_layer_index), Block::kLn2LayerName, nn::LayerNorm::kParamWeightName)];
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
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, nn::TransformerChunkABI::kHLayerName,
                std::to_string(local_layer_index), Block::kLn2LayerName, nn::LayerNorm::kParamBiasName)];
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
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, nn::TransformerChunkABI::kHLayerName,
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
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, nn::TransformerChunkABI::kHLayerName,
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
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, nn::TransformerChunkABI::kHLayerName,
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
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", GPT2::kTransformerLayerName, nn::TransformerChunkABI::kHLayerName,
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
        auto &transformer_ln_f_weight
            = state_dict[std::format("{}.{}.{}", GPT2::kTransformerLayerName,
                                     nn::TransformerLastStageABI::kLnFLayerName, nn::LayerNorm::kParamWeightName)];
        ReadVectorAllFloat(ifs, static_cast<float *>(transformer_ln_f_weight->DataPtr()), n_embd);
        // transformer.ln_f.bias
        auto &transformer_ln_f_bias
            = state_dict[std::format("{}.{}.{}", GPT2::kTransformerLayerName,
                                     nn::TransformerLastStageABI::kLnFLayerName, nn::LayerNorm::kParamBiasName)];
        ReadVectorAllFloat(ifs, static_cast<float *>(transformer_ln_f_bias->DataPtr()), n_embd);
    } else {
        size_t ln_f_w_bytes = n_embd * sizeof(float);
        size_t ln_f_b_bytes = n_embd * sizeof(float);
        ifs.seekg(ln_f_w_bytes + ln_f_b_bytes, std::ios::cur);
    }
    return local_gpt2;
}
