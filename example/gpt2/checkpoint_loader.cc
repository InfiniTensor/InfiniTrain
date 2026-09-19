#include "example/gpt2/checkpoint_loader.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "example/common/utils.h"
#include "example/gpt2/config.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"
#include "infini_train/include/nn/modules/transformer/mlp.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
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

namespace gpt2 {
namespace {
using LayerIndex = nn::parallel::LayerIndex;
using PipelineLayout = nn::parallel::PipelineLayout;

void ValidateCheckpointConfig(const nn::TransformerConfig &expected, const nn::TransformerConfig &actual) {
    const auto check = [](std::string_view field, auto expected_value, auto actual_value) {
        if (expected_value != actual_value) {
            throw std::invalid_argument("GPT-2 checkpoint config mismatch: " + std::string(field)
                                        + ", expected=" + std::to_string(expected_value)
                                        + ", checkpoint=" + std::to_string(actual_value));
        }
    };
    check("block_size", expected.block_size, actual.block_size);
    check("original_vocab_size", expected.original_vocab_size, actual.original_vocab_size);
    check("vocab_size", expected.vocab_size, actual.vocab_size);
    check("n_layer", expected.n_layer, actual.n_layer);
    check("n_head", expected.n_head, actual.n_head);
    check("n_kv_head", expected.n_kv_head, actual.n_kv_head);
    check("n_embd", expected.n_embd, actual.n_embd);

    // LLMC GPT-2 stores no architecture metadata beyond the header dimensions.
    const auto canonical = gpt2::GPT2Config();
    if (expected.position_embedding_type != canonical.position_embedding_type
        || expected.activation_type != canonical.activation_type || expected.norm_type != canonical.norm_type
        || expected.ffn_type != canonical.ffn_type || expected.add_bias_linear != canonical.add_bias_linear
        || expected.add_bias_lm_head != canonical.add_bias_lm_head
        || expected.ffn_expansion_ratio != canonical.ffn_expansion_ratio
        || expected.ffn_dim_multiplier != canonical.ffn_dim_multiplier || expected.multiple_of != canonical.multiple_of
        || expected.moe_config.has_value()) {
        throw std::invalid_argument("LLMC GPT-2 checkpoint requires the canonical GPT-2 architecture");
    }
}
} // namespace

std::shared_ptr<nn::TransformerModel> LoadFromLLMC(const std::string &filepath,
                                                       const nn::TransformerConfig &expected_config,
                                                       std::shared_ptr<const PipelineLayout> pipeline_layout) {
    if (!std::filesystem::exists(filepath)) {
        LOG(FATAL) << "File not found: " << filepath;
    }

    std::ifstream ifs(filepath, std::ios::binary);
    const auto header = ReadSeveralBytesFromIfstream(256 * sizeof(int32_t), &ifs);
    if (!ifs) {
        throw std::invalid_argument("checkpoint header must contain 1024 bytes: " + filepath);
    }

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
    // ========== pp_size：num_stages; vpp_size: num_chunks_per_stage ==========
    int pp_size = nn::parallel::global::GetPipelineParallelSize();
    int vpp_size = nn::parallel::global::GetVirtualPipelineParallelSize();
    auto pp_rank = nn::parallel::pp_rank;
    nn::TransformerConfig gpt2_config = gpt2::GPT2Config();
    gpt2_config.block_size = block_size;
    gpt2_config.vocab_size = model_vocab_size;
    gpt2_config.original_vocab_size = vocab_size;
    gpt2_config.n_layer = n_layer;
    gpt2_config.n_head = n_head;
    gpt2_config.n_kv_head = n_head;
    gpt2_config.n_embd = n_embd;
    // Cross-rank tying is not implemented. Keep every PP>1 layout on the
    // existing split-weight semantics, even when both endpoints share a rank.
    gpt2_config.tie_weights = gpt2_config.tie_weights && pp_size == 1;
    ValidateCheckpointConfig(expected_config, gpt2_config);
    if (block_size == 0 || vocab_size == 0 || n_layer == 0 || n_head == 0 || n_embd == 0
        || model_vocab_size < vocab_size) {
        throw std::invalid_argument("GPT-2 checkpoint dimensions must be positive and vocab must cover original vocab");
    }
    if (tp_size <= 0) {
        throw std::invalid_argument("GPT-2 requires positive TP size");
    }
    CHECK_EQ(n_embd % tp_size, 0) << "n_embd must be divisible by TP world size.";
    CHECK_EQ(n_embd % n_head, 0) << "n_embd must be divisible by n_head.";
    CHECK_EQ(n_head % tp_size, 0) << "n_head must be divisible by TP world size.";

    if (pipeline_layout) {
        if (pipeline_layout->GetNumLayers() != static_cast<LayerIndex>(n_layer)) {
            throw std::invalid_argument("pipeline layout layer count does not match checkpoint n_layer");
        }
        if (pipeline_layout->GetNumStages() != pp_size) {
            throw std::invalid_argument("pipeline layout stage count does not match PP world size");
        }
        if (pipeline_layout->GetMaxLocalChunks() != vpp_size) {
            throw std::invalid_argument("pipeline layout max local chunk count does not match vPP");
        }
    }
    // Use the caller's config after validating dimensions and the GPT-2 architecture.
    gpt2_config = expected_config;
    gpt2_config.tie_weights = gpt2_config.tie_weights && pp_size == 1;
    gpt2::SanitizeGPT2Config(gpt2_config);
    auto local_gpt2 = pipeline_layout ? std::make_shared<nn::TransformerModel>(gpt2_config, pipeline_layout, pp_rank)
                                      : std::make_shared<nn::TransformerModel>(gpt2_config);

    LOG(INFO) << "magic: " << magic << " version: " << version << " block_size: " << block_size
              << " vocab_size: " << vocab_size << " n_layer: " << n_layer << " n_head: " << n_head
              << " n_embd: " << n_embd << " padded_vocab_size: " << padded_vocab_size;

    bool has_embedding = false;
    bool has_final_norm = false;
    bool has_lm_head = false;

    std::vector<LayerIndex> local_indices(n_layer, -1);

    LayerIndex local = 0;
    if (pipeline_layout) {
        const auto &stage = pipeline_layout->GetStage(pp_rank);
        has_embedding = stage.has_embedding;
        has_final_norm = stage.has_final_norm;
        has_lm_head = stage.has_lm_head;

        for (const auto &chunk : stage.chunks) {
            for (LayerIndex layer = chunk.layer_range.begin; layer < chunk.layer_range.end; ++layer) {
                local_indices.at(layer) = local++;
            }
        }
    } else {
        const auto legacy = nn::parallel::PipelineParallel::GetStageInfo(n_layer, pp_size, pp_rank, vpp_size);

        has_embedding = legacy.is_first_stage;
        has_final_norm = legacy.is_last_stage;
        has_lm_head = legacy.is_last_stage;

        for (const auto &[begin, end] : legacy.layer_ranges_per_chunk) {
            for (int layer = begin; layer < end; ++layer) { local_indices.at(layer) = local++; }
        }
    }

    auto tp_rank = nn::parallel::tp_rank;
    // calculate xx_size_per_partition
    const int64_t vpp = model_vocab_size / tp_size;
    const int64_t v_start = static_cast<int64_t>(tp_rank) * vpp;

    const int64_t qkv_out = 3 * n_embd;

    const int64_t fc_out = 4 * n_embd;
    const int64_t fc_pp = fc_out / tp_size;
    const int64_t fc_start = static_cast<int64_t>(tp_rank) * fc_pp;

    const int64_t in_pp = n_embd / tp_size;        // for c_proj (row-parallel, shard on input)
    const int64_t in4_pp = (4 * n_embd) / tp_size; // for mlp.c_proj (input shard)

    auto state_dict = local_gpt2->StateDict();

    // transformer.wte.weight (also transformer.lm_head.weight)
    // full: (model_vocab_size, n_embd)
    // local: (vocab_size_per_partition, n_embd)
    const auto wte_offset = ifs.tellg();
    if (has_embedding) {
        auto &transformer_wte_weight = state_dict[std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                              nn::TransformerFirstStage::kWTELayerName,
                                                              nn::parallel::VocabParallelEmbedding::kParamWeightName)];
        ReadMatrixRowShardFloat(ifs, static_cast<float *>(transformer_wte_weight->DataPtr()), model_vocab_size, n_embd,
                                v_start, vpp);
    }
    if (has_lm_head) {
        ifs.seekg(wte_offset);
        auto &lm_head_weight = state_dict[std::format("{}.{}", nn::TransformerLastStage::kLMHeadLayerName,
                                                      nn::parallel::ColumnParallelLinear::kParamWeightName)];
        ReadMatrixRowShardFloat(ifs, static_cast<float *>(lm_head_weight->DataPtr()), model_vocab_size, n_embd, v_start,
                                vpp);
    }
    ifs.seekg(wte_offset
              + static_cast<std::streamoff>(static_cast<uint64_t>(model_vocab_size) * n_embd * sizeof(float)));

    if (tp_size == 1) {
        // Skip padded vocab part when TP is not enabled
        ifs.ignore((padded_vocab_size - model_vocab_size) * n_embd * sizeof(float));
    }

    if (has_embedding) {
        // transformer.wpe.weight
        auto &transformer_wpe_weight
            = state_dict[std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                     nn::TransformerFirstStage::kWPELayerName, nn::Embedding::kParamWeightName)];
        ReadMatrixAllFloat(ifs, static_cast<float *>(transformer_wpe_weight->DataPtr()), block_size, n_embd);
    } else {
        size_t wpe_bytes = block_size * n_embd * sizeof(float);
        ifs.seekg(wpe_bytes, std::ios::cur);
    }

    // transformer.h.{i}.ln_1.weight
    for (int idx = 0; idx < n_layer; ++idx) {
        if (local_indices.at(idx) >= 0) {
            const LayerIndex local_layer_index = local_indices.at(idx);
            auto &tensor
                = state_dict[std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                         nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                         nn::TransformerLayer::kLn1LayerName, nn::LayerNorm::kParamWeightName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
        } else {
            size_t ln_1_w_bytes = n_embd * sizeof(float);
            ifs.seekg(ln_1_w_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.ln_1.bias
    for (int idx = 0; idx < n_layer; ++idx) {
        if (local_indices.at(idx) >= 0) {
            const LayerIndex local_layer_index = local_indices.at(idx);
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kLn1LayerName, nn::LayerNorm::kParamBiasName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
        } else {
            size_t ln_1_b_bytes = n_embd * sizeof(float);
            ifs.seekg(ln_1_b_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.attn.c_attn.weight (ColumnParallelLinear, but actually applies on "rows")
    for (int idx = 0; idx < n_layer; ++idx) {
        if (local_indices.at(idx) >= 0) {
            const LayerIndex local_layer_index = local_indices.at(idx);
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName,
                std::to_string(local_layer_index), nn::TransformerLayer::kAttnLayerName,
                nn::CausalSelfAttention::kCAttnLayerName, nn::parallel::ColumnParallelLinear::kParamWeightName)];
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

        } else {
            size_t c_attn_w_bytes = qkv_out * n_embd * sizeof(float);
            ifs.seekg(c_attn_w_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.attn.c_attn.bias (ColumnParallelLinear)
    for (int idx = 0; idx < n_layer; ++idx) {
        if (local_indices.at(idx) >= 0) {
            const LayerIndex local_layer_index = local_indices.at(idx);
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName,
                std::to_string(local_layer_index), nn::TransformerLayer::kAttnLayerName,
                nn::CausalSelfAttention::kCAttnLayerName, nn::parallel::ColumnParallelLinear::kParamBiasName)];
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

        } else {
            size_t c_attn_b_bytes = qkv_out * sizeof(float);
            ifs.seekg(c_attn_b_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.attn.c_proj.weight (RowParallelLinear, but actually applies on "columns")
    for (int idx = 0; idx < n_layer; ++idx) {
        if (local_indices.at(idx) >= 0) {
            const LayerIndex local_layer_index = local_indices.at(idx);
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName,
                std::to_string(local_layer_index), nn::TransformerLayer::kAttnLayerName,
                nn::CausalSelfAttention::kCProjLayerName, nn::parallel::RowParallelLinear::kParamWeightName)];
            ReadMatrixColShardFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd, n_embd, tp_rank * in_pp,
                                    in_pp);
        } else {
            size_t c_proj_w_bytes = n_embd * n_embd * sizeof(float);
            ifs.seekg(c_proj_w_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.attn.c_proj.bias (RowParallelLinear, no shard on bias)
    for (int idx = 0; idx < n_layer; ++idx) {
        if (local_indices.at(idx) >= 0) {
            const LayerIndex local_layer_index = local_indices.at(idx);
            auto &tensor = state_dict[std::format(
                "{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName,
                std::to_string(local_layer_index), nn::TransformerLayer::kAttnLayerName,
                nn::CausalSelfAttention::kCProjLayerName, nn::parallel::RowParallelLinear::kParamBiasName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
        } else {
            size_t c_proj_b_bytes = n_embd * sizeof(float);
            ifs.seekg(c_proj_b_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.ln_2.weight
    for (int idx = 0; idx < n_layer; ++idx) {
        if (local_indices.at(idx) >= 0) {
            const LayerIndex local_layer_index = local_indices.at(idx);
            auto &tensor
                = state_dict[std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                         nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                         nn::TransformerLayer::kLn2LayerName, nn::LayerNorm::kParamWeightName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
        } else {
            size_t ln_2_w_bytes = n_embd * sizeof(float);
            ifs.seekg(ln_2_w_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.ln_2.bias
    for (int idx = 0; idx < n_layer; ++idx) {
        if (local_indices.at(idx) >= 0) {
            const LayerIndex local_layer_index = local_indices.at(idx);
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kLn2LayerName, nn::LayerNorm::kParamBiasName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
        } else {
            size_t ln_2_b_bytes = n_embd * sizeof(float);
            ifs.seekg(ln_2_b_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.mlp.c_fc.weight (ColumnParallelLinear, but actually applies on "rows")
    for (int idx = 0; idx < n_layer; ++idx) {
        if (local_indices.at(idx) >= 0) {
            const LayerIndex local_layer_index = local_indices.at(idx);
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kMlpLayerName, nn::MLP::kCFcLayerName,
                                                  nn::parallel::ColumnParallelLinear::kParamWeightName)];
            ReadMatrixRowShardFloat(ifs, static_cast<float *>(tensor->DataPtr()), fc_out, n_embd, fc_start, fc_pp);
        } else {
            size_t c_fc_w_bytes = fc_out * n_embd * sizeof(float);
            ifs.seekg(c_fc_w_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.mlp.c_fc.bias (ColumnParallelLinear)
    for (int idx = 0; idx < n_layer; ++idx) {
        if (local_indices.at(idx) >= 0) {
            const LayerIndex local_layer_index = local_indices.at(idx);
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kMlpLayerName, nn::MLP::kCFcLayerName,
                                                  nn::parallel::ColumnParallelLinear::kParamBiasName)];
            ReadVectorShardFloat(ifs, static_cast<float *>(tensor->DataPtr()), fc_out, fc_start, fc_pp);
        } else {
            size_t c_fc_b_bytes = fc_out * sizeof(float);
            ifs.seekg(c_fc_b_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.mlp.c_proj.weight (RowParallelLinear, but actually applies on "columns")
    for (int idx = 0; idx < n_layer; ++idx) {
        if (local_indices.at(idx) >= 0) {
            const LayerIndex local_layer_index = local_indices.at(idx);
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kMlpLayerName, nn::MLP::kCProjLayerName,
                                                  nn::parallel::RowParallelLinear::kParamWeightName)];
            ReadMatrixColShardFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd, fc_out, tp_rank * in4_pp,
                                    in4_pp);
        } else {
            size_t c_proj_w_bytes = fc_out * n_embd * sizeof(float);
            ifs.seekg(c_proj_w_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.mlp.c_proj.bias (RowParallelLinear, no shard on bias)
    for (int idx = 0; idx < n_layer; ++idx) {
        if (local_indices.at(idx) >= 0) {
            const LayerIndex local_layer_index = local_indices.at(idx);
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kMlpLayerName, nn::MLP::kCProjLayerName,
                                                  nn::parallel::RowParallelLinear::kParamBiasName)];
            ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
        } else {
            size_t c_proj_b_bytes = n_embd * sizeof(float);
            ifs.seekg(c_proj_b_bytes, std::ios::cur);
        }
    }

    if (has_final_norm) {
        // transformer.ln_f.weight
        auto &transformer_ln_f_weight
            = state_dict[std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                     nn::TransformerLastStage::kLnFLayerName, nn::LayerNorm::kParamWeightName)];
        ReadVectorAllFloat(ifs, static_cast<float *>(transformer_ln_f_weight->DataPtr()), n_embd);
        // transformer.ln_f.bias
        auto &transformer_ln_f_bias
            = state_dict[std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                     nn::TransformerLastStage::kLnFLayerName, nn::LayerNorm::kParamBiasName)];
        ReadVectorAllFloat(ifs, static_cast<float *>(transformer_ln_f_bias->DataPtr()), n_embd);
    } else {
        size_t ln_f_w_bytes = n_embd * sizeof(float);
        size_t ln_f_b_bytes = n_embd * sizeof(float);
        ifs.seekg(ln_f_w_bytes + ln_f_b_bytes, std::ios::cur);
    }

    return local_gpt2;
}

std::shared_ptr<nn::TransformerModel> LoadFromLLMC(const std::string &filepath) {
    auto config = gpt2::GPT2Config();
    if (nn::parallel::global::GetTensorParallelSize() == 1) {
        config.vocab_size = config.original_vocab_size;
    }
    return LoadFromLLMC(filepath, config, nullptr);
}

} // namespace gpt2
