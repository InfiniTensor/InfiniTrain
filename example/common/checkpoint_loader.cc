#include "example/common/checkpoint_loader.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "example/common/utils.h"
#include "example/gpt2/config.h"
#include "example/llama3/config.h"
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
constexpr int32_t kGPT2Magic = 20240326;
constexpr int32_t kGPT2FP32Version = 3;
constexpr int32_t kGPT2BF16Version = 5;

constexpr int32_t kLLaMA3Magic = 20240803;
constexpr int32_t kLLaMA3FP32Version = 3;

std::tuple<int32_t, infini_train::DataType> DetermineAndCheckVersion(const std::vector<uint8_t> &header,
                                                                     size_t offset) {
    const auto version = BytesToType<uint32_t>(header, offset);
    switch (version) {
    case kGPT2FP32Version:
        return {version, infini_train::DataType::kBFLOAT16};
    case kGPT2BF16Version:
        return {version, infini_train::DataType::kFLOAT32};
    default:
        LOG(FATAL) << "Unsupported version: " << version << " at " << __FILE__ << ":" << __LINE__;
        return {}; // Unreachable, but keeps compiler happy
    }
}
} // namespace

namespace infini_train {
namespace gpt2 {

std::shared_ptr<nn::TransformerModel> LoadFromLLMC(const std::string &filepath) {
    if (!std::filesystem::exists(filepath)) {
        LOG(FATAL) << "File not found: " << filepath;
    }

    std::ifstream ifs(filepath, std::ios::binary);
    const auto header = ReadSeveralBytesFromIfstream(256 * sizeof(int32_t), &ifs);

    const auto magic = BytesToType<uint32_t>(header, 0);
    CHECK_EQ(magic, kGPT2Magic);
    auto [version, dtype] = DetermineAndCheckVersion(header, 4);
    CHECK_EQ(version, kGPT2FP32Version);

    auto tp_size = nn::parallel::global::GetTensorParallelSize();

    const auto block_size = BytesToType<uint32_t>(header, 8);
    const auto vocab_size = BytesToType<uint32_t>(header, 12);
    const auto n_layer = BytesToType<uint32_t>(header, 16);
    const auto n_head = BytesToType<uint32_t>(header, 20);
    const auto n_embd = BytesToType<uint32_t>(header, 24);
    const auto padded_vocab_size = BytesToType<uint32_t>(header, 28);
    // NOTE(zbl): vocab_size needs to be padded to multiple of TP size
    const auto model_vocab_size = tp_size > 1 ? padded_vocab_size : vocab_size;

    nn::TransformerConfig gpt2_config = infini_train::gpt2::GPT2Config();
    gpt2_config.block_size = block_size;
    gpt2_config.vocab_size = model_vocab_size;
    gpt2_config.original_vocab_size = vocab_size;
    gpt2_config.n_layer = n_layer;
    gpt2_config.n_head = n_head;
    gpt2_config.n_embd = n_embd;
    auto local_gpt2 = std::make_shared<nn::TransformerModel>(gpt2_config);

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
        auto &transformer_wte_weight = state_dict[std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                              nn::TransformerFirstStage::kWTELayerName,
                                                              nn::parallel::VocabParallelEmbedding::kParamWeightName)];
        ReadMatrixRowShardFloat(ifs, static_cast<float *>(transformer_wte_weight->DataPtr()), model_vocab_size, n_embd,
                                v_start, vpp);
    } else if (pp_size > 1 && is_last_stage) {
        auto &lm_head_weight = state_dict[std::format("{}.{}", nn::TransformerLastStage::kLMHeadLayerName,
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
            = state_dict[std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                     nn::TransformerFirstStage::kWPELayerName, nn::Embedding::kParamWeightName)];
        ReadMatrixAllFloat(ifs, static_cast<float *>(transformer_wpe_weight->DataPtr()), block_size, n_embd);
    } else {
        size_t wpe_bytes = block_size * n_embd * sizeof(float);
        ifs.seekg(wpe_bytes, std::ios::cur);
    }

    // transformer.h.{i}.ln_1.weight
    int local_layer_index = 0;
    for (int idx = 0; idx < n_layer; ++idx) {
        if (owned_layers[idx]) {
            auto &tensor
                = state_dict[std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                         nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                         nn::TransformerLayer::kLn1LayerName, nn::LayerNorm::kParamWeightName)];
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
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kLn1LayerName, nn::LayerNorm::kParamBiasName)];
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
                "{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName,
                std::to_string(local_layer_index), nn::TransformerLayer::kAttnLayerName,
                nn::CausalSelfAttention::kCProjLayerName, nn::parallel::RowParallelLinear::kParamWeightName)];
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
                "{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName,
                std::to_string(local_layer_index), nn::TransformerLayer::kAttnLayerName,
                nn::CausalSelfAttention::kCProjLayerName, nn::parallel::RowParallelLinear::kParamBiasName)];
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
            auto &tensor
                = state_dict[std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                         nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                         nn::TransformerLayer::kLn2LayerName, nn::LayerNorm::kParamWeightName)];
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
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kLn2LayerName, nn::LayerNorm::kParamBiasName)];
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
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kMlpLayerName, nn::MLP::kCFcLayerName,
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
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kMlpLayerName, nn::MLP::kCFcLayerName,
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
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kMlpLayerName, nn::MLP::kCProjLayerName,
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
            auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                  nn::TransformerChunk::kHLayerName, std::to_string(local_layer_index),
                                                  nn::TransformerLayer::kMlpLayerName, nn::MLP::kCProjLayerName,
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

void SaveAsLLMC(const std::shared_ptr<nn::TransformerModel> &model, const std::string &filepath) {
    CHECK_EQ(nn::parallel::global::GetTensorParallelSize(), 1) << "SaveAsLLMC currently supports TP=1 only.";
    CHECK_EQ(nn::parallel::global::GetPipelineParallelSize(), 1) << "SaveAsLLMC currently supports PP=1 only.";

    std::ofstream ofs(filepath, std::ios::binary);
    CHECK(ofs.is_open()) << "Failed to open model file for write: " << filepath;

    auto config = model->Config();
    std::vector<int32_t> header(256, 0);
    header[0] = kGPT2Magic;
    header[1] = kGPT2FP32Version;
    header[2] = static_cast<int32_t>(config.block_size);
    header[3] = static_cast<int32_t>(config.original_vocab_size);
    header[4] = static_cast<int32_t>(config.n_layer);
    header[5] = static_cast<int32_t>(config.n_head);
    header[6] = static_cast<int32_t>(config.n_embd);
    header[7] = static_cast<int32_t>(config.vocab_size);
    ofs.write(reinterpret_cast<const char *>(header.data()),
              static_cast<std::streamsize>(header.size() * sizeof(int32_t)));

    const auto state_dict = model->StateDict();
    auto get_tensor = [&](const std::string &name) -> std::shared_ptr<Tensor> {
        CHECK(state_dict.contains(name)) << "Missing tensor in GPT2 state_dict: " << name;
        return state_dict.at(name);
    };

    auto write_tensor_fp32 = [&](const std::shared_ptr<Tensor> &tensor) {
        Tensor cpu = tensor->To(Device());
        if (cpu.Dtype() != DataType::kFLOAT32) {
            cpu = cpu.To(DataType::kFLOAT32);
        }
        const auto bytes = static_cast<std::streamsize>(cpu.SizeInBytes());
        ofs.write(reinterpret_cast<const char *>(cpu.DataPtr()), bytes);
    };

    // transformer.wte.weight
    write_tensor_fp32(get_tensor(std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                             nn::TransformerFirstStage::kWTELayerName,
                                             nn::parallel::VocabParallelEmbedding::kParamWeightName)));

    // transformer.wpe.weight
    write_tensor_fp32(
        get_tensor(std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                               nn::TransformerFirstStage::kWPELayerName, nn::Embedding::kParamWeightName)));

    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(get_tensor(std::format(
            "{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName, idx,
            nn::TransformerLayer::kLn1LayerName, nn::LayerNorm::kParamWeightName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(get_tensor(std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                 nn::TransformerChunk::kHLayerName, idx,
                                                 nn::TransformerLayer::kLn1LayerName, nn::LayerNorm::kParamBiasName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(get_tensor(std::format(
            "{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName, idx,
            nn::TransformerLayer::kAttnLayerName, nn::CausalSelfAttention::kCAttnLayerName,
            nn::parallel::ColumnParallelLinear::kParamWeightName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(get_tensor(
            std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                        nn::TransformerChunk::kHLayerName, idx, nn::TransformerLayer::kAttnLayerName,
                        nn::CausalSelfAttention::kCAttnLayerName, nn::parallel::ColumnParallelLinear::kParamBiasName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(get_tensor(
            std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                        nn::TransformerChunk::kHLayerName, idx, nn::TransformerLayer::kAttnLayerName,
                        nn::CausalSelfAttention::kCProjLayerName, nn::parallel::RowParallelLinear::kParamWeightName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(get_tensor(
            std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                        nn::TransformerChunk::kHLayerName, idx, nn::TransformerLayer::kAttnLayerName,
                        nn::CausalSelfAttention::kCProjLayerName, nn::parallel::RowParallelLinear::kParamBiasName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(get_tensor(std::format(
            "{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName, idx,
            nn::TransformerLayer::kLn2LayerName, nn::LayerNorm::kParamWeightName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(get_tensor(std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                 nn::TransformerChunk::kHLayerName, idx,
                                                 nn::TransformerLayer::kLn2LayerName, nn::LayerNorm::kParamBiasName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(
            get_tensor(std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                   nn::TransformerChunk::kHLayerName, idx, nn::TransformerLayer::kMlpLayerName,
                                   nn::MLP::kCFcLayerName, nn::parallel::ColumnParallelLinear::kParamWeightName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(
            get_tensor(std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                   nn::TransformerChunk::kHLayerName, idx, nn::TransformerLayer::kMlpLayerName,
                                   nn::MLP::kCFcLayerName, nn::parallel::ColumnParallelLinear::kParamBiasName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(
            get_tensor(std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                   nn::TransformerChunk::kHLayerName, idx, nn::TransformerLayer::kMlpLayerName,
                                   nn::MLP::kCProjLayerName, nn::parallel::RowParallelLinear::kParamWeightName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(
            get_tensor(std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                   nn::TransformerChunk::kHLayerName, idx, nn::TransformerLayer::kMlpLayerName,
                                   nn::MLP::kCProjLayerName, nn::parallel::RowParallelLinear::kParamBiasName)));
    }

    write_tensor_fp32(
        get_tensor(std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                               nn::TransformerLastStage::kLnFLayerName, nn::LayerNorm::kParamWeightName)));
    write_tensor_fp32(get_tensor(std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                             nn::TransformerLastStage::kLnFLayerName, nn::LayerNorm::kParamBiasName)));

    ofs.flush();
    CHECK(ofs.good()) << "Failed to flush model file: " << filepath;
}
} // namespace gpt2

namespace llama3 {

std::shared_ptr<nn::TransformerModel> LoadFromLLMC(const std::string &filepath) {
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

    nn::TransformerConfig llama3_config = infini_train::llama3::LLaMA3Config();
    llama3_config.block_size = block_size;
    llama3_config.vocab_size = vocab_size;
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
    auto llama3 = std::make_shared<nn::TransformerModel>(llama3_config);

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
        for (size_t i = 0; i < layer_ranges_per_chunk.size(); ++i) {
            LOG(INFO) << "  Chunk " << i << ": layers " << layer_ranges_per_chunk[i].first << " to "
                      << layer_ranges_per_chunk[i].second;
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
    const int64_t attn_local_rows = q_local_rows + 2 * kv_local_rows;

    // RowParallel (proj)
    const int64_t in_pp = static_cast<int64_t>(n_embd) / tp_size;
    // nn::MLP: c_fc/c_fc2（shard along row），c_proj（shard along col）
    const int64_t fc_out = ffn_hidden;
    const int64_t fc_pp = fc_out / tp_size;
    const int64_t in_fc_pp = ffn_hidden / tp_size;

    auto state_dict = llama3->StateDict();

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

    // transformer.h.{i}.attn.c_attn.weight : ColumnParallelLinear, but actually applies on "rows"
    // W-qkv should be [Q(=n_embd) | K(=n_kv_head*head_dim) | V(=n_kv_head*head_dim)] × n_embd
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

    // transformer.h.{i}.mlp.c_fc.weight : ColumnParallelLinear, but actually applies on "rows"
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
            size_t fc_bytes = static_cast<size_t>(ffn_hidden) * n_embd * sizeof(float);
            ifs.seekg(fc_bytes, std::ios::cur);
        }
    }

    // transformer.h.{i}.mlp.c_fc2.weight : ColumnParallelLinear, but actually applies on "rows"
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

    return llama3;
}

void SaveAsLLMC(const std::shared_ptr<nn::TransformerModel> &model, const std::string &filepath) {
    CHECK_EQ(nn::parallel::global::GetTensorParallelSize(), 1) << "SaveAsLLMC currently supports TP=1 only.";
    CHECK_EQ(nn::parallel::global::GetPipelineParallelSize(), 1) << "SaveAsLLMC currently supports PP=1 only.";

    std::ofstream ofs(filepath, std::ios::binary);
    CHECK(ofs.is_open()) << "Failed to open model file for write: " << filepath;

    auto pack_float = [](float value) -> int32_t {
        int32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(float));
        return bits;
    };

    auto config = model->Config();
    std::vector<int32_t> header(256, 0);
    header[0] = kLLaMA3Magic;
    header[1] = kLLaMA3FP32Version;
    header[2] = static_cast<int32_t>(config.block_size);
    header[3] = static_cast<int32_t>(config.vocab_size);
    header[4] = static_cast<int32_t>(config.n_layer);
    header[5] = static_cast<int32_t>(config.n_head);
    header[6] = static_cast<int32_t>(config.n_kv_head);
    header[7] = static_cast<int32_t>(config.n_embd);
    header[8] = pack_float(config.ffn_dim_multiplier.value_or(0.0f));
    header[9] = static_cast<int32_t>(config.multiple_of);
    header[10] = pack_float(config.norm_eps);
    header[11] = pack_float(config.rope_theta);
    header[12] = static_cast<int32_t>(config.use_scaled_rope ? 1 : 0);
    header[13] = static_cast<int32_t>(config.max_gen_batch_size);
    header[14] = 1; // version_major
    header[15] = 0; // version_minor
    ofs.write(reinterpret_cast<const char *>(header.data()),
              static_cast<std::streamsize>(header.size() * sizeof(int32_t)));

    const auto state_dict = model->StateDict();
    auto get_tensor = [&](const std::string &name) -> std::shared_ptr<Tensor> {
        CHECK(state_dict.contains(name)) << "Missing tensor in LLaMA3 state_dict: " << name;
        return state_dict.at(name);
    };

    auto write_tensor_fp32 = [&](const std::shared_ptr<Tensor> &tensor) {
        Tensor cpu = tensor->To(Device());
        if (cpu.Dtype() != DataType::kFLOAT32) {
            cpu = cpu.To(DataType::kFLOAT32);
        }
        const auto bytes = static_cast<std::streamsize>(cpu.SizeInBytes());
        ofs.write(reinterpret_cast<const char *>(cpu.DataPtr()), bytes);
    };

    write_tensor_fp32(get_tensor(std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                             nn::TransformerFirstStage::kWTELayerName,
                                             nn::parallel::VocabParallelEmbedding::kParamWeightName)));

    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(get_tensor(std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                 nn::TransformerChunk::kHLayerName, idx,
                                                 nn::TransformerLayer::kLn1LayerName, nn::RMSNorm::kParamWeightName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(get_tensor(std::format(
            "{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName, nn::TransformerChunk::kHLayerName, idx,
            nn::TransformerLayer::kAttnLayerName, nn::CausalSelfAttention::kCAttnLayerName,
            nn::parallel::ColumnParallelLinear::kParamWeightName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(get_tensor(
            std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                        nn::TransformerChunk::kHLayerName, idx, nn::TransformerLayer::kAttnLayerName,
                        nn::CausalSelfAttention::kCProjLayerName, nn::parallel::RowParallelLinear::kParamWeightName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(get_tensor(std::format("{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                                 nn::TransformerChunk::kHLayerName, idx,
                                                 nn::TransformerLayer::kLn2LayerName, nn::RMSNorm::kParamWeightName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(
            get_tensor(std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                   nn::TransformerChunk::kHLayerName, idx, nn::TransformerLayer::kMlpLayerName,
                                   nn::MLP::kCFcLayerName, nn::parallel::ColumnParallelLinear::kParamWeightName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(
            get_tensor(std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                   nn::TransformerChunk::kHLayerName, idx, nn::TransformerLayer::kMlpLayerName,
                                   nn::MLP::kCFc2LayerName, nn::parallel::ColumnParallelLinear::kParamWeightName)));
    }
    for (int idx = 0; idx < config.n_layer; ++idx) {
        write_tensor_fp32(
            get_tensor(std::format("{}.{}.{}.{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                   nn::TransformerChunk::kHLayerName, idx, nn::TransformerLayer::kMlpLayerName,
                                   nn::MLP::kCProjLayerName, nn::parallel::RowParallelLinear::kParamWeightName)));
    }

    write_tensor_fp32(get_tensor(std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                             nn::TransformerLastStage::kLnFLayerName, nn::RMSNorm::kParamWeightName)));
    write_tensor_fp32(get_tensor(std::format("{}.{}", nn::TransformerLastStage::kLMHeadLayerName,
                                             nn::parallel::ColumnParallelLinear::kParamWeightName)));

    ofs.flush();
    CHECK(ofs.good()) << "Failed to flush model file: " << filepath;
}
} // namespace llama3
  // namespace infini_train

ResumeFromCheckpointResult ResumeFromCheckpoint(const ResumeFromCheckpointArgs &args) {
    ResumeFromCheckpointResult result;
    int ddp_world_size = nn::parallel::global::GetDataParallelSize();

    if (args.resume_root.empty()) {
        LOG(INFO) << "No checkpoint specified for resume. Starting training from scratch.";
        return result;
    }

    std::filesystem::path resume_dir = args.resume_root;
    if (args.rank.IsParallel()) {
        const auto rank_dir = resume_dir / std::format("rank_{:06d}", args.rank.GlobalRank());
        if (std::filesystem::exists(rank_dir)) {
            resume_dir = rank_dir;
        }
    }

    Checkpoint::Load(resume_dir, args.model.get(), args.optimizer.get(), &args.state, args.load_options);

    result.global_step = static_cast<int>(args.state.global_step);
    result.best_loss = args.state.best_loss;
    if (args.state.data_batch_stride != static_cast<int64_t>(ddp_world_size)) {
        LOG(FATAL) << std::format("Checkpoint data_batch_stride {} mismatches current ddp_world_size {}. "
                                  "Proceeding with recorded data_batch_idx {}.",
                                  args.state.data_batch_stride, ddp_world_size, args.state.data_batch_idx);
    }
    result.data_batch_idx = static_cast<size_t>(std::max<int64_t>(args.state.data_batch_idx, 0));
    args.train_iter = args.train_loader.IteratorAtBatchIndex(result.data_batch_idx);
    if (args.rank.IsMainRank()) {
        LOG(INFO) << std::format(
            "Resume training from step {} with best_loss {:.6f}, last_lr {:.3e}, data_batch_idx {}",
            args.state.global_step, args.state.best_loss, args.state.last_lr, args.state.data_batch_idx);
    }

    return result;
}

void SaveCheckpoint(const SaveCheckpointArgs &args) {
    const auto ckpt_start = std::chrono::high_resolution_clock::now();

    TrainerState state;
    state.global_step = args.global_step;
    state.data_batch_idx = static_cast<int64_t>(args.data_batch_idx);
    state.data_batch_stride = args.ddp_size;
    state.best_loss = args.best_loss;
    state.last_lr = args.last_lr;
    state.optimizer_type = args.optimizer_type;
    state.checkpoint_format = args.checkpoint_format;
    state.ddp_size = args.ddp_size;
    state.tp_size = args.tp_size;
    state.sp_size = args.sp_size;
    state.pp_size = args.pp_size;

    CheckpointOptions options;
    options.format = args.checkpoint_format;
    options.save_optimizer_state = args.save_optimizer_state;
    options.model_bin_writer = args.model_bin_writer;
    Checkpoint::Save(args.save_dir, args.model, args.optimizer, state, options);

    const auto ckpt_end = std::chrono::high_resolution_clock::now();
    const double ckpt_ms = std::chrono::duration<double, std::milli>(ckpt_end - ckpt_start).count();

    if (!args.rank.IsMainRank()) {
        return;
    }

    LOG(INFO) << std::format("Checkpoint saved at: {} ({:.2f} ms)", args.save_dir.string(), ckpt_ms);

    if (!args.prune_step_checkpoints) {
        return;
    }

    std::vector<std::filesystem::path> ckpts;
    if (std::filesystem::exists(args.checkpoint_root_dir)) {
        for (const auto &entry : std::filesystem::directory_iterator(args.checkpoint_root_dir)) {
            if (entry.is_directory() && entry.path().filename().string().starts_with("checkpoint_step_")) {
                ckpts.push_back(entry.path());
            }
        }
        std::sort(ckpts.begin(), ckpts.end());
        while (ckpts.size() > args.max_checkpoint_keep) {
            std::filesystem::remove_all(ckpts.front());
            ckpts.erase(ckpts.begin());
        }
    }
}
} // namespace infini_train
