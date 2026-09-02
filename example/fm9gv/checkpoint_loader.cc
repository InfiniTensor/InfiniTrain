#include "example/fm9gv/checkpoint_loader.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "example/common/utils.h"
#include "example/fm9gv/config.h"
#include "infini_train/include/nn/modules/linear.h"
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
constexpr int32_t kFM9GMagic = 20260826;
constexpr int32_t kFM9GFP32Version = 1;
} // namespace

namespace fm9gv {

std::shared_ptr<nn::TransformerModel> LoadFromFM9GBin(const std::string &filepath) {
    if (!std::filesystem::exists(filepath)) {
        LOG(FATAL) << "File not found: " << filepath;
    }

    std::ifstream ifs(filepath, std::ios::binary);
    const auto header = ReadSeveralBytesFromIfstream(256 * sizeof(int32_t), &ifs);

    const auto magic = BytesToType<int32_t>(header, 0);
    CHECK_EQ(magic, kFM9GMagic);
    const auto version = BytesToType<int32_t>(header, 4);
    CHECK_EQ(version, kFM9GFP32Version);

    nn::TransformerConfig config = fm9gv::FM9GVTextConfig();
    config.block_size = BytesToType<int32_t>(header, 8);
    config.vocab_size = BytesToType<int32_t>(header, 12);
    config.original_vocab_size = config.vocab_size;
    config.n_layer = BytesToType<int32_t>(header, 16);
    config.n_head = BytesToType<int32_t>(header, 20);
    config.n_kv_head = config.n_head;
    config.n_embd = BytesToType<int32_t>(header, 24);
    config.ffn_hidden_size = BytesToType<int32_t>(header, 28);
    config.q_lora_rank = BytesToType<int32_t>(header, 32);
    config.kv_lora_rank = BytesToType<int32_t>(header, 36);
    config.qk_nope_head_dim = BytesToType<int32_t>(header, 40);
    config.qk_rope_head_dim = BytesToType<int32_t>(header, 44);
    config.v_head_dim = BytesToType<int32_t>(header, 48);
    config.rope_theta = BytesToType<float>(header, 52);
    config.scale_emb = BytesToType<float>(header, 56);
    config.scale_depth = BytesToType<float>(header, 60);
    config.norm_eps = BytesToType<float>(header, 64);

    auto model = std::make_shared<nn::TransformerModel>(config);
    auto state_dict = model->StateDict();

    const int pp_size = nn::parallel::global::GetPipelineParallelSize();
    const int vpp_size = nn::parallel::global::GetVirtualPipelineParallelSize();
    auto [is_first_stage, is_last_stage, layer_ranges_per_chunk]
        = nn::parallel::PipelineParallel::GetStageInfo(config.n_layer, pp_size, nn::parallel::pp_rank, vpp_size);

    std::vector<bool> owned_layers(config.n_layer, false);
    for (const auto &[start, end] : layer_ranges_per_chunk) {
        for (int i = start; i < end; ++i) { owned_layers[i] = true; }
    }

    const int tp_size = nn::parallel::global::GetTensorParallelSize();
    const int tp_rank = nn::parallel::tp_rank;
    CHECK_EQ(config.n_head % tp_size, 0);
    CHECK_EQ(config.vocab_size % tp_size, 0);
    CHECK_EQ(config.ffn_hidden_size % tp_size, 0);

    const int64_t vocab_rows = config.vocab_size / tp_size;
    const int64_t vocab_start = static_cast<int64_t>(tp_rank) * vocab_rows;
    const int64_t local_heads = config.n_head / tp_size;
    const int64_t q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;
    const int64_t q_b_rows = config.n_head * q_head_dim;
    const int64_t q_b_rows_local = local_heads * q_head_dim;
    const int64_t kv_b_rows = config.n_head * (config.qk_nope_head_dim + config.v_head_dim);
    const int64_t kv_b_rows_local = local_heads * (config.qk_nope_head_dim + config.v_head_dim);
    const int64_t o_proj_cols = config.n_head * config.v_head_dim;
    const int64_t o_proj_cols_local = local_heads * config.v_head_dim;
    const int64_t ffn_rows_local = config.ffn_hidden_size / tp_size;

    if (is_first_stage) {
        auto &wte = state_dict[std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                           nn::TransformerFirstStage::kWTELayerName,
                                           nn::parallel::VocabParallelEmbedding::kParamWeightName)];
        ReadMatrixRowShardFloat(ifs, static_cast<float *>(wte->DataPtr()), config.vocab_size, config.n_embd,
                                vocab_start, vocab_rows);
    } else {
        ifs.seekg(static_cast<size_t>(config.vocab_size) * config.n_embd * sizeof(float), std::ios::cur);
    }

    int local_layer_index = 0;
    for (int i = 0; i < static_cast<int>(config.n_layer); ++i) {
        if (owned_layers[i]) {
            auto layer_prefix = std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                            nn::TransformerChunk::kHLayerName, local_layer_index);

            ReadVectorAllFloat(ifs,
                               static_cast<float *>(state_dict[std::format("{}.{}.{}", layer_prefix,
                                                                            nn::TransformerLayer::kLn1LayerName,
                                                                            nn::RMSNorm::kParamWeightName)]
                                                        ->DataPtr()),
                               config.n_embd);
            ReadMatrixAllFloat(ifs,
                               static_cast<float *>(state_dict[std::format("{}.{}.{}.{}", layer_prefix,
                                                                            nn::TransformerLayer::kAttnLayerName,
                                                                            nn::CausalSelfAttention::kQAProjLayerName,
                                                                            nn::Linear::kParamWeightName)]
                                                        ->DataPtr()),
                               config.q_lora_rank, config.n_embd);
            ReadVectorAllFloat(
                ifs,
                static_cast<float *>(state_dict[std::format("{}.{}.{}.{}", layer_prefix,
                                                            nn::TransformerLayer::kAttnLayerName,
                                                            nn::CausalSelfAttention::kQALayerNormLayerName,
                                                            nn::RMSNorm::kParamWeightName)]
                                         ->DataPtr()),
                config.q_lora_rank);
            ReadMatrixRowShardFloat(
                ifs,
                static_cast<float *>(state_dict[std::format("{}.{}.{}.{}", layer_prefix,
                                                            nn::TransformerLayer::kAttnLayerName,
                                                            nn::CausalSelfAttention::kQBProjLayerName,
                                                            nn::parallel::ColumnParallelLinear::kParamWeightName)]
                                         ->DataPtr()),
                q_b_rows, config.q_lora_rank, tp_rank * q_b_rows_local, q_b_rows_local);
            ReadMatrixAllFloat(ifs,
                               static_cast<float *>(state_dict[std::format("{}.{}.{}.{}", layer_prefix,
                                                                            nn::TransformerLayer::kAttnLayerName,
                                                                            nn::CausalSelfAttention::kKVAProjWithMQALayerName,
                                                                            nn::Linear::kParamWeightName)]
                                                        ->DataPtr()),
                               config.kv_lora_rank + config.qk_rope_head_dim, config.n_embd);
            ReadVectorAllFloat(
                ifs,
                static_cast<float *>(state_dict[std::format("{}.{}.{}.{}", layer_prefix,
                                                            nn::TransformerLayer::kAttnLayerName,
                                                            nn::CausalSelfAttention::kKVALayerNormLayerName,
                                                            nn::RMSNorm::kParamWeightName)]
                                         ->DataPtr()),
                config.kv_lora_rank);
            ReadMatrixRowShardFloat(
                ifs,
                static_cast<float *>(state_dict[std::format("{}.{}.{}.{}", layer_prefix,
                                                            nn::TransformerLayer::kAttnLayerName,
                                                            nn::CausalSelfAttention::kKVBProjLayerName,
                                                            nn::parallel::ColumnParallelLinear::kParamWeightName)]
                                         ->DataPtr()),
                kv_b_rows, config.kv_lora_rank, tp_rank * kv_b_rows_local, kv_b_rows_local);
            ReadMatrixColShardFloat(
                ifs,
                static_cast<float *>(state_dict[std::format("{}.{}.{}.{}", layer_prefix,
                                                            nn::TransformerLayer::kAttnLayerName,
                                                            nn::CausalSelfAttention::kOProjLayerName,
                                                            nn::parallel::RowParallelLinear::kParamWeightName)]
                                         ->DataPtr()),
                config.n_embd, o_proj_cols, tp_rank * o_proj_cols_local, o_proj_cols_local);
            ReadVectorAllFloat(ifs,
                               static_cast<float *>(state_dict[std::format("{}.{}.{}", layer_prefix,
                                                                            nn::TransformerLayer::kLn2LayerName,
                                                                            nn::RMSNorm::kParamWeightName)]
                                                        ->DataPtr()),
                               config.n_embd);
            ReadMatrixRowShardFloat(
                ifs,
                static_cast<float *>(state_dict[std::format("{}.{}.{}.{}", layer_prefix,
                                                            nn::TransformerLayer::kMlpLayerName, nn::MLP::kCFc2LayerName,
                                                            nn::parallel::ColumnParallelLinear::kParamWeightName)]
                                         ->DataPtr()),
                config.ffn_hidden_size, config.n_embd, tp_rank * ffn_rows_local, ffn_rows_local);
            ReadMatrixRowShardFloat(
                ifs,
                static_cast<float *>(state_dict[std::format("{}.{}.{}.{}", layer_prefix,
                                                            nn::TransformerLayer::kMlpLayerName, nn::MLP::kCFcLayerName,
                                                            nn::parallel::ColumnParallelLinear::kParamWeightName)]
                                         ->DataPtr()),
                config.ffn_hidden_size, config.n_embd, tp_rank * ffn_rows_local, ffn_rows_local);
            ReadMatrixColShardFloat(
                ifs,
                static_cast<float *>(state_dict[std::format("{}.{}.{}.{}", layer_prefix,
                                                            nn::TransformerLayer::kMlpLayerName,
                                                            nn::MLP::kCProjLayerName,
                                                            nn::parallel::RowParallelLinear::kParamWeightName)]
                                         ->DataPtr()),
                config.n_embd, config.ffn_hidden_size, tp_rank * ffn_rows_local, ffn_rows_local);
            ++local_layer_index;
        } else {
            const size_t layer_bytes
                = (static_cast<size_t>(config.n_embd) + static_cast<size_t>(config.q_lora_rank) * config.n_embd
                   + static_cast<size_t>(config.q_lora_rank)
                   + static_cast<size_t>(q_b_rows) * config.q_lora_rank
                   + static_cast<size_t>(config.kv_lora_rank + config.qk_rope_head_dim) * config.n_embd
                   + static_cast<size_t>(config.kv_lora_rank)
                   + static_cast<size_t>(kv_b_rows) * config.kv_lora_rank
                   + static_cast<size_t>(config.n_embd) * o_proj_cols + static_cast<size_t>(config.n_embd)
                   + 2 * static_cast<size_t>(config.ffn_hidden_size) * config.n_embd
                   + static_cast<size_t>(config.n_embd) * config.ffn_hidden_size)
                  * sizeof(float);
            ifs.seekg(layer_bytes, std::ios::cur);
        }
    }

    if (is_last_stage) {
        auto &ln_f = state_dict[std::format("{}.{}.{}", nn::TransformerModel::kTransformerModelName,
                                            nn::TransformerLastStage::kLnFLayerName, nn::RMSNorm::kParamWeightName)];
        auto &lm_head = state_dict[std::format("{}.{}", nn::TransformerLastStage::kLMHeadLayerName,
                                               nn::parallel::ColumnParallelLinear::kParamWeightName)];
        ReadVectorAllFloat(ifs, static_cast<float *>(ln_f->DataPtr()), config.n_embd);
        ReadMatrixRowShardFloat(ifs, static_cast<float *>(lm_head->DataPtr()), config.vocab_size, config.n_embd,
                                vocab_start, vocab_rows);
    } else {
        ifs.seekg((static_cast<size_t>(config.n_embd) + static_cast<size_t>(config.vocab_size) * config.n_embd)
                      * sizeof(float),
                  std::ios::cur);
    }

    return model;
}

} // namespace fm9gv

