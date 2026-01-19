#include "infini_train/include/models/llama3/llama3.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "example/common/utils.h"
#include "infini_train/include/device.h"
#include "infini_train/include/models/llama3/spec.h"
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
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;
namespace nn = infini_train::nn;

namespace {
// Used in Grouped Query Attention(GQA), broadcasts the key and value tensors
// FIXME(zbl): implement Expand() instead of using RepeatInterleave()
std::shared_ptr<Tensor> RepeatKV(const std::shared_ptr<Tensor> &x, int64_t n_rep) {
    const auto &shape = x->Dims();
    const int64_t B = shape[0], T = shape[1], H = shape[2], D = shape[3];
    if (n_rep == 1) {
        return x;
    }
    return x->View({B, T, H, 1, D})->RepeatInterleave(n_rep, 3)->Contiguous()->View({B, T, H * n_rep, D});
}

// -----------------------------------------------------------------
// RoPE related
// NOTE(zbl): this RoPE implementation has no "learnable" params, as is stated in LLaMA paper
std::shared_ptr<Tensor> ReshapeForBroadcast(const std::shared_ptr<Tensor> &freqs_cis,
                                            const std::shared_ptr<Tensor> &x) {
    // freqs_cis: (T, D / 2, 2)
    CHECK(freqs_cis != nullptr) << "freqs_cis is null.";
    const auto &x_shape = x->Dims(); // (B, T, H, D)
    CHECK_GE(x_shape.size(), 4);
    const int64_t T = x_shape[1];
    const int64_t D = x_shape[3];
    CHECK_EQ(freqs_cis->Dims()[0], x_shape[1]);
    CHECK_EQ(freqs_cis->Dims()[1], x_shape[3] / 2);
    std::vector<int64_t> target_shape = {1, T, 1, D / 2, 2};
    return freqs_cis->View(target_shape);
}

// TODO(zbl): ApplyScaling(const std::shared_ptr<Tensor> &) when use_scaled
// std::shared_ptr<Tensor> ApplyScaling(const std::shared_ptr<Tensor> &freqs, float old_context_len = 8192) {}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
ApplyRotaryEmbedding(const std::shared_ptr<Tensor> &xq, const std::shared_ptr<Tensor> &xk,
                     const std::shared_ptr<Tensor> &freqs_cis) {
    // Shape assumptions: xq: (B, T, H, D)
    auto cos_sin = ReshapeForBroadcast(freqs_cis, xq); // -> (1, T, 1, D/2, 2)
    std::vector<int64_t> target_shape(cos_sin->Dims().begin(), cos_sin->Dims().end() - 1);
    auto cos = cos_sin->Slice(-1, 0, 1, 1)->Squeeze(-1); // (1, T, 1, D/2)
    auto sin = cos_sin->Slice(-1, 1, 2, 1)->Squeeze(-1); // (1, T, 1, D/2)

    auto slice_pair = [](const std::shared_ptr<Tensor> &x) {
        auto even = x->Slice(-1, 0, x->Dims().back(), 2);
        auto odd = x->Slice(-1, 1, x->Dims().back(), 2);
        return std::make_pair(even, odd);
    };

    auto [q_even, q_odd] = slice_pair(xq);
    auto q_rotated_left = q_even * cos - q_odd * sin;
    auto q_rotated_right = q_even * sin + q_odd * cos;
    auto q_rotated
        = nn::function::Stack(std::vector<std::shared_ptr<Tensor>>{q_rotated_left, q_rotated_right}, -1)->Flatten(-2);

    auto [k_even, k_odd] = slice_pair(xk);
    auto k_rotated_left = k_even * cos - k_odd * sin;
    auto k_rotated_right = k_even * sin + k_odd * cos;
    auto k_rotated
        = nn::function::Stack(std::vector<std::shared_ptr<Tensor>>{k_rotated_left, k_rotated_right}, -1)->Flatten(-2);

    return {q_rotated, k_rotated};
}

} // namespace

std::vector<std::shared_ptr<Tensor>> SwiGLU::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    return {x[0] * nn::function::Sigmoid(x[0])};
}

RMSNorm::RMSNorm(int64_t dim, float eps, const infini_train::Device *device) : eps_(eps) {
    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{dim}, DataType::kFLOAT32, device)->RequiresGrad();
    nn::init::Ones(parameters_[kParamWeightName]);
}

std::vector<std::shared_ptr<Tensor>> RMSNorm::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    // broadcasted Mul([4, 64, 2048] * [4, 64, 1])
    auto norm = x[0] * nn::function::Rsqrt(nn::function::Mean(nn::function::Pow(x[0], 2), -1, true) + eps_);
    return {norm * parameters_[kParamWeightName]};
}

CausalSelfAttention::CausalSelfAttention(const nn::TransformerConfig &config)
    : config_(config), n_head_(config.n_head), n_embd_(config.n_embd), n_kv_head_(config.n_kv_head),
      n_rep_(config.n_head / config.n_kv_head), head_dim_(config.n_embd / config.n_head) {
    CHECK_LE(config.n_kv_head, config.n_head);
    CHECK_EQ(config.n_head % config.n_kv_head, 0);
    CHECK_EQ(config.n_embd % config.n_head, 0);

    int64_t qkv_dim = (config.n_head + 2 * n_kv_head_) * head_dim_;
    // qkv: ColumnParallel (do not gather output)
    modules_[kCAttnLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
        /*in_features=*/n_embd_,
        /*out_features=*/qkv_dim,
        /*bias=*/false,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    // proj: RowParallel (input is parallel and output is full)
    modules_[kCProjLayerName] = std::make_shared<nn::parallel::RowParallelLinear>(
        /*in_features=*/n_embd_,
        /*out_features=*/n_embd_,
        /*bias=*/false,
        /*reduce_output=*/true,
        /*input_is_parallel=*/true,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());
}

std::vector<std::shared_ptr<Tensor>> CausalSelfAttention::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    const auto B = x[0]->Dims()[0]; // bs
    const auto C = x[0]->Dims()[2]; // n_embd

    const auto tp_size = nn::parallel::global::GetTensorParallelSize();

    const auto C_local = C / tp_size;
    const auto H_local = n_head_ / tp_size;
    const auto KV_local = n_kv_head_ / tp_size;
    const auto D = head_dim_; // n_embd / n_head

    const auto freqs_cis = x.size() > 1 ? x[1] : nullptr;
    const auto start_pos = x.size() > 2 ? x[2] : nullptr;
    const auto mask = x.size() > 3 ? x[3] : nullptr;
    CHECK(freqs_cis != nullptr) << "freqs_cis is null.";

    // (B, T, C) -> (B, T, (H + 2 * n_kv_head) * D)
    auto qkv = modules_[kCAttnLayerName]->Forward({x[0]})[0];
    // NOTE(zbl): Acquire full T after AllGather is performed in ColumnParallelLinear
    const auto T = qkv->Dims()[1];
    // NOTE(zbl): torch script uses torch.split({...}, dim) to split tensors into sub-tensors in different sizes
    //            use Slice() to work around here
    const int64_t q_size_local = H_local * D;
    const int64_t kv_size_local = KV_local * D;
    // -> Split into q, k, v
    // q: (B, T, H_local, D)
    auto q = qkv->Slice(2, 0, q_size_local)->View({B, T, H_local, D});
    // k: (B, T, KV_local, D)
    auto k = qkv->Slice(2, q_size_local, q_size_local + kv_size_local)->View({B, T, KV_local, D});
    // v: (B, T, KV_local, D)
    auto v = qkv->Slice(2, q_size_local + kv_size_local, q_size_local + 2 * kv_size_local)->View({B, T, KV_local, D});

    // -> RoPE on q, k
    // q: (B, T, H_local, D)
    // k: (B, T, KV_local, D)
    std::tie(q, k) = ApplyRotaryEmbedding(q, k, freqs_cis);

    // TODO(zbl): use kv cache during inference
    // if (use_kv_) { ... }

    // align n_head in GQA
    // (B, T, KV_local, D) -> (B, T, H_local, D) via RepeatKV
    k = RepeatKV(k, n_rep_);
    v = RepeatKV(v, n_rep_);

    // (B, T, H_local, D) -> (B, H_local, T, D)
    q = q->Transpose(1, 2);
    k = k->Transpose(1, 2);
    v = v->Transpose(1, 2);

    // TODO(zbl): support flash attention later
    // if (flash_) { ... }

    // manual implementation of attention
    // this materializes the large (T,T) matrix for all the queries and keys

    // q: (B, H_local, T, D)
    // k: (B, H_local, T, D) -> (B, H_local, D, T)
    // q @ k.T: (B, H_local, T, T) -> mul 1.0 / sqrt(D) -> (B, H_local, T, T)
    auto att = q->Matmul(k->Transpose(-2, -1)) * (1.0 / std::sqrt(static_cast<float>(D)));
    if (mask) {
        // mask: (1, 1, T, T)
        att = att->MaskedFill(mask, std::numeric_limits<float>::lowest());
    }
    // (B, H_local, T, T)
    att = nn::function::Softmax(att, -1);
    // att: (B, H_local, T, T) @ v: (B, H_local, T, D) -> y: (B, H_local, T, D)
    auto y = att->Matmul(v);
    // (B, H_local, T, D) -> Transpose(1, 2) -> (B, T, H_local, D) -> (B, T, C_local)
    y = y->Transpose(1, 2)->Contiguous()->View({B, T, C_local});
    // output projection
    // (B, T, C_local) -> RowParallelLinear(C, C) -> (B, T, C)
    y = modules_[kCProjLayerName]->Forward({y})[0];
    // (B, H, C) == (bs, seq_len, n_embd)
    return {y};
}

MLP::MLP(const nn::TransformerConfig &config) {
    hidden_dim_ = 4 * config.n_embd;
    hidden_dim_ = int(2 * hidden_dim_ / 3);
    // use custom dim factor multiplier
    if (config.ffn_dim_multiplier.has_value()) {
        hidden_dim_ = int(config.ffn_dim_multiplier.value() * hidden_dim_);
    }
    hidden_dim_ = config.multiple_of * ((hidden_dim_ + config.multiple_of - 1) / config.multiple_of);

    // c_fc: ColumnParallel (input full, output parallel)
    modules_[kCFcLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
        /*in_features=*/config.n_embd, /*out_features=*/hidden_dim_,
        /*bias=*/false,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    // c_fc2: ColumnParallel (input full, output parallel)
    modules_[kCFc2LayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
        /*in_features=*/config.n_embd, /*out_features=*/hidden_dim_,
        /*bias=*/false,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    modules_[kSiluLayerName] = std::make_shared<SwiGLU>();

    // c_proj: RowParallel (input parallel, output full)
    modules_[kCProjLayerName] = std::make_shared<nn::parallel::RowParallelLinear>(
        /*in_features=*/hidden_dim_, /*out_features=*/config.n_embd,
        /*bias=*/false,
        /*reduce_output=*/true,
        /*input_is_parallel=*/true,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());
}

std::vector<std::shared_ptr<Tensor>> MLP::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    // (bs, seq_len, n_embd) -> Linear(n_embd, hidden_dim) -> (bs, seq_len, hidden_dim)
    auto x1 = modules_[kCFcLayerName]->Forward(x)[0];
    // (bs, seq_len, n_embd) -> Linear(n_embd, hidden_dim) -> (bs, seq_len, hidden_dim)
    auto x2 = modules_[kCFc2LayerName]->Forward(x)[0];
    // (bs, seq_len, hidden_dim) -> SwiGLU -> (bs, seq_len, hidden_dim)
    x2 = modules_[kSiluLayerName]->Forward({x2})[0];
    // (bs, seq_len, hidden_dim)
    auto x3 = x1 * x2;
    // (bs, seq_len, hidden_dim) -> Linear(hidden_dim, n_embd) -> (bs, seq_len, n_embd)
    auto x4 = modules_[kCProjLayerName]->Forward({x3});
    // (bs, seq_len, n_embd)
    return x4;
}

Block::Block(const nn::TransformerConfig &config) {
    modules_[kLn1LayerName] = std::make_shared<RMSNorm>(config.n_embd, config.norm_eps);
    modules_[kAttnLayerName] = std::make_shared<CausalSelfAttention>(config);
    modules_[kLn2LayerName] = std::make_shared<RMSNorm>(config.n_embd, config.norm_eps);
    modules_[kMlpLayerName] = std::make_shared<MLP>(config);
}

std::vector<std::shared_ptr<Tensor>> Block::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    const auto freqs_cis = x.size() > 1 ? x[1] : nullptr;
    const auto start_pos = x.size() > 2 ? x[2] : nullptr;
    const auto mask = x.size() > 3 ? x[3] : nullptr;

    // (bs, seq_len, n_embd) -> RMSNorm -> (bs, seq_len, n_embd) -> attention -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x1 = x[0]
            + modules_[kAttnLayerName]->Forward(std::vector<std::shared_ptr<Tensor>>{
                modules_[kLn1LayerName]->Forward({x[0]})[0], freqs_cis, start_pos, mask})[0];
    // (bs, seq_len, n_embd) -> RMSNorm -> (bs, seq_len, n_embd) -> MLP -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x2 = x1
            + modules_[kMlpLayerName]->Forward(
                std::vector<std::shared_ptr<Tensor>>(modules_[kLn2LayerName]->Forward({x1})))[0];
    // (bs, seq_len, n_embd)
    return {x2};
}

std::shared_ptr<nn::Module> LLaMA3Kernel::MakeBlock(const nn::TransformerConfig &config) {
    return std::make_shared<Block>(config); // uses RoPE + GQA + SwiGLU
}

std::shared_ptr<infini_train::nn::Module> LLaMA3Kernel::MakeFinalNorm(const nn::TransformerConfig &config) {
    return std::make_shared<RMSNorm>(config.n_embd);
}

LLaMA3::LLaMA3(const nn::TransformerConfig &config)
    : config_(config), stage_info_(nn::parallel::PipelineParallel::GetStageInfo(
                           config_.n_layer, nn::parallel::global::GetPipelineParallelSize(), nn::parallel::pp_rank,
                           nn::parallel::global::GetVirtualPipelineParallelSize())) {
    std::unordered_map<std::string, std::shared_ptr<nn::Module>> transformer;

    auto kernel = std::make_shared<LLaMA3Kernel>();

    if (stage_info_.is_first_stage) {
        modules_[kPPFirstStageName] = std::make_shared<nn::TransformerFirstStageABI>(config_, kernel);
        transformer[nn::TransformerFirstStageABI::kWTELayerName]
            = modules_[kPPFirstStageName]->mutable_module(nn::TransformerFirstStageABI::kWTELayerName);
    }

    {
        std::map<int, std::pair<int, std::shared_ptr<nn::TransformerChunkABI>>> start_layer_to_layer_size_and_chunk;
        for (int chunk_idx = 0; chunk_idx < stage_info_.layer_ranges_per_chunk.size(); ++chunk_idx) {
            const auto [start_layer, end_layer] = stage_info_.layer_ranges_per_chunk[chunk_idx];
            // fix(jym):更通用的构造chunk的方法
            auto chunk = std::make_shared<nn::LLaMA3ChunkABI>(config_, start_layer, end_layer, kernel);
            start_layer_to_layer_size_and_chunk[start_layer] = std::make_pair(end_layer - start_layer, chunk);
        }
        std::vector<std::shared_ptr<nn::Module>> h;
        int chunk_idx = 0;
        for (auto &[start_layer, layer_size_and_chunk] : start_layer_to_layer_size_and_chunk) {
            auto [layer_size, chunk] = layer_size_and_chunk;
            for (int idx = 0; idx < layer_size; ++idx) {
                h.push_back(
                    chunk->mutable_module(nn::TransformerChunkABI::kHLayerName)->mutable_module(std::to_string(idx)));
            }
            modules_[kPPChunkNamePrefix + std::to_string(chunk_idx)] = std::move(chunk);
            ++chunk_idx;
        }
        transformer[nn::TransformerChunkABI::kHLayerName] = std::make_shared<nn::ModuleList>(std::move(h));
    }

    if (stage_info_.is_last_stage) {
        modules_[kPPLastStageName] = std::make_shared<nn::TransformerLastStageABI>(config_, kernel);
        transformer[nn::TransformerLastStageABI::kLnFLayerName]
            = modules_[kPPLastStageName]->mutable_module(nn::TransformerLastStageABI::kLnFLayerName);
        // NOTE(zbl): weight-tying is possible but torch script did not do so
        modules_[nn::TransformerLastStageABI::kLMHeadLayerName]
            = modules_[kPPLastStageName]->mutable_module(nn::TransformerLastStageABI::kLMHeadLayerName);
    }
    modules_[kTransformerLayerName] = std::make_shared<nn::ModuleDict>(std::move(transformer));
}

std::vector<std::shared_ptr<Tensor>> LLaMA3::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    auto x1 = modules_[kPPFirstStageName]->Forward({x[0]});
    for (int chunk_idx = 0; chunk_idx < stage_info_.layer_ranges_per_chunk.size(); ++chunk_idx) {
        x1 = modules_[kPPChunkNamePrefix + std::to_string(chunk_idx)]->Forward(x1);
    }
    return modules_[kPPLastStageName]->Forward(x1);
}
