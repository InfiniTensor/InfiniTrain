#include "example/llama3/net.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;
namespace nn = infini_train::nn;

namespace {
constexpr int kRandomSeed = 42;

// TODO(zbl): make this rng generator compatible with torch later
static std::mt19937 gen{kRandomSeed};
} // namespace

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

std::shared_ptr<Tensor> PrecomputeFreqsCis(int64_t dim, int64_t end, float theta = 10000.0f, bool use_scaled = false,
                                           infini_train::Device device = infini_train::Device()) {
    CHECK_GE(dim, 2) << "dim must be >= 2 for slicing";
    auto arange = nn::init::Arange(0, dim, DataType::kFLOAT32, device)->Slice(0, 0, dim, 2);
    auto freqs = 1.0f / nn::function::Pow(theta, arange / float(dim));
    // TODO(zbl): use_scaled
    // if (use_scaled) {
    //     freqs = ApplyScaling(freqs, 8192.0f);
    // }
    auto t = nn::init::Arange(0, end, DataType::kFLOAT32, device);
    // (end, dim / 2)
    auto freqs_outer = t->Outer(freqs);
    auto cos = nn::function::Cos(freqs_outer);
    auto sin = nn::function::Sin(freqs_outer);
    // NOTE(zbl): torch script uses cis expression, here use stack
    // (end, dim / 2, 2)
    auto freqs_cis = nn::function::Stack(std::vector<std::shared_ptr<Tensor>>{cos, sin}, -1)->Contiguous();
    return freqs_cis;
}

} // namespace

std::vector<std::shared_ptr<Tensor>> SwiGLU::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    return {x[0] * nn::function::Sigmoid(x[0])};
}

RMSNorm::RMSNorm(int64_t dim, float eps, infini_train::Device device) : eps_(eps) {
    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{dim}, DataType::kFLOAT32, device)->RequiresGrad();
    nn::init::Ones(parameters_[kParamWeightName]);
}

std::vector<std::shared_ptr<Tensor>> RMSNorm::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    // broadcasted Mul([4, 64, 2048] * [4, 64, 1])
    auto norm = x[0] * nn::function::Rsqrt(nn::function::Mean(nn::function::Pow(x[0], 2), -1, true) + eps_);
    return {norm * parameters_[kParamWeightName]};
}

CausalSelfAttention::CausalSelfAttention(const LLaMA3Config &config)
    : config_(config), n_head_(config.n_head), n_embd_(config.n_embd), n_kv_head_(config.n_kv_head),
      n_rep_(config.n_head / config.n_kv_head), head_dim_(config.n_embd / config.n_head) {
    CHECK_LE(config.n_kv_head, config.n_head);
    CHECK_EQ(config.n_head % config.n_kv_head, 0);
    CHECK_EQ(config.n_embd % config.n_head, 0);

    int64_t qkv_dim = (config.n_head + 2 * n_kv_head_) * head_dim_;
    modules_[kCAttnLayerName] = std::make_unique<nn::Linear>(n_embd_, qkv_dim, false);
    modules_[kCProjLayerName] = std::make_unique<nn::Linear>(n_embd_, n_embd_, false);
}

std::vector<std::shared_ptr<Tensor>> CausalSelfAttention::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    const auto B = x[0]->Dims()[0]; // bs
    const auto T = x[0]->Dims()[1]; // seq_len
    const auto C = x[0]->Dims()[2]; // n_embd
    const auto H = n_head_;         // n_head
    const auto D = head_dim_;       // n_embd / n_head

    const auto freqs_cis = x.size() > 1 ? x[1] : nullptr;
    const auto start_pos = x.size() > 2 ? x[2] : nullptr;
    const auto mask = x.size() > 3 ? x[3] : nullptr;

    // (B, T, C) -> (B, T, (H + 2 * n_kv_head) * D)
    auto qkv = modules_[kCAttnLayerName]->Forward({x[0]})[0];
    // NOTE(zbl): torch script uses torch.split({...}, dim) to split tensors into sub-tensors in different sizes
    //            use Slice() to work around here
    int64_t q_size = H * D;
    int64_t kv_size = n_kv_head_ * D;
    // -> Split into q, k, v
    // q: (B, T, H, D)
    auto q = qkv->Slice(2, 0, q_size)->View({B, T, H, D});
    // k: (B, T, n_kv_head, D)
    auto k = qkv->Slice(2, q_size, q_size + kv_size)->View({B, T, n_kv_head_, D});
    // v: (B, T, n_kv_head, D)
    auto v = qkv->Slice(2, q_size + kv_size, q_size + 2 * kv_size)->View({B, T, n_kv_head_, D});

    // -> RoPE on q, k
    // q: (B, T, H, D)
    // k: (B, T, n_kv_head, D)
    std::tie(q, k) = ApplyRotaryEmbedding(q, k, freqs_cis);

    // TODO(zbl): use kv cache during inference
    // if (use_kv_) { ... }

    // align n_head in GQA
    // (B, T, n_kv_head, D) -> (B, T, H, D) via RepeatKV
    k = RepeatKV(k, n_rep_);
    v = RepeatKV(v, n_rep_);

    // (B, T, H, D) -> (B, H, T, D)
    q = q->Transpose(1, 2);
    k = k->Transpose(1, 2);
    v = v->Transpose(1, 2);

    // TODO(zbl): support flash attention later
    // if (flash_) { ... }

    // manual implementation of attention
    // this materializes the large (T,T) matrix for all the queries and keys

    // q: (B, H, T, D)
    // k: (B, H, T, D) -> (B, H, D, T)
    // q @ k.T: (B, H, T, T) -> mul 1.0 / sqrt(D) -> (B, H, T, T)
    auto att = q->Matmul(k->Transpose(-2, -1)) * (1.0 / std::sqrt(*k->Dims().rbegin()));
    if (mask) {
        // mask: (1, 1, T, T)
        att = att->MaskedFill(mask, std::numeric_limits<float>::lowest());
    }
    // (B, H, T, T)
    att = nn::function::Softmax(att, -1);
    // att: (B, H, T, T) @ v: (B, H, T, D) -> y: (B, H, T, D)
    auto y = att->Matmul(v);
    // (B, H, T, D) -> Transpose(1, 2) -> (B, T, H, D) -> (B, H, C)
    y = y->Transpose(1, 2)->Contiguous()->View({B, T, C});
    // output projection
    // (B, H, C) -> Linear(C, C) -> (B, H, C)
    y = modules_[kCProjLayerName]->Forward({y})[0];
    // (B, H, C) == (bs, seq_len, n_embd)
    return {y};
}

MLP::MLP(const LLaMA3Config &config) {
    hidden_dim_ = 4 * config.n_embd;
    hidden_dim_ = int(2 * hidden_dim_ / 3);
    // use custom dim factor multiplier
    if (config.ffn_dim_multiplier.has_value()) {
        hidden_dim_ = int(config.ffn_dim_multiplier.value() * hidden_dim_);
    }
    hidden_dim_ = config.multiple_of * ((hidden_dim_ + config.multiple_of - 1) / config.multiple_of);

    modules_[kCFcLayerName] = std::make_unique<nn::Linear>(config.n_embd, hidden_dim_, false);
    modules_[kCFc2LayerName] = std::make_unique<nn::Linear>(config.n_embd, hidden_dim_, false);
    modules_[kSiluLayerName] = std::make_unique<SwiGLU>();
    modules_[kCProjLayerName] = std::make_unique<nn::Linear>(hidden_dim_, config.n_embd, false);
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

Block::Block(const LLaMA3Config &config) {
    modules_[kLn1LayerName] = std::make_unique<RMSNorm>(config.n_embd, config.norm_eps);
    modules_[kAttnLayerName] = std::make_unique<CausalSelfAttention>(config);
    modules_[kLn2LayerName] = std::make_unique<RMSNorm>(config.n_embd, config.norm_eps);
    modules_[kMlpLayerName] = std::make_unique<MLP>(config);
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
    return {x2, freqs_cis, start_pos, mask};
}

LLaMA3::LLaMA3(const LLaMA3Config &config) : config_(config) {
    {
        std::unordered_map<std::string, std::unique_ptr<nn::Module>> transformer;
        transformer[kWTELayerName] = std::make_unique<nn::Embedding>(config.vocab_size, config.n_embd);
        {
            std::vector<std::unique_ptr<nn::Module>> h;
            for (int64_t i = 0; i < config.n_layer; i++) { h.push_back(std::make_unique<Block>(config)); }
            transformer[kHLayerName] = std::make_unique<nn::Sequential>(std::move(h));
        }
        transformer[kLnFLayerName] = std::make_unique<RMSNorm>(config.n_embd, config.norm_eps);
        modules_[kTransformerLayerName] = std::make_unique<nn::ModuleDict>(std::move(transformer));
    }
    // NOTE(zbl): weight-tying is possible but torch script did not do so
    modules_[kLMHeadLayerName] = std::make_unique<nn::Linear>(config.n_embd, config.vocab_size, false);
}

std::vector<std::shared_ptr<Tensor>> LLaMA3::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    // (bs, seq_len)
    auto &idx = x[0];
    const auto device = idx->GetDevice();
    const auto t = idx->Dims()[1]; // seq_len
    CHECK_LE(t, config_.block_size) << "Cannot forward sequence of length " << t << ", block size is only "
                                    << config_.block_size;

    // Init freqs_cis on device only once
    if (freqs_cis_ == nullptr) {
        freqs_cis_ = PrecomputeFreqsCis(config_.n_embd / config_.n_head, config_.block_size * 2, config_.rope_theta,
                                        config_.use_scaled_rope, device);
        // TODO(zbl): freqs_cis is built in FP32 by default
        if (modules_[kLMHeadLayerName]->parameter(nn::Linear::kParamWeightName)->Dtype() == DataType::kBFLOAT16) {
            freqs_cis_ = std::make_shared<Tensor>(freqs_cis_->To(DataType::kBFLOAT16));
        }
    }

    // forward the LLaMA3 model itself
    auto &transformer = modules_[kTransformerLayerName];
    // (bs, seq_len) -> Embedding(vocab_size, n_embd) -> (bs, seq_len, n_embd)
    auto x1 = transformer->mutable_module(kWTELayerName)->Forward({idx})[0];

    // TODO(zbl): dynamic start_pos
    int64_t start_pos = 0;
    std::shared_ptr<Tensor> freqs_cis = freqs_cis_->Slice(0, start_pos, start_pos + t, 1);

    std::shared_ptr<Tensor> ones = std::make_shared<Tensor>(nn::function::Ones({t, t})->To(idx->GetDevice()));
    std::shared_ptr<Tensor> mask = nn::function::Triu(ones, 1)->View({1, 1, t, t});
    // TODO(zbl): nn::function::Ones builds tensor in FP32 by default
    if (modules_[kLMHeadLayerName]->parameter(nn::Linear::kParamWeightName)->Dtype() == DataType::kBFLOAT16) {
        mask = std::make_shared<Tensor>(mask->To(DataType::kBFLOAT16));
    }
    std::shared_ptr<Tensor> start_pos_ptr = nullptr;

    // (bs, seq_len, n_embd) -> transformer -> (bs, seq_len, n_embd)
    auto x2 = transformer->mutable_module(kHLayerName)
                  ->Forward(std::vector<std::shared_ptr<Tensor>>{x1, freqs_cis, start_pos_ptr, mask})[0];
    // (bs, seq_len, n_embd) -> RMSNorm -> (bs, seq_len, n_embd)
    auto x3 = transformer->mutable_module(kLnFLayerName)->Forward({x2});

    // TODO(zbl): add inference-time mini-optimization
    // (bs, seq_len, n_embd) -> Linear(n_embd, vocab_size) -> (bs, seq_len, vocab_size)
    auto logits = modules_[kLMHeadLayerName]->Forward(x3);

    // (bs, seq_len, vocab_size)
    return logits;
}

std::unique_ptr<LLaMA3> LLaMA3::FromPretrained(ModelType model_type) {
    // TODO(zbl): implement this later
    LOG(FATAL) << "Not implemented yet";
    return nullptr;
}

namespace {
std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

constexpr int32_t kLLaMA3Magic = 20240803;
constexpr int32_t kLLaMA3FP32Version = 3;
} // namespace

std::unique_ptr<LLaMA3> LLaMA3::FromLLMC(const std::string &filepath) {
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

    auto llama3 = std::make_unique<LLaMA3>(LLaMA3Config{.block_size = block_size,
                                                        .vocab_size = vocab_size,
                                                        .n_layer = n_layer,
                                                        .n_head = n_head,
                                                        .n_kv_head = n_kv_head,
                                                        .n_embd = n_embd,
                                                        .ffn_dim_multiplier = ffn_dim_multiplier,
                                                        .multiple_of = multiple_of,
                                                        .rope_theta = rope_theta,
                                                        .use_scaled_rope = static_cast<bool>(use_scaled_rope),
                                                        .norm_eps = norm_eps,
                                                        .max_gen_batch_size = max_gen_bs});

    LOG(INFO) << "Finish build model.";

    auto state_dict = llama3->StateDict();

    LOG(INFO) << "Start reading params.";

    // --- Read model parameters in the saved order ---
    // transformer.wte.weight
    auto &wte
        = state_dict[std::format("{}.{}.{}", kTransformerLayerName, kWTELayerName, nn::Embedding::kParamWeightName)];
    ifs.read(reinterpret_cast<char *>(wte->DataPtr()), wte->SizeInBytes());

    // transformer.h.{i}.ln_1.weight
    for (int i = 0; i < n_layer; ++i) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", kTransformerLayerName, kHLayerName, std::to_string(i),
                                              Block::kLn1LayerName, RMSNorm::kParamWeightName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }

    // transformer.h.{i}.attn.c_attn.weight
    for (int i = 0; i < n_layer; ++i) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", kTransformerLayerName, kHLayerName,
                                              std::to_string(i), Block::kAttnLayerName,
                                              CausalSelfAttention::kCAttnLayerName, nn::Linear::kParamWeightName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }

    // transformer.h.{i}.attn.c_proj.weight
    for (int i = 0; i < n_layer; ++i) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", kTransformerLayerName, kHLayerName,
                                              std::to_string(i), Block::kAttnLayerName,
                                              CausalSelfAttention::kCProjLayerName, nn::Linear::kParamWeightName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }

    // transformer.h.{i}.ln_2.weight
    for (int i = 0; i < n_layer; ++i) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", kTransformerLayerName, kHLayerName, std::to_string(i),
                                              Block::kLn2LayerName, RMSNorm::kParamWeightName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }

    // transformer.h.{i}.mlp.c_fc.weight
    for (int i = 0; i < n_layer; ++i) {
        auto &tensor
            = state_dict[std::format("{}.{}.{}.{}.{}.{}", kTransformerLayerName, kHLayerName, std::to_string(i),
                                     Block::kMlpLayerName, MLP::kCFcLayerName, nn::Linear::kParamWeightName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }

    // transformer.h.{i}.mlp.c_fc2.weight
    for (int i = 0; i < n_layer; ++i) {
        auto &tensor
            = state_dict[std::format("{}.{}.{}.{}.{}.{}", kTransformerLayerName, kHLayerName, std::to_string(i),
                                     Block::kMlpLayerName, MLP::kCFc2LayerName, nn::Linear::kParamWeightName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }

    // transformer.h.{i}.mlp.c_proj.weight
    for (int i = 0; i < n_layer; ++i) {
        auto &tensor
            = state_dict[std::format("{}.{}.{}.{}.{}.{}", kTransformerLayerName, kHLayerName, std::to_string(i),
                                     Block::kMlpLayerName, MLP::kCProjLayerName, nn::Linear::kParamWeightName)];
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    }

    // transformer.ln_f.weight
    auto &ln_f = state_dict[std::format("{}.{}.{}", kTransformerLayerName, kLnFLayerName, RMSNorm::kParamWeightName)];
    ifs.read(reinterpret_cast<char *>(ln_f->DataPtr()), ln_f->SizeInBytes());

    // lm_head.weight
    auto &lm_head = state_dict[std::format("{}.{}", kLMHeadLayerName, nn::Linear::kParamWeightName)];
    ifs.read(reinterpret_cast<char *>(lm_head->DataPtr()), lm_head->SizeInBytes());

    LOG(INFO) << "Finish reading params.";
    return llama3;
}
