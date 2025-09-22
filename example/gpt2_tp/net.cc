#include "example/gpt2_tp/net.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>

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

namespace nn = infini_train::nn;
namespace tp = infini_train::nn::parallel;

namespace {
constexpr int kRandomSeed = 42;

// TODO(dcj): make this rng generator compatible with torch later
static std::mt19937 gen{kRandomSeed};

inline int64_t PaddedVocab(int64_t v, int64_t tp_size, int64_t align = 128) {
    int64_t m = tp_size * align;
    return ((v + m - 1) / m) * m;
}
} // namespace

std::vector<std::shared_ptr<infini_train::Tensor>>
NewGELU::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto &input = x[0];
    return {0.5 * input
            * (1.0 + nn::function::Tanh(std::sqrt(2.0 / M_PI) * (input + 0.044715 * nn::function::Pow(input, 3.0))))};
}

TPCausalSelfAttention::TPCausalSelfAttention(const TPGPT2Config &config)
    : config_(config), n_head_(config.n_head), n_embd_(config.n_embd) {
    CHECK_EQ(config.n_embd % config.n_head, 0);
    CHECK_EQ(n_head_ % config.tp_group.WorldSize(), 0) << "n_head must be divisible by TP world size";
    local_n_head_ = n_head_ / config.tp_group.WorldSize();

    // qkv: ColumnParallel (do not gather output) -> each rank gets 3 * (n_embd / tp_world) channels
    modules_[kCAttnLayerName] = std::make_shared<tp::ColumnParallelLinear>(
        /*in_features=*/n_embd_,
        /*out_features=*/3 * n_embd_,
        /*bias=*/true, config_.tp_group,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/config_.tp_group.sequence_parallel_enabled);

    // proj: RowParallel (input is parallel and output is full)
    modules_[kCProjLayerName] = std::make_shared<tp::RowParallelLinear>(
        /*in_features=*/n_embd_,
        /*out_features=*/n_embd_,
        /*bias=*/true, config_.tp_group,
        /*reduce_output=*/true,
        /*input_is_parallel=*/true,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/config_.tp_group.sequence_parallel_enabled);

    // causal mask: (1, 1, block_size, block_size)
    buffers_[kParamBiasName] = nn::function::Tril(nn::function::Ones({config_.block_size, config_.block_size}))
                                   ->View({1, 1, config_.block_size, config_.block_size});
}

std::vector<std::shared_ptr<infini_train::Tensor>>
TPCausalSelfAttention::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    const auto B = x[0]->Dims()[0];                                 // bs
    const auto C = x[0]->Dims()[2];                                 // n_embd
    const int64_t head_dim = n_embd_ / n_head_;                     // per-head dim (global)
    const int64_t local_C = n_embd_ / config_.tp_group.WorldSize(); // per-rank hidden

    // (B, T, C) -> ColumnParallelLinear(C, 3*C) -> (B, T, 3 * local_C)
    // -> Split -> (3, B, T, local_C)
    auto qkv = modules_[kCAttnLayerName]->Forward(x)[0]->Split(local_C, 2);

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
    y = modules_[kCProjLayerName]->Forward({y})[0];
    // (B, T, C) == (bs, seq_len, n_embd)
    return {y};
}

TPMLP::TPMLP(const TPGPT2Config &config) {
    // c_fc: ColumnParallel (input full, output parallel)
    modules_[kCFcLayerName] = std::make_shared<tp::ColumnParallelLinear>(
        /*in_features=*/config.n_embd, /*out_features=*/4 * config.n_embd,
        /*bias=*/true, config.tp_group,
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/config.tp_group.sequence_parallel_enabled);

    modules_[kGeluLayerName] = std::make_shared<NewGELU>();

    // c_proj: RowParallel (input parallel, output full)
    modules_[kCProjLayerName] = std::make_shared<tp::RowParallelLinear>(
        /*in_features=*/4 * config.n_embd, /*out_features=*/config.n_embd,
        /*bias=*/true, config.tp_group,
        /*reduce_output=*/true,
        /*input_is_parallel=*/true,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/config.tp_group.sequence_parallel_enabled);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
TPMLP::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // (B, T, C) -> ColumnParallelLinear(C, 4 * C) -> (B, T, 4 * C_local)
    auto x1 = modules_[kCFcLayerName]->Forward(x);
    // (B, T, 4 * C_local) -> GELU -> (B, T, 4 * C_local)
    auto x2 = modules_[kGeluLayerName]->Forward(x1);
    // (B, T, 4 * C_local) -> RowParallelLinear(4 * C, C) -> (B, T, C)
    auto x3 = modules_[kCProjLayerName]->Forward(x2);
    // (B, T, C)
    return x3;
}

TPBlock::TPBlock(const TPGPT2Config &config) {
    modules_[kLn1LayerName] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.n_embd});
    modules_[kAttnLayerName] = std::make_shared<TPCausalSelfAttention>(config);
    modules_[kLn2LayerName] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config.n_embd});
    modules_[kMlpLayerName] = std::make_shared<TPMLP>(config);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
TPBlock::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> attention -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x1 = x[0] + modules_[kAttnLayerName]->Forward(modules_[kLn1LayerName]->Forward(x))[0];
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> MLP -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x2 = x1 + modules_[kMlpLayerName]->Forward(modules_[kLn2LayerName]->Forward({x1}))[0];
    // (bs, seq_len, n_embd)
    return {x2};
}

TensorParallelGPT2::TensorParallelGPT2(const TPGPT2Config &config) : config_(config) {
    // NOTE(zbl): VocabParallelEmbedding requires vocab_size % tp_size == 0
    //            Megatron-LM has an optional argument `--make-vocab-size-divisible-by`, would do padding to vocab
    //            Here we introduce padding by default, might need modify Tokenizer correspondingly later
    CHECK_EQ(config.vocab_size % config.tp_group.WorldSize(), 0) << "Vocab size should be divisible by TP world size";
    {
        std::unordered_map<std::string, std::shared_ptr<nn::Module>> transformer;
        transformer[kWTELayerName] = std::make_shared<tp::VocabParallelEmbedding>(
            config_.vocab_size, config_.n_embd, config_.tp_group.sequence_parallel_enabled, config_.tp_group);
        transformer[kWPELayerName] = std::make_shared<nn::Embedding>(config_.block_size, config_.n_embd);
        {
            std::vector<std::shared_ptr<nn::Module>> h;
            for (int64_t i = 0; i < config_.n_layer; i++) { h.push_back(std::make_shared<TPBlock>(config_)); }
            transformer[kHLayerName] = std::make_shared<nn::Sequential>(std::move(h));
        }
        transformer[kLnFLayerName] = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{config_.n_embd});
        modules_[kTransformerLayerName] = std::make_shared<nn::ModuleDict>(std::move(transformer));
    }
    // don't init this one, we will tie weights
    modules_[kLMHeadLayerName] = std::make_shared<tp::ColumnParallelLinear>(
        /*in_features=*/config_.n_embd, /*out_features=*/config_.vocab_size,
        /*bias=*/false, config_.tp_group,
        // NOTE(zbl): each rank would get sharded [B, T, V_local] as logits
        /*gather_output=*/false,
        /*input_is_parallel=*/false,
        /*skip_bias_add=*/false,
        /*sequence_parallel=*/config.tp_group.sequence_parallel_enabled);

    // https://paperswithcode.com/method/weight-tying
    *mutable_module(kTransformerLayerName)
         ->mutable_module(kWTELayerName)
         ->mutable_parameter(tp::VocabParallelEmbedding::kParamWeightName)
        = module(kLMHeadLayerName).parameter(tp::ColumnParallelLinear::kParamWeightName);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
TensorParallelGPT2::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // (B, T)
    auto &idx = x[0];
    const auto device = idx->GetDevice();
    CHECK(config_.tp_group.Rank() == config_.tp_group.RankOf(device)) << "Input is not on the same device as model";

    const auto t = idx->Dims()[1]; // T
    CHECK_LE(t, config_.block_size) << "Cannot forward sequence of length " << t << ", block size is only "
                                    << config_.block_size;
    // (T_local)
    // NOTE(zbl): Slice pos sequence when SP is enabled
    int64_t t_local = config_.tp_group.sequence_parallel_enabled ? (t / config_.tp_group.WorldSize()) : t;
    int64_t start = config_.tp_group.sequence_parallel_enabled ? config_.tp_group.Rank() * t_local : 0;
    auto pos = nn::init::Arange(start, start + t_local, infini_train::DataType::kINT64, device);
    // forward the GPT2 model itself
    auto &transformer = modules_[kTransformerLayerName];
    // (B, T) -> Embedding(V_local, C) -> (B, T, C)
    auto tok_emb = transformer->mutable_module(kWTELayerName)->Forward({idx})[0];
    // (T) -> Embedding(T_max, C) -> (T, C)
    auto pos_emb = transformer->mutable_module(kWPELayerName)->Forward({pos})[0];
    // (B, T, C)
    auto x1 = tok_emb + pos_emb;

    // (B, T, C) -> transformer -> (B, T, C)
    auto x2 = transformer->mutable_module(kHLayerName)->Forward({x1});
    // (B, T, C) -> Layernorm -> (B, T, C)
    auto x3 = transformer->mutable_module(kLnFLayerName)->Forward(x2);

    // TODO(dcj): add inference-time mini-optimization
    // (B, T, C) -> Linear(C, V) -> (B, T, V)
    auto logits = modules_[kLMHeadLayerName]->Forward(x3)[0];

    // (B, T, V_original)
    return {logits};
}

std::shared_ptr<TensorParallelGPT2> TensorParallelGPT2::FromPretrained(ModelType model_type) {
    // TODO(dcj): implement this later
    LOG(FATAL) << "Not implemented yet";
    return nullptr;
}

namespace {

constexpr int32_t kHeaderMagic = 20240326;
constexpr int32_t kHeaderFP32Version = 3;
constexpr int32_t kHeaderBF16Version = 5;

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

inline void ReadMatrixAllFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols) {
    const size_t bytes = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
    ifs.read(reinterpret_cast<char *>(dst), bytes);
}

// Shard Reader Functions
//
// Read Row Shard: [row_start : row_start+row_cnt) × [0:cols]
inline void ReadMatrixRowShardFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols, int64_t row_start,
                                    int64_t row_cnt) {
    std::streampos base = ifs.tellg();
    const size_t row_bytes = static_cast<size_t>(cols) * sizeof(float);
    ifs.seekg(base + std::streamoff(row_start * row_bytes));
    // assume row-major
    ifs.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(row_cnt * row_bytes));
    ifs.seekg(base + std::streamoff(rows * row_bytes));
}

// Read Column Shard: [0:rows) × [col_start : col_start+col_cnt)
inline void ReadMatrixColShardFloat(std::ifstream &ifs, float *dst, int64_t rows, int64_t cols, int64_t col_start,
                                    int64_t col_cnt) {
    std::streampos base = ifs.tellg();
    const size_t row_bytes = static_cast<size_t>(cols) * sizeof(float);
    const size_t pick_bytes = static_cast<size_t>(col_cnt) * sizeof(float);
    // assume row-major, need loop
    for (int64_t r = 0; r < rows; ++r) {
        ifs.seekg(base + std::streamoff(r * row_bytes + col_start * sizeof(float)));
        ifs.read(reinterpret_cast<char *>(dst + r * col_cnt), static_cast<std::streamsize>(pick_bytes));
    }
    ifs.seekg(base + std::streamoff(rows * row_bytes));
}

// Read Whole Array
inline void ReadVectorAllFloat(std::ifstream &ifs, float *dst, int64_t len) {
    ifs.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(len * sizeof(float)));
}
// Read Array Shard: [start : start+cnt)
inline void ReadVectorShardFloat(std::ifstream &ifs, float *dst, int64_t len, int64_t start, int64_t cnt) {
    std::streampos base = ifs.tellg();
    ifs.seekg(base + std::streamoff(start * sizeof(float)));
    ifs.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(cnt * sizeof(float)));
    ifs.seekg(base + std::streamoff(len * sizeof(float)));
}

} // namespace

std::shared_ptr<TensorParallelGPT2> TensorParallelGPT2::FromLLMC(const std::string &filepath,
                                                                 tp::TensorParallelGroup tp_group) {
    if (!std::filesystem::exists(filepath)) {
        LOG(FATAL) << "File not found: " << filepath;
    }

    std::ifstream ifs(filepath, std::ios::binary);
    const auto header = ReadSeveralBytesFromIfstream(256 * sizeof(int32_t), &ifs);

    const auto magic = BytesToType<uint32_t>(header, 0);
    CHECK_EQ(magic, kHeaderMagic);
    auto [version, dtype] = DetermineAndCheckVersion(header, 4);
    CHECK_EQ(version, kHeaderFP32Version);

    const auto block_size = BytesToType<uint32_t>(header, 8);
    const auto vocab_size = BytesToType<uint32_t>(header, 12);
    const auto n_layer = BytesToType<uint32_t>(header, 16);
    const auto n_head = BytesToType<uint32_t>(header, 20);
    const auto n_embd = BytesToType<uint32_t>(header, 24);
    const auto padded_vocab_size = BytesToType<uint32_t>(header, 28);
    auto local_gpt2 = std::make_shared<TensorParallelGPT2>(TPGPT2Config{.block_size = block_size,
                                                                        // NOTE(zbl): use padded vocab size
                                                                        .vocab_size = padded_vocab_size,
                                                                        .original_vocab_size = vocab_size,
                                                                        .n_layer = n_layer,
                                                                        .n_head = n_head,
                                                                        .n_embd = n_embd,
                                                                        .tp_group = tp_group});

    LOG(ERROR) << "magic: " << magic << " version: " << version << " block_size: " << block_size
               << " vocab_size: " << vocab_size << " n_layer: " << n_layer << " n_head: " << n_head
               << " n_embd: " << n_embd << " padded_vocab_size: " << padded_vocab_size;

    auto world_size = tp_group.WorldSize();
    CHECK_EQ(n_embd % world_size, 0) << "n_embd must be divisible by TP world size.";
    CHECK_EQ(n_embd % n_head, 0) << "n_embd must be divisible by n_head.";
    CHECK_EQ(n_head % world_size, 0) << "n_head must be divisible by TP world size.";

    auto rank = tp_group.Rank();
    // calculate xx_size_per_partition
    const int64_t vpp = padded_vocab_size / world_size;
    const int64_t v_start = static_cast<int64_t>(rank) * vpp;
    const int64_t v_end = v_start + vpp;

    const int64_t qkv_out = 3 * n_embd;
    const int64_t qkv_pp = qkv_out / world_size;
    const int64_t qkv_start = static_cast<int64_t>(rank) * qkv_pp;

    const int64_t fc_out = 4 * n_embd;
    const int64_t fc_pp = fc_out / world_size;
    const int64_t fc_start = static_cast<int64_t>(rank) * fc_pp;

    const int64_t in_pp = n_embd / world_size;        // for c_proj (row-parallel, shard on input)
    const int64_t in4_pp = (4 * n_embd) / world_size; // for mlp.c_proj (input shard)

    auto state_dict = local_gpt2->StateDict();

    // transformer.wte.weight (also transformer.lm_head.weight)
    // full: (padded_vocab_size, n_embd)
    // local: (vocab_size_per_partition, n_embd)
    auto &transformer_wte_weight
        = state_dict[std::format("{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                 TensorParallelGPT2::kWTELayerName, tp::VocabParallelEmbedding::kParamWeightName)];
    ReadMatrixRowShardFloat(ifs, static_cast<float *>(transformer_wte_weight->DataPtr()), padded_vocab_size, n_embd,
                            v_start, vpp);
    // transformer.wpe.weight
    auto &transformer_wpe_weight
        = state_dict[std::format("{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                 TensorParallelGPT2::kWPELayerName, nn::Embedding::kParamWeightName)];
    ReadMatrixAllFloat(ifs, static_cast<float *>(transformer_wpe_weight->DataPtr()), block_size, n_embd);
    // transformer.h.{i}.ln_1.weight
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                              TensorParallelGPT2::kHLayerName, std::to_string(idx),
                                              TPBlock::kLn1LayerName, nn::LayerNorm::kParamWeightName)];
        ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
    }
    // transformer.h.{i}.ln_1.bias
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                              TensorParallelGPT2::kHLayerName, std::to_string(idx),
                                              TPBlock::kLn1LayerName, nn::LayerNorm::kParamBiasName)];
        ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
    }
    // transformer.h.{i}.attn.c_attn.weight (ColumnParallelLinear, but actually applies on "rows")
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                              TensorParallelGPT2::kHLayerName, std::to_string(idx),
                                              TPBlock::kAttnLayerName, TPCausalSelfAttention::kCAttnLayerName,
                                              tp::ColumnParallelLinear::kParamWeightName)];
        // NOTE(zbl): In the .bin model file, Q/K/V is concated along last dim,
        //            i.e. [Q|K|V].T = [q1|q2|...|qn|k1|k2|...|kn|v1|v2|...|vn].T
        //            However, each rank needs to get [q_i|k_i|v_i].T, so we need to jump and read them respectively
        float *dst = static_cast<float *>(tensor->DataPtr());
        const int64_t local_C = n_embd / world_size;
        const int64_t rows_all = 3 * n_embd;
        const int64_t cols_all = n_embd;
        const std::streampos base_pos = ifs.tellg();
        // Read q_i -> write to dst rows of [0 : local_C)
        ifs.seekg(base_pos);
        ReadMatrixRowShardFloat(ifs,
                                /*dst=*/dst + (0 * local_C) * cols_all,
                                /*rows=*/rows_all, /*cols=*/cols_all,
                                /*row_start=*/rank * local_C, /*row_cnt=*/local_C);
        // Read k_i -> write to dst rows of [local_C : 2*local_C)
        ifs.seekg(base_pos);
        ReadMatrixRowShardFloat(ifs,
                                /*dst=*/dst + (1 * local_C) * cols_all,
                                /*rows=*/rows_all, /*cols=*/cols_all,
                                /*row_start=*/n_embd + rank * local_C, /*row_cnt=*/local_C);
        // Read v_i -> write to dst rows of [2*local_C : 3*local_C) 行
        ifs.seekg(base_pos);
        ReadMatrixRowShardFloat(ifs,
                                /*dst=*/dst + (2 * local_C) * cols_all,
                                /*rows=*/rows_all, /*cols=*/cols_all,
                                /*row_start=*/2 * n_embd + rank * local_C, /*row_cnt=*/local_C);
    }
    // transformer.h.{i}.attn.c_attn.bias (ColumnParallelLinear)
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor
            = state_dict[std::format("{}.{}.{}.{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                     TensorParallelGPT2::kHLayerName, std::to_string(idx), TPBlock::kAttnLayerName,
                                     TPCausalSelfAttention::kCAttnLayerName, tp::ColumnParallelLinear::kParamBiasName)];
        // NOTE(zbl): Same as c_attn.weight, the bias for Q/K/V is concated
        //            i.e. [Q|K|V] = [q1|q2|...|qn|k1|k2|...|kn|v1|v2|...|vn]
        //            However, each rank needs to get [q_i|k_i|v_i], so we need to jump and read them respectively
        float *dst = static_cast<float *>(tensor->DataPtr());
        const int64_t local_C = n_embd / world_size;
        const int64_t len_all = 3 * n_embd;
        const std::streampos base_pos = ifs.tellg();
        // Read q_i
        ifs.seekg(base_pos);
        ReadVectorShardFloat(ifs,
                             /*dst=*/dst + (0 * local_C),
                             /*len=*/len_all,
                             /*start=*/rank * local_C, /*cnt=*/local_C);
        // Read k_i
        ifs.seekg(base_pos);
        ReadVectorShardFloat(ifs,
                             /*dst=*/dst + (1 * local_C),
                             /*len=*/len_all,
                             /*start=*/n_embd + rank * local_C, /*cnt=*/local_C);
        // Read v_i
        ifs.seekg(base_pos);
        ReadVectorShardFloat(ifs,
                             /*dst=*/dst + (2 * local_C),
                             /*len=*/len_all,
                             /*start=*/2 * n_embd + rank * local_C, /*cnt=*/local_C);
    }
    // transformer.h.{i}.attn.c_proj.weight (RowParallelLinear, but actually applies on "columns")
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor
            = state_dict[std::format("{}.{}.{}.{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                     TensorParallelGPT2::kHLayerName, std::to_string(idx), TPBlock::kAttnLayerName,
                                     TPCausalSelfAttention::kCProjLayerName, tp::RowParallelLinear::kParamWeightName)];
        ReadMatrixColShardFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd, n_embd, rank * in_pp, in_pp);
    }
    // transformer.h.{i}.attn.c_proj.bias (RowParallelLinear, no shard on bias)
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor
            = state_dict[std::format("{}.{}.{}.{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                     TensorParallelGPT2::kHLayerName, std::to_string(idx), TPBlock::kAttnLayerName,
                                     TPCausalSelfAttention::kCProjLayerName, tp::RowParallelLinear::kParamBiasName)];
        ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
    }
    // transformer.h.{i}.ln_2.weight
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                              TensorParallelGPT2::kHLayerName, std::to_string(idx),
                                              TPBlock::kLn2LayerName, nn::LayerNorm::kParamWeightName)];
        ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
    }
    // transformer.h.{i}.ln_2.bias
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor = state_dict[std::format("{}.{}.{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                              TensorParallelGPT2::kHLayerName, std::to_string(idx),
                                              TPBlock::kLn2LayerName, nn::LayerNorm::kParamBiasName)];
        ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
    }
    // transformer.h.{i}.mlp.c_fc.weight (ColumnParallelLinear, but actually applies on "rows")
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor
            = state_dict[std::format("{}.{}.{}.{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                     TensorParallelGPT2::kHLayerName, std::to_string(idx), TPBlock::kMlpLayerName,
                                     TPMLP::kCFcLayerName, tp::ColumnParallelLinear::kParamWeightName)];
        ReadMatrixRowShardFloat(ifs, static_cast<float *>(tensor->DataPtr()), fc_out, n_embd, fc_start, fc_pp);
    }
    // transformer.h.{i}.mlp.c_fc.bias (ColumnParallelLinear)
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor
            = state_dict[std::format("{}.{}.{}.{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                     TensorParallelGPT2::kHLayerName, std::to_string(idx), TPBlock::kMlpLayerName,
                                     TPMLP::kCFcLayerName, tp::ColumnParallelLinear::kParamBiasName)];
        ReadVectorShardFloat(ifs, static_cast<float *>(tensor->DataPtr()), fc_out, fc_start, fc_pp);
    }
    // transformer.h.{i}.mlp.c_proj.weight (RowParallelLinear, but actually applies on "columns")
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor
            = state_dict[std::format("{}.{}.{}.{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                     TensorParallelGPT2::kHLayerName, std::to_string(idx), TPBlock::kMlpLayerName,
                                     TPMLP::kCProjLayerName, tp::RowParallelLinear::kParamWeightName)];
        ReadMatrixColShardFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd, fc_out, rank * in4_pp, in4_pp);
    }
    // transformer.h.{i}.mlp.c_proj.bias (RowParallelLinear, no shard on bias)
    for (int idx = 0; idx < n_layer; idx++) {
        auto &tensor
            = state_dict[std::format("{}.{}.{}.{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                     TensorParallelGPT2::kHLayerName, std::to_string(idx), TPBlock::kMlpLayerName,
                                     TPMLP::kCProjLayerName, tp::RowParallelLinear::kParamBiasName)];
        ReadVectorAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), n_embd);
    }
    // transformer.ln_f.weight
    auto &transformer_ln_f_weight
        = state_dict[std::format("{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                 TensorParallelGPT2::kLnFLayerName, nn::LayerNorm::kParamWeightName)];
    ReadVectorAllFloat(ifs, static_cast<float *>(transformer_ln_f_weight->DataPtr()), n_embd);
    // transformer.ln_f.bias
    auto &transformer_ln_f_bias
        = state_dict[std::format("{}.{}.{}", TensorParallelGPT2::kTransformerLayerName,
                                 TensorParallelGPT2::kLnFLayerName, nn::LayerNorm::kParamBiasName)];
    ReadVectorAllFloat(ifs, static_cast<float *>(transformer_ln_f_bias->DataPtr()), n_embd);

    return local_gpt2;
}
