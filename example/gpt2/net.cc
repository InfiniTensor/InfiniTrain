#include "example/gpt2/net.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/tensor.h"

namespace nn = infini_train::nn;

namespace {
constexpr int kRandomSeed = 42;

class GPT2Linear : public nn::Linear {
public:
    GPT2Linear(int64_t in_features, int64_t out_features, bool residual_scale = false, bool skip_init = false)
        : Linear(in_features, out_features), residual_scale_(residual_scale), skip_init_(skip_init) {}

    bool residual_scale() const { return residual_scale_; }
    bool skip_init() const { return skip_init_; }

private:
    const bool residual_scale_ = false;
    const bool skip_init_ = false;
};

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
    : config_(config), n_head_(config.n_head), n_embd_(config.n_embd) {
    CHECK_EQ(config.n_embd % config.n_head, 0);
    modules_[kCAttnLayerName] = std::make_unique<GPT2Linear>(config.n_embd, config.n_embd * 3);
    modules_[kCProjLayerName] = std::make_unique<GPT2Linear>(config.n_embd, config.n_embd, true, false);
    // TODO(dcj): in torch, here need register buffer for bias
    // (1, 1, block_size, block_size)
    bias_ = nn::function::Tril(nn::function::Ones({config_.block_size, config_.block_size}))
                ->View({1, 1, config_.block_size, config_.block_size});
}

std::vector<std::shared_ptr<infini_train::Tensor>>
CausalSelfAttention::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    const auto B = x[0]->Dims()[0]; // bs
    const auto T = x[0]->Dims()[1]; // seq_len
    const auto C = x[0]->Dims()[2]; // n_embd

    // calculate query, key, values for all heads in batch and move head forward to be the batch dim
    // (bs, seq_len, n_embd) -> Linear(n_embd, 3 * n_embd) -> (bs, seq_len, 3 * n_embd)
    // -> Split -> (3, bs, seq_len, n_embd)
    auto qkv = modules_[kCAttnLayerName]->Forward(x)[0]->Split(n_embd_, 2);
    // (bs, seq_len, n_embd)
    auto q = qkv[0];
    // (bs, seq_len, n_embd)
    auto k = qkv[1];
    // (bs, seq_len, n_embd)
    auto v = qkv[2];
    // (bs, seq_len, n_embd) -> (bs, seq_len, n_head, n_embd / n_head) -> (bs, n_head, seq_len, n_embd / n_head)
    k = k->View({B, T, n_head_, C / n_head_})->Transpose(1, 2);
    // (bs, seq_len, n_embd) -> (bs, seq_len, n_head, n_embd / n_head) -> (bs, n_head, seq_len, n_embd / n_head)
    q = q->View({B, T, n_head_, C / n_head_})->Transpose(1, 2);
    // (bs, seq_len, n_embd) -> (bs, seq_len, n_head, n_embd / n_head) -> (bs, n_head, seq_len, n_embd / n_head)
    v = v->View({B, T, n_head_, C / n_head_})->Transpose(1, 2);

    // TODO(dcj): support flash attention later
    // manual implementation of attention
    // this materializes the large (T,T) matrix for all the queries and keys

    // q: (bs, n_head, seq_len, n_embd / n_head)
    // k: (bs, n_head, seq_len, n_embd / n_head) -> (bs, n_head, n_embd / n_head, seq_len)
    // q matmul k: (bs, n_head, seq_len, seq_len) -> mul 1.0 / sqrt(n_embd / n_head) -> (bs, n_head, seq_len, seq_len)
    auto att = q->Matmul(k->Transpose(-2, -1)) * (1.0 / std::sqrt(*k->Dims().rbegin()));
    // (1, 1, seq_len, seq_len)
    auto mask = bias_->Slice({0, 0, 0, 0}, {1, 1, T, T}, {1, 1, 1, 1});
    // (1, 1, seq_len, seq_len) -> eq 0 -> (1, 1, seq_len, seq_len) -> masked_fill -> (bs, n_head, seq_len, seq_len)
    att = att->MaskedFill(mask == 0, -std::numeric_limits<float>::infinity());
    // (bs, n_head, seq_len, seq_len)
    att = nn::function::Softmax(att, -1);
    // (bs, n_head, seq_len, n_embd / n_head)
    auto y = att->Matmul(v);
    // (bs, n_head, seq_len, n_embd / n_head) -> Transpose(1, 2) -> (bs, seq_len, n_head, n_embd / n_head)
    // -> (bs, seq_len, n_embd)
    y = y->Transpose(1, 2)->Contiguous()->View({B, T, C});
    // output projection
    // (bs, seq_len, n_embd) -> Linear(n_embd, n_embd) -> (bs, seq_len, n_embd)
    y = modules_[kCProjLayerName]->Forward({y})[0];
    // (bs, seq_len, n_embd)
    return {y};
}

MLP::MLP(const GPT2Config &config) {
    modules_[kCFclayerName] = std::make_unique<GPT2Linear>(config.n_embd, config.n_embd * 4);
    modules_[kGeluLayerName] = std::make_unique<NewGELU>();
    modules_[kCProjLayerName] = std::make_unique<GPT2Linear>(config.n_embd * 4, config.n_embd, true, false);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MLP::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // (bs, seq_len, n_embd) -> Linear(n_embd, 4 * n_embd) -> (bs, seq_len, 4 * n_embd)
    auto x1 = modules_[kCFclayerName]->Forward(x);
    // (bs, seq_len, 4 * n_embd) -> GELU -> (bs, seq_len, 4 * n_embd)
    auto x2 = modules_[kGeluLayerName]->Forward(x1);
    // (bs, seq_len, 4 * n_embd) -> Linear(4 * n_embd, n_embd) -> (bs, seq_len, n_embd)
    auto x3 = modules_[kCProjLayerName]->Forward(x2);
    // (bs, seq_len, n_embd)
    return x3;
}

Block::Block(const GPT2Config &config) {
    modules_[kLn1LayerName] = std::make_unique<nn::LayerNorm>(std::vector<int64_t>{config.n_embd});
    modules_[kAttnLayerName] = std::make_unique<CausalSelfAttention>(config);
    modules_[kLn2LayerName] = std::make_unique<nn::LayerNorm>(std::vector<int64_t>{config.n_embd});
    modules_[kMlpLayerName] = std::make_unique<MLP>(config);
}

std::vector<std::shared_ptr<infini_train::Tensor>>
Block::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> attention -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x1 = x[0] + modules_[kAttnLayerName]->Forward(modules_[kLn1LayerName]->Forward(x))[0];
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> MLP -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x2 = x1 + modules_[kMlpLayerName]->Forward(modules_[kLn2LayerName]->Forward({x1}))[0];
    // (bs, seq_len, n_embd)
    return {x2};
}

GPT2::GPT2(const GPT2Config &config) : config_(config) {
    {
        std::unordered_map<std::string, std::unique_ptr<nn::Module>> transformer;
        transformer[kWTELayerName] = std::make_unique<nn::Embedding>(config.vocab_size, config.n_embd);
        transformer[kWPELayerName] = std::make_unique<nn::Embedding>(config.block_size, config.n_embd);
        {
            std::vector<std::unique_ptr<nn::Module>> h;
            for (int64_t i = 0; i < config.n_layer; i++) { h.push_back(std::make_unique<Block>(config)); }
            transformer[kHLayerName] = std::make_unique<nn::Sequential>(std::move(h));
        }
        transformer[kLnFLayerName] = std::make_unique<nn::LayerNorm>(std::vector<int64_t>{config.n_embd});
        modules_[kTransformerLayerName] = std::make_unique<nn::ModuleDict>(std::move(transformer));
    }
    // don't init this one, we will tie weights
    modules_[kLMHeadLayerName] = std::make_unique<GPT2Linear>(config.n_embd, config.vocab_size, false, true);
    // https://paperswithcode.com/method/weight-tying
    mutable_module(kTransformerLayerName)
        ->mutable_module(kWTELayerName)
        ->mutable_parameter(GPT2Linear::kParamWeightName)
        ->reset(module(kLMHeadLayerName).parameter(GPT2Linear::kParamWeightName).get());

    // init all weights
    Apply([&](Module *module) {
        if (module->type() == nn::Linear::kType) {
            auto *linear = static_cast<GPT2Linear *>(module);
            const float std = linear->residual_scale() ? 0.02 / std::sqrt(2 * config_.n_layer) : 0.02;
            if (!linear->skip_init()) {
                nn::init::Normal(*linear->mutable_parameter(GPT2Linear::kParamWeightName), 0.0f, std, gen);
            }
            if (linear->has_parameter(nn::Linear::kParamWeightName)) {
                nn::init::Zeros(*linear->mutable_parameter(nn::Linear::kParamBiasName));
            }
        } else if (module->type() == nn::Embedding::kType) {
            nn::init::Normal(*module->mutable_parameter(nn::Embedding::kParamWeightName), 0.0f, 0.02, gen);
        }
    });
}

std::vector<std::shared_ptr<infini_train::Tensor>>
GPT2::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // (bs, seq_len)
    auto &idx = x[0];
    const auto device = idx->GetDevice();
    const auto t = idx->Dims()[1]; // seq_len
    CHECK_LE(t, config_.block_size) << "Cannot forward sequence of length " << t << ", block size is only "
                                    << config_.block_size;
    // (seq_len)
    auto pos = nn::init::Arange(0, t, infini_train::DataType::kINT64, device);

    // forward the GPT2 model itself
    auto &transformer = modules_[kTransformerLayerName];
    // (bs, seq_len) -> Embedding(vocab_size, n_embd) -> (bs, seq_len, n_embd)
    auto tok_emb = transformer->mutable_module(kWTELayerName)->Forward({idx})[0];
    // (seq_len) -> Embedding(block_size, n_embd) -> (seq_len, n_embd)
    auto pos_emb = transformer->mutable_module(kWPELayerName)->Forward({pos})[0];
    // (bs, seq_len, n_embd)
    auto x1 = tok_emb + pos_emb;

    // (bs, seq_len, n_embd) -> transformer -> (bs, seq_len, n_embd)
    auto x2 = transformer->mutable_module(kHLayerName)->Forward({x1});
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd)
    auto x3 = transformer->mutable_module(kLnFLayerName)->Forward(x2);

    // TODO(dcj): add inference-time mini-optimization
    // (bs, seq_len, n_embd) -> Linear(n_embd, vocab_size) -> (bs, seq_len, vocab_size)
    auto logits = modules_[kLMHeadLayerName]->Forward(x3);

    // (bs, seq_len, vocab_size)
    return logits;
}

std::unique_ptr<GPT2> GPT2::FromPretrained(ModelType model_type) {
    // TODO(dcj): implement this later
    return nullptr;
}
