#include "infini_train/include/core/transformer/transformer_block.h"

#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/core/transformer/transformer_layer.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {

RMSNorm::RMSNorm(int64_t dim, float eps, infini_train::Device device) : CloneableModule(kType), eps_(eps) {
    parameters_[kParamWeightName]
        = std::make_shared<Tensor>(std::vector<int64_t>{dim}, DataType::kFLOAT32, device)->RequiresGrad();
    nn::init::Ones(parameters_[kParamWeightName]);
}

std::vector<std::shared_ptr<Tensor>> RMSNorm::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    // broadcasted Mul([4, 64, 2048] * [4, 64, 1])
    auto norm = x[0] * nn::function::Rsqrt(nn::function::Mean(nn::function::Pow(x[0], 2), -1, true) + eps_);
    return {norm * parameters_[kParamWeightName]};
}

std::vector<std::shared_ptr<infini_train::Tensor>>
NewGELU::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto &input = x[0];
    return {0.5 * input
            * (1.0 + nn::function::Tanh(std::sqrt(2.0 / M_PI) * (input + 0.044715 * nn::function::Pow(input, 3.0))))};
}

CausalSelfAttention::CausalSelfAttention(const TransformerConfig &config, const ModuleSpec &spec)
    : CloneableModule(kType), config_(config), n_head_(config.n_head), n_embd_(config.n_embd) {
    auto tp_world_size = nn::parallel::global::GetTensorParallelSize();
    CHECK_EQ(config.n_embd % config.n_head, 0);
    CHECK_EQ(n_head_ % tp_world_size, 0) << "n_head must be divisible by TP world size";
    local_n_head_ = n_head_ / tp_world_size;

    // qkv: ColumnParallel (do not gather output) -> each tp_rank gets 3 * (n_embd / tp_world) channels
    modules_[kCAttnLayerName] = build_module(config, spec.submodules_.at(kCAttnLayerName));

    // proj: RowParallel (input is parallel and output is full)
    modules_[kCProjLayerName] = build_module(config, spec.submodules_.at(kCProjLayerName));

    // modules_[kCAttnLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
    //     /*in_features=*/n_embd_,
    //     /*out_features=*/3 * n_embd_,
    //     /*bias=*/true,
    //     /*gather_output=*/false,
    //     /*input_is_parallel=*/false,
    //     /*skip_bias_add=*/false,
    //     /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    // proj: RowParallel (input is parallel and output is full)
    // modules_[kCProjLayerName] = std::make_shared<nn::parallel::RowParallelLinear>(
    //     /*in_features=*/n_embd_,
    //     /*out_features=*/n_embd_,
    //     /*bias=*/true,
    //     /*reduce_output=*/true,
    //     /*input_is_parallel=*/true,
    //     /*skip_bias_add=*/false,
    //     /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    // causal mask: (1, 1, block_size, block_size)
    buffers_[kParamBiasName] = function::Tril(nn::function::Ones({config_.block_size, config_.block_size}))
                                   ->View({1, 1, config_.block_size, config_.block_size});
}

std::vector<std::shared_ptr<infini_train::Tensor>>
CausalSelfAttention::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    auto tp_world_size = parallel::global::GetTensorParallelSize();

    const auto B = x[0]->Dims()[0];                  // bs
    const auto C = x[0]->Dims()[2];                  // n_embd
    const int64_t head_dim = n_embd_ / n_head_;      // per-head dim (global)
    const int64_t local_C = n_embd_ / tp_world_size; // per-rank hidden

    // (B, T, C) -> ColumnParallelLinear(C, 3*C) -> (B, T, 3 * local_C)
    // -> Split -> (3, B, T, local_C)
    auto qkv = (*modules_[kCAttnLayerName])(x)[0]->Split(local_C, 2);

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
    y = (*modules_[kCProjLayerName])({y})[0];
    // (B, T, C) == (bs, seq_len, n_embd)
    return {y};
}

MLP::MLP(const TransformerConfig &config, const ModuleSpec &spec) : CloneableModule(kType) {
    // c_fc: ColumnParallel (input full, output parallel)
    modules_[kCFcLayerName] = build_module(config, spec.submodules_.at(kCFcLayerName));

    modules_[kGeluLayerName] = build_module(config, spec.submodules_.at(kGeluLayerName));

    // c_proj: RowParallel (input parallel, output full)
    modules_[kCProjLayerName] = build_module(config, spec.submodules_.at(kCProjLayerName));

    // // c_fc: ColumnParallel (input full, output parallel)
    // modules_[kCFcLayerName] = std::make_shared<nn::parallel::ColumnParallelLinear>(
    //     /*in_features=*/config.n_embd, /*out_features=*/4 * config.n_embd,
    //     /*bias=*/true,
    //     /*gather_output=*/false,
    //     /*input_is_parallel=*/false,
    //     /*skip_bias_add=*/false,
    //     /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());

    // modules_[kGeluLayerName] = std::make_shared<NewGELU>();

    // // c_proj: RowParallel (input parallel, output full)
    // modules_[kCProjLayerName] = std::make_shared<nn::parallel::RowParallelLinear>(
    //     /*in_features=*/4 * config.n_embd, /*out_features=*/config.n_embd,
    //     /*bias=*/true,
    //     /*reduce_output=*/true,
    //     /*input_is_parallel=*/true,
    //     /*skip_bias_add=*/false,
    //     /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MLP::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // (B, T, C) -> ColumnParallelLinear(C, 4 * C) -> (B, T, 4 * C_local)
    auto x1 = (*modules_[kCFcLayerName])(x);
    // (B, T, 4 * C_local) -> GELU -> (B, T, 4 * C_local)
    auto x2 = (*modules_[kGeluLayerName])(x1);
    // (B, T, 4 * C_local) -> RowParallelLinear(4 * C, C) -> (B, T, C)
    auto x3 = (*modules_[kCProjLayerName])(x2);
    // (B, T, C)
    return x3;
}

TransformerBlock::TransformerBlock(const nn::TransformerConfig &config, const ModuleSpec &spec)
    : CloneableModule(kType) {
    modules_[kLn1LayerName] = build_module(config, spec.submodules_.at(kLn1LayerName));
    modules_[kAttnLayerName] = build_module(config, spec.submodules_.at(kAttnLayerName));
    modules_[kLn2LayerName] = build_module(config, spec.submodules_.at(kLn2LayerName));
    modules_[kMlpLayerName] = build_module(config, spec.submodules_.at(kMlpLayerName));
}

std::vector<std::shared_ptr<infini_train::Tensor>>
TransformerBlock::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> attention -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x1 = x[0] + (*modules_[kAttnLayerName])((*modules_[kLn1LayerName])(x))[0];
    // (bs, seq_len, n_embd) -> Layernorm -> (bs, seq_len, n_embd) -> MLP -> (bs, seq_len, n_embd)
    // -> Add -> (bs, seq_len, n_embd)
    auto x2 = x1 + (*modules_[kMlpLayerName])((*modules_[kLn2LayerName])({x1}))[0];
    // (bs, seq_len, n_embd)
    return {x2};
}

static bool causal_self_attn_registered = []() {
    ModuleRegistry::Instance().Register(
        typeid(CausalSelfAttention),
        [](const TransformerConfig &config, const ModuleSpec &spec) -> std::shared_ptr<Module> {
            return std::make_shared<CausalSelfAttention>(config, spec);
        });
    return true;
}();

static bool ln_registered = []() {
    ModuleRegistry::Instance().Register(
        typeid(LayerNorm), [](const TransformerConfig &config, const ModuleSpec &) -> std::shared_ptr<Module> {
            return std::make_shared<LayerNorm>(std::vector<int64_t>{config.n_embd});
        });
    return true;
}();

static bool mlp_registered = []() {
    ModuleRegistry::Instance().Register(
        typeid(MLP), [](const TransformerConfig &config, const ModuleSpec &spec) -> std::shared_ptr<Module> {
            return std::make_shared<MLP>(config, spec);
        });
    return true;
}();

static bool c_attn_registered = []() {
    ModuleRegistry::Instance().Register(
        typeid(parallel::ColumnParallelLinear),
        [](const TransformerConfig &config, const ModuleSpec &spec) -> std::shared_ptr<Module> {
            int in = any_cast<int>(spec.params_.at("in"));
            int out = any_cast<int>(spec.params_.at("out"));
            return std::make_shared<parallel::ColumnParallelLinear>(
                /*in_features=*/in,
                /*out_features=*/out,
                /*bias=*/true,
                /*gather_output=*/false,
                /*input_is_parallel=*/false,
                /*skip_bias_add=*/false,
                /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());
        });
    return true;
}();

static bool c_proj_registered = []() {
    ModuleRegistry::Instance().Register(
        typeid(parallel::RowParallelLinear),
        [](const TransformerConfig &config, const ModuleSpec &spec) -> std::shared_ptr<Module> {
            int in = any_cast<int>(spec.params_.at("in"));
            int out = any_cast<int>(spec.params_.at("out"));
            return std::make_shared<parallel::RowParallelLinear>(
                /*in_features=*/in,
                /*out_features=*/out,
                /*bias=*/true,
                /*reduce_output=*/true,
                /*input_is_parallel=*/true,
                /*skip_bias_add=*/false,
                /*sequence_parallel=*/nn::parallel::global::GetSequenceParallelEnabled());
        });
    return true;
}();

static bool gelu_registered_registered = []() {
    ModuleRegistry::Instance().Register(
        typeid(NewGELU), [](const TransformerConfig &config, const ModuleSpec &) -> std::shared_ptr<Module> {
            return std::make_shared<NewGELU>();
        });
    return true;
}();
} // namespace infini_train::nn