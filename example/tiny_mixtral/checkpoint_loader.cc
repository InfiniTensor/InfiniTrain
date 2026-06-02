#include "example/tiny_mixtral/checkpoint_loader.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/tensor.h"

#include "example/common/utils.h"
#include "example/tiny_mixtral/config.h"

namespace nn = infini_train::nn;

namespace {

constexpr int32_t kTinyMixtralLLMCMagic = 20260513;
constexpr int32_t kTinyMixtralLLMCVersion = 2;
constexpr int64_t kLLMCHeaderEntries = 256;

} // namespace

namespace tiny_mixtral {

namespace {

template <typename T>
void CompareCheckpointValue(const std::string &name, const T &checkpoint_value, const T &runtime_value) {
    CHECK_EQ(checkpoint_value, runtime_value) << name << " value from checkpoint (" << checkpoint_value
                                              << ") is not equal to runtime config value (" << runtime_value << ")";
}

} // namespace

nn::TransformerConfig ConfigFromLLMC(const std::string &filepath) {
    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs) << "Failed to open tiny Mixtral LLMC file: " << filepath;
    const auto header = infini_train::ReadSeveralBytesFromIfstream(kLLMCHeaderEntries * sizeof(int32_t), &ifs);
    CHECK(ifs) << "Failed to read tiny Mixtral LLMC header: " << filepath;
    CHECK_EQ(infini_train::BytesToType<int32_t>(header, 0 * sizeof(int32_t)), kTinyMixtralLLMCMagic);
    CHECK_EQ(infini_train::BytesToType<int32_t>(header, 1 * sizeof(int32_t)), kTinyMixtralLLMCVersion);

    auto config = TinyMixtralConfig();
    config.block_size = infini_train::BytesToType<int32_t>(header, 2 * sizeof(int32_t));
    config.vocab_size = infini_train::BytesToType<int32_t>(header, 3 * sizeof(int32_t));
    config.original_vocab_size = config.vocab_size;
    config.n_layer = infini_train::BytesToType<int32_t>(header, 4 * sizeof(int32_t));
    config.n_head = infini_train::BytesToType<int32_t>(header, 5 * sizeof(int32_t));
    config.n_kv_head = infini_train::BytesToType<int32_t>(header, 6 * sizeof(int32_t));
    config.n_embd = infini_train::BytesToType<int32_t>(header, 7 * sizeof(int32_t));
    config.ffn_expansion_ratio = infini_train::BytesToType<float>(header, 9 * sizeof(int32_t));
    // Header slots 10 and 11 store dense-MLP helpers; MoE expert size is stored in moe_ffn_hidden_size.
    config.norm_eps = infini_train::BytesToType<float>(header, 12 * sizeof(int32_t));
    config.rope_theta = infini_train::BytesToType<float>(header, 13 * sizeof(int32_t));
    config.use_scaled_rope = infini_train::BytesToType<int32_t>(header, 14 * sizeof(int32_t)) != 0;

    nn::MoEConfig moe_config;
    moe_config.num_experts = infini_train::BytesToType<int32_t>(header, 8 * sizeof(int32_t));
    moe_config.expert_parallel_size = 1;
    moe_config.router_topk = infini_train::BytesToType<int32_t>(header, 15 * sizeof(int32_t));
    moe_config.moe_ffn_hidden_size = infini_train::BytesToType<int32_t>(header, 16 * sizeof(int32_t));
    moe_config.token_dispatcher_type = nn::MoEConfig::TokenDispatcherType::kAllGather;
    moe_config.expert_impl = nn::MoEConfig::ExpertImpl::kSequential;
    config.moe_config = moe_config;
    SanitizeTinyMixtralConfig(config);
    return config;
}

void CheckLLMCConfig(const std::string &filepath, const nn::TransformerConfig &expected_config) {
    SanitizeTinyMixtralConfig(expected_config);
    const auto checkpoint_config = ConfigFromLLMC(filepath);
    CompareCheckpointValue("block_size", checkpoint_config.block_size, expected_config.block_size);
    CompareCheckpointValue("vocab_size", checkpoint_config.vocab_size, expected_config.vocab_size);
    CompareCheckpointValue("original_vocab_size", checkpoint_config.original_vocab_size,
                           expected_config.original_vocab_size);
    CompareCheckpointValue("n_layer", checkpoint_config.n_layer, expected_config.n_layer);
    CompareCheckpointValue("n_head", checkpoint_config.n_head, expected_config.n_head);
    CompareCheckpointValue("n_kv_head", checkpoint_config.n_kv_head, expected_config.n_kv_head);
    CompareCheckpointValue("n_embd", checkpoint_config.n_embd, expected_config.n_embd);
    CompareCheckpointValue("ffn_expansion_ratio", checkpoint_config.ffn_expansion_ratio,
                           expected_config.ffn_expansion_ratio);
    CompareCheckpointValue("norm_eps", checkpoint_config.norm_eps, expected_config.norm_eps);
    CompareCheckpointValue("rope_theta", checkpoint_config.rope_theta, expected_config.rope_theta);
    CompareCheckpointValue("use_scaled_rope", checkpoint_config.use_scaled_rope, expected_config.use_scaled_rope);

    CHECK(expected_config.moe_config.has_value()) << "tiny Mixtral runtime config requires MoE config";
    const auto &checkpoint_moe = checkpoint_config.moe_config.value();
    const auto &expected_moe = expected_config.moe_config.value();
    CompareCheckpointValue("num_experts", checkpoint_moe.num_experts, expected_moe.num_experts);
    CompareCheckpointValue("router_topk", checkpoint_moe.router_topk, expected_moe.router_topk);
    CompareCheckpointValue("moe_ffn_hidden_size", checkpoint_moe.moe_ffn_hidden_size, expected_moe.moe_ffn_hidden_size);
}

std::shared_ptr<nn::TransformerModel> LoadFromLLMC(const std::string &filepath,
                                                   const nn::TransformerConfig &expected_config) {
    CheckLLMCConfig(filepath, expected_config);
    auto model = std::make_shared<nn::TransformerModel>(expected_config);

    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs) << "Failed to open tiny Mixtral LLMC file: " << filepath;
    const auto header = infini_train::ReadSeveralBytesFromIfstream(kLLMCHeaderEntries * sizeof(int32_t), &ifs);
    CHECK(ifs) << "Failed to read tiny Mixtral LLMC header: " << filepath;
    CHECK_EQ(infini_train::BytesToType<int32_t>(header, 0 * sizeof(int32_t)), kTinyMixtralLLMCMagic);
    CHECK_EQ(infini_train::BytesToType<int32_t>(header, 1 * sizeof(int32_t)), kTinyMixtralLLMCVersion);

    const auto &config = expected_config;
    auto state = model->StateDict();
    auto read_tensor_by_state_key = [&](const std::string &name) {
        CHECK(state.contains(name)) << "Model state_dict does not contain " << name;
        std::shared_ptr<infini_train::Tensor> tensor = state.at(name);
        CHECK(tensor->Dtype() == infini_train::DataType::kFLOAT32)
            << "Only float32 tiny Mixtral LLMC files are supported: " << name;
        infini_train::ReadMatrixAllFloat(ifs, static_cast<float *>(tensor->DataPtr()), tensor->NumElements(), 1);
        CHECK(ifs) << "Failed to read tensor " << name;
    };

    auto read_projection_into_packed_qkv = [&](const std::string &packed_qkv_name, int64_t row_offset, int64_t num_rows,
                                               const std::string &projection_name) {
        CHECK(state.contains(packed_qkv_name)) << "Model state_dict does not contain " << packed_qkv_name;
        std::shared_ptr<infini_train::Tensor> tensor = state.at(packed_qkv_name);
        CHECK(tensor->Dtype() == infini_train::DataType::kFLOAT32)
            << "Only float32 tiny Mixtral LLMC files are supported: " << projection_name;
        CHECK_EQ(tensor->Dims().size(), 2);
        CHECK_GE(row_offset, 0);
        CHECK_GT(num_rows, 0);
        CHECK_LE(row_offset + num_rows, tensor->Dims()[0]);
        const int64_t cols = tensor->Dims()[1];
        auto *data = static_cast<float *>(tensor->DataPtr()) + row_offset * cols;
        infini_train::ReadMatrixAllFloat(ifs, data, num_rows, cols);
        CHECK(ifs) << "Failed to read tensor rows " << projection_name;
    };

    const auto &moe_config = config.moe_config.value();
    read_tensor_by_state_key("transformer.wte.weight");
    for (int64_t layer = 0; layer < config.n_layer; ++layer) {
        const std::string prefix = "transformer.h." + std::to_string(layer);
        read_tensor_by_state_key(prefix + ".ln_1.weight");
        const auto c_attn_name = prefix + ".attn.c_attn.weight";
        const int64_t head_dim = config.n_embd / config.n_head;
        const int64_t q_rows = config.n_head * head_dim;
        const int64_t kv_rows = config.n_kv_head * head_dim;
        read_projection_into_packed_qkv(c_attn_name, 0, q_rows, c_attn_name + ".q_proj");
        read_projection_into_packed_qkv(c_attn_name, q_rows, kv_rows, c_attn_name + ".k_proj");
        read_projection_into_packed_qkv(c_attn_name, q_rows + kv_rows, kv_rows, c_attn_name + ".v_proj");
        read_tensor_by_state_key(prefix + ".attn.c_proj.weight");
        read_tensor_by_state_key(prefix + ".ln_2.weight");
        read_tensor_by_state_key(prefix + ".mlp.router.weight");
        for (int64_t expert = 0; expert < moe_config.num_experts; ++expert) {
            const std::string expert_prefix = prefix + ".mlp.experts.expert_" + std::to_string(expert);
            read_tensor_by_state_key(expert_prefix + ".c_fc2.weight");  // Mixtral w1/gate_proj
            read_tensor_by_state_key(expert_prefix + ".c_fc.weight");   // Mixtral w3/up_proj
            read_tensor_by_state_key(expert_prefix + ".c_proj.weight"); // Mixtral w2/down_proj
        }
    }
    read_tensor_by_state_key("transformer.ln_f.weight");
    read_tensor_by_state_key("lm_head.weight");

    CHECK_EQ(ifs.peek(), std::ifstream::traits_type::eof()) << "Unexpected trailing bytes in tiny Mixtral LLMC file";
    return model;
}

} // namespace tiny_mixtral
