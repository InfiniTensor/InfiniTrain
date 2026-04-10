#include <cmath>
#include <iostream>
#include <memory>

#include "glog/logging.h"

#include "example/gpt2/config.h"
#include "example/llama3/config.h"
#include "infini_train/include/nn/modules/activations.h"
#include "infini_train/include/nn/modules/normalization.h"
#include "infini_train/include/nn/modules/sparse.h"
#include "infini_train/include/nn/modules/transformer/causal_self_attention.h"
#include "infini_train/include/nn/modules/transformer/mlp.h"
#include "infini_train/include/nn/modules/transformer/transformer.h"
#include "infini_train/include/nn/modules/transformer/transformer_config.h"
#include "infini_train/include/nn/modules/transformer/utils.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;
namespace nn = infini_train::nn;

// ============================================================================
// Test 1: TransformerConfig Validation
// ============================================================================
void TestConfigValidation() {
    std::cout << "\n=== Test 1: TransformerConfig Validation ===" << std::endl;

    bool all_passed = true;

    // Test GPT2 config
    auto gpt2_config = gpt2::GPT2Config();
    if (gpt2_config.attention_type != nn::AttentionType::kStandard) {
        std::cout << "FAIL: GPT2 config should use Standard attention" << std::endl;
        all_passed = false;
    }
    if (gpt2_config.activation_type != nn::MLPType::kGELU) {
        std::cout << "FAIL: GPT2 config should use GELU activation" << std::endl;
        all_passed = false;
    }
    if (gpt2_config.norm_type != nn::NormType::kLayerNorm) {
        std::cout << "FAIL: GPT2 config should use LayerNorm" << std::endl;
        all_passed = false;
    }
    if (!gpt2_config.add_bias_linear) {
        std::cout << "FAIL: GPT2 config should have bias enabled" << std::endl;
        all_passed = false;
    }
    if (!gpt2_config.tie_weights) {
        std::cout << "FAIL: GPT2 config should have tied weights" << std::endl;
        all_passed = false;
    }

    // Test LLaMA3 config
    auto llama3_config = llama3::LLaMA3Config();
    if (llama3_config.attention_type != nn::AttentionType::kRoPE) {
        std::cout << "FAIL: LLaMA3 config should use RoPE attention" << std::endl;
        all_passed = false;
    }
    if (llama3_config.activation_type != nn::MLPType::kSwiGLU) {
        std::cout << "FAIL: LLaMA3 config should use SwiGLU activation" << std::endl;
        all_passed = false;
    }
    if (llama3_config.norm_type != nn::NormType::kRMSNorm) {
        std::cout << "FAIL: LLaMA3 config should use RMSNorm" << std::endl;
        all_passed = false;
    }
    if (llama3_config.add_bias_linear) {
        std::cout << "FAIL: LLaMA3 config should have bias disabled" << std::endl;
        all_passed = false;
    }
    if (llama3_config.tie_weights) {
        std::cout << "FAIL: LLaMA3 config should not have tied weights" << std::endl;
        all_passed = false;
    }

    // Test GQA detection
    if (!llama3_config.UseGQA()) {
        std::cout << "FAIL: LLaMA3 config should detect GQA (n_kv_head < n_head)" << std::endl;
        all_passed = false;
    }
    if (gpt2_config.UseGQA()) {
        std::cout << "FAIL: GPT2 config should not detect GQA (n_kv_head == n_head)" << std::endl;
        all_passed = false;
    }

    if (all_passed) {
        std::cout << "SUCCESS: All config validations passed!" << std::endl;
    }
}

// ============================================================================
// Test 2: Embedding Layer
// ============================================================================
void TestEmbedding() {
    std::cout << "\n=== Test 2: Embedding Layer ===" << std::endl;

    const int64_t vocab_size = 1000;
    const int64_t embedding_dim = 128;
    const int64_t batch_size = 2;
    const int64_t seq_len = 16;

    try {
        auto embedding = std::make_shared<nn::Embedding>(vocab_size, embedding_dim);

        // Check parameters
        auto params = embedding->Parameters();
        if (params.size() != 1) {
            std::cout << "FAIL: Embedding should have 1 parameter, got " << params.size() << std::endl;
            return;
        }

        // Check weight shape
        auto weight = embedding->parameter(nn::Embedding::kParamWeightName);
        if (weight->Dims() != std::vector<int64_t>{vocab_size, embedding_dim}) {
            std::cout << "FAIL: Embedding weight shape mismatch" << std::endl;
            return;
        }

        // Forward pass
        auto input = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, seq_len}, DataType::kINT64);
        auto output = (*embedding)({input});

        if (output.size() != 1) {
            std::cout << "FAIL: Embedding forward should return 1 tensor" << std::endl;
            return;
        }

        const auto &out_dims = output[0]->Dims();
        if (out_dims != std::vector<int64_t>{batch_size, seq_len, embedding_dim}) {
            std::cout << "FAIL: Embedding output shape mismatch. Expected [" << batch_size << ", " << seq_len << ", "
                      << embedding_dim << "], got [" << out_dims[0] << ", " << out_dims[1] << ", " << out_dims[2] << "]"
                      << std::endl;
            return;
        }

        std::cout << "SUCCESS: Embedding layer works correctly!" << std::endl;

    } catch (const std::exception &e) { std::cout << "FAIL: Exception: " << e.what() << std::endl; }
}

// ============================================================================
// Test 3: Normalization Layers (LayerNorm vs RMSNorm)
// ============================================================================
void TestNormalization() {
    std::cout << "\n=== Test 3: Normalization Layers ===" << std::endl;

    const int64_t hidden_size = 64;
    const int64_t batch_size = 2;
    const int64_t seq_len = 8;

    try {
        // Test LayerNorm
        auto layernorm = std::make_shared<nn::LayerNorm>(std::vector<int64_t>{hidden_size});
        auto ln_params = layernorm->Parameters();
        if (ln_params.size() != 2) {
            std::cout << "FAIL: LayerNorm should have 2 parameters (weight, bias), got " << ln_params.size()
                      << std::endl;
            return;
        }

        // Test RMSNorm
        auto rmsnorm = std::make_shared<nn::RMSNorm>(hidden_size);
        auto rms_params = rmsnorm->Parameters();
        if (rms_params.size() != 1) {
            std::cout << "FAIL: RMSNorm should have 1 parameter (weight), got " << rms_params.size() << std::endl;
            return;
        }

        // Forward pass for both
        auto input
            = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, seq_len, hidden_size}, DataType::kFLOAT32);

        auto ln_output = (*layernorm)({input});
        auto rms_output = (*rmsnorm)({input});

        if (ln_output[0]->Dims() != input->Dims()) {
            std::cout << "FAIL: LayerNorm output shape mismatch" << std::endl;
            return;
        }

        if (rms_output[0]->Dims() != input->Dims()) {
            std::cout << "FAIL: RMSNorm output shape mismatch" << std::endl;
            return;
        }

        std::cout << "SUCCESS: Normalization layers work correctly!" << std::endl;

    } catch (const std::exception &e) { std::cout << "FAIL: Exception: " << e.what() << std::endl; }
}

// ============================================================================
// Test 4: MLP Layer (GELU vs SwiGLU)
// ============================================================================
void TestMlp() {
    std::cout << "\n=== Test 4: MLP Layer ===" << std::endl;

    const int64_t hidden_size = 64;
    const int64_t batch_size = 2;
    const int64_t seq_len = 8;

    try {
        // Test GPT2-style MLP (GELU)
        nn::TransformerConfig gpt2_mlp_config;
        gpt2_mlp_config.n_embd = hidden_size;
        gpt2_mlp_config.activation_type = nn::MLPType::kGELU;
        gpt2_mlp_config.ffn_expansion_ratio = 4.0f;
        gpt2_mlp_config.add_bias_linear = true;

        auto gpt2_mlp = std::make_shared<nn::MLP>(gpt2_mlp_config);
        auto gpt2_params = gpt2_mlp->Parameters();

        // GPT2 MLP should have: c_fc.weight, c_fc.bias, c_proj.weight, c_proj.bias
        if (gpt2_params.size() != 4) {
            std::cout << "FAIL: GPT2 MLP should have 4 parameters, got " << gpt2_params.size() << std::endl;
            return;
        }

        // Test LLaMA3-style MLP (SwiGLU)
        nn::TransformerConfig llama3_mlp_config;
        llama3_mlp_config.n_embd = hidden_size;
        llama3_mlp_config.activation_type = nn::MLPType::kSwiGLU;
        llama3_mlp_config.ffn_expansion_ratio = 4.0f;
        llama3_mlp_config.add_bias_linear = false;
        llama3_mlp_config.ffn_dim_multiplier = 1.5f;
        llama3_mlp_config.multiple_of = 256;

        auto llama3_mlp = std::make_shared<nn::MLP>(llama3_mlp_config);
        auto llama3_params = llama3_mlp->Parameters();

        // LLaMA3 MLP should have: c_fc.weight, c_fc2.weight, c_proj.weight (no bias)
        if (llama3_params.size() != 3) {
            std::cout << "FAIL: LLaMA3 MLP should have 3 parameters, got " << llama3_params.size() << std::endl;
            return;
        }

        // Forward pass
        auto input
            = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, seq_len, hidden_size}, DataType::kFLOAT32);

        auto gpt2_output = (*gpt2_mlp)({input});
        auto llama3_output = (*llama3_mlp)({input});

        // Output should have same hidden dimension
        if (gpt2_output[0]->Dims()[2] != hidden_size) {
            std::cout << "FAIL: GPT2 MLP output hidden dim mismatch" << std::endl;
            return;
        }

        if (llama3_output[0]->Dims()[2] != hidden_size) {
            std::cout << "FAIL: LLaMA3 MLP output hidden dim mismatch" << std::endl;
            return;
        }

        std::cout << "SUCCESS: MLP layers work correctly!" << std::endl;

    } catch (const std::exception &e) { std::cout << "FAIL: Exception: " << e.what() << std::endl; }
}

// ============================================================================
// Test 5: CausalSelfAttention
// ============================================================================
void TestAttention() {
    std::cout << "\n=== Test 5: CausalSelfAttention ===" << std::endl;

    const int64_t hidden_size = 64;
    const int64_t batch_size = 2;
    const int64_t seq_len = 8;
    const int64_t n_head = 4;

    try {
        // Test standard attention (GPT2-style)
        nn::TransformerConfig standard_config;
        standard_config.n_embd = hidden_size;
        standard_config.n_head = n_head;
        standard_config.n_kv_head = n_head;
        standard_config.attention_type = nn::AttentionType::kStandard;
        standard_config.add_bias_linear = true;

        auto standard_attn = std::make_shared<nn::CausalSelfAttention>(standard_config);
        auto standard_params = standard_attn->Parameters();

        // Should have c_attn (QKV combined) and c_proj with biases
        if (standard_params.size() != 4) {
            std::cout << "FAIL: Standard attention should have 4 parameters, got " << standard_params.size()
                      << std::endl;
            return;
        }

        // Test RoPE attention with GQA (LLaMA3-style)
        nn::TransformerConfig rope_config;
        rope_config.n_embd = hidden_size;
        rope_config.n_head = n_head;
        rope_config.n_kv_head = 2; // GQA: fewer KV heads
        rope_config.attention_type = nn::AttentionType::kRoPE;
        rope_config.add_bias_linear = false;

        auto rope_attn = std::make_shared<nn::CausalSelfAttention>(rope_config);
        auto rope_params = rope_attn->Parameters();

        // RoPE attention without bias should have fewer params
        if (rope_params.empty()) {
            std::cout << "FAIL: RoPE attention should have parameters" << std::endl;
            return;
        }

        // Forward pass
        auto input
            = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, seq_len, hidden_size}, DataType::kFLOAT32);

        auto standard_output = (*standard_attn)({input});
        if (standard_output[0]->Dims() != input->Dims()) {
            std::cout << "FAIL: Standard attention output shape mismatch" << std::endl;
            return;
        }

        std::cout << "SUCCESS: CausalSelfAttention works correctly!" << std::endl;

    } catch (const std::exception &e) { std::cout << "FAIL: Exception: " << e.what() << std::endl; }
}

// ============================================================================
// Test 6: TransformerLayer
// ============================================================================
void TestTransformerLayer() {
    std::cout << "\n=== Test 6: TransformerLayer ===" << std::endl;

    const int64_t hidden_size = 64;
    const int64_t batch_size = 2;
    const int64_t seq_len = 8;

    try {
        // Test GPT2-style layer
        auto gpt2_config = gpt2::GPT2Config();
        gpt2_config.n_embd = hidden_size;
        gpt2_config.n_head = 4;
        gpt2_config.n_layer = 1;

        auto gpt2_layer = std::make_shared<nn::TransformerLayer>(gpt2_config);
        auto gpt2_params = gpt2_layer->Parameters();

        if (gpt2_params.empty()) {
            std::cout << "FAIL: GPT2 TransformerLayer should have parameters" << std::endl;
            return;
        }

        // Forward pass
        auto input
            = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, seq_len, hidden_size}, DataType::kFLOAT32);

        auto output = (*gpt2_layer)({input});
        if (output[0]->Dims() != input->Dims()) {
            std::cout << "FAIL: TransformerLayer output shape mismatch" << std::endl;
            return;
        }

        // Test LLaMA3-style layer
        auto llama3_config = llama3::LLaMA3Config();
        llama3_config.n_embd = hidden_size;
        llama3_config.n_head = 4;
        llama3_config.n_kv_head = 2;
        llama3_config.n_layer = 1;

        auto llama3_layer = std::make_shared<nn::TransformerLayer>(llama3_config);
        auto llama3_params = llama3_layer->Parameters();

        if (llama3_params.empty()) {
            std::cout << "FAIL: LLaMA3 TransformerLayer should have parameters" << std::endl;
            return;
        }

        std::cout << "SUCCESS: TransformerLayer works correctly!" << std::endl;

    } catch (const std::exception &e) { std::cout << "FAIL: Exception: " << e.what() << std::endl; }
}

// ============================================================================
// Test 7: TransformerModel Instantiation (GPT2)
// ============================================================================
void TestGpt2Model() {
    std::cout << "\n=== Test 7: GPT2 Model Instantiation ===" << std::endl;

    auto config = gpt2::GPT2Config();
    // Use smaller config for faster testing
    config.n_layer = 2;
    config.n_head = 4;
    config.n_embd = 64;

    try {
        auto model = std::make_shared<nn::TransformerModel>(config);

        if (model == nullptr) {
            std::cout << "FAIL: Failed to create GPT2 model" << std::endl;
            return;
        }

        auto params = model->Parameters();
        if (params.empty()) {
            std::cout << "FAIL: GPT2 model has no parameters" << std::endl;
            return;
        }

        std::cout << "SUCCESS: GPT2 model created with " << params.size() << " parameters!" << std::endl;

    } catch (const std::exception &e) { std::cout << "FAIL: Exception: " << e.what() << std::endl; }
}

// ============================================================================
// Test 8: TransformerModel Instantiation (LLaMA3)
// ============================================================================
void TestLlama3Model() {
    std::cout << "\n=== Test 8: LLaMA3 Model Instantiation ===" << std::endl;

    auto config = llama3::LLaMA3Config();
    // Use smaller config for faster testing
    config.n_layer = 2;
    config.n_head = 4;
    config.n_kv_head = 2;
    config.n_embd = 64;

    try {
        auto model = std::make_shared<nn::TransformerModel>(config);

        if (model == nullptr) {
            std::cout << "FAIL: Failed to create LLaMA3 model" << std::endl;
            return;
        }

        auto params = model->Parameters();
        if (params.empty()) {
            std::cout << "FAIL: LLaMA3 model has no parameters" << std::endl;
            return;
        }

        std::cout << "SUCCESS: LLaMA3 model created with " << params.size() << " parameters!" << std::endl;

    } catch (const std::exception &e) { std::cout << "FAIL: Exception: " << e.what() << std::endl; }
}

// ============================================================================
// Test 9: RoPE Utilities
// ============================================================================
void TestRopeUtils() {
    std::cout << "\n=== Test 9: RoPE Utilities ===" << std::endl;

    const int64_t head_dim = 64;
    const int64_t seq_len = 128;

    try {
        // Test precompute freqs_cis
        auto freqs_cis = PrecomputeFreqsCis(head_dim, seq_len);

        // freqs_cis shape: [seq_len, head_dim/2, 2] (cos and sin stacked on last dim)
        const auto &dims = freqs_cis->Dims();
        if (dims.size() != 3) {
            std::cout << "FAIL: freqs_cis should be 3D, got " << dims.size() << "D" << std::endl;
            return;
        }
        if (dims[0] != seq_len) {
            std::cout << "FAIL: freqs_cis seq_len mismatch. Expected " << seq_len << ", got " << dims[0] << std::endl;
            return;
        }
        if (dims[1] != head_dim / 2) {
            std::cout << "FAIL: freqs_cis head_dim/2 mismatch. Expected " << head_dim / 2 << ", got " << dims[1]
                      << std::endl;
            return;
        }
        if (dims[2] != 2) {
            std::cout << "FAIL: freqs_cis last dim should be 2 (cos, sin), got " << dims[2] << std::endl;
            return;
        }

        std::cout << "SUCCESS: RoPE utilities work correctly!" << std::endl;

    } catch (const std::exception &e) { std::cout << "FAIL: Exception: " << e.what() << std::endl; }
}

// ============================================================================
// Test 10: Model StateDict
// ============================================================================
void TestStateDict() {
    std::cout << "\n=== Test 10: Model StateDict ===" << std::endl;

    nn::TransformerConfig config;
    config.n_layer = 1;
    config.n_head = 2;
    config.n_kv_head = 2; // Must set explicitly
    config.n_embd = 32;
    config.vocab_size = 1000;
    config.attention_type = nn::AttentionType::kStandard;
    config.activation_type = nn::MLPType::kGELU;
    config.norm_type = nn::NormType::kLayerNorm;
    config.add_bias_linear = true;

    try {
        auto model = std::make_shared<nn::TransformerModel>(config);
        auto state_dict = model->StateDict();

        if (state_dict.empty()) {
            std::cout << "FAIL: StateDict should not be empty" << std::endl;
            return;
        }

        // StateDict includes both parameters and buffers, so it should have >= parameters count
        auto params = model->Parameters();
        auto buffers = model->Buffers();

        if (state_dict.size() < params.size()) {
            std::cout << "FAIL: StateDict size (" << state_dict.size() << ") should be >= parameter count ("
                      << params.size() << ")" << std::endl;
            return;
        }

        // Expected: state_dict.size() == params.size() + buffers.size()
        size_t expected_size = params.size() + buffers.size();
        if (state_dict.size() != expected_size) {
            std::cout << "FAIL: StateDict size (" << state_dict.size() << ") should equal params (" << params.size()
                      << ") + buffers (" << buffers.size() << ") = " << expected_size << std::endl;
            return;
        }

        std::cout << "SUCCESS: StateDict works correctly with " << state_dict.size() << " entries (" << params.size()
                  << " params + " << buffers.size() << " buffers)!" << std::endl;

    } catch (const std::exception &e) { std::cout << "FAIL: Exception: " << e.what() << std::endl; }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1);

    std::cout << "========================================" << std::endl;
    std::cout << "    Transformer architecture Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    TestConfigValidation();
    TestEmbedding();
    TestNormalization();
    TestMlp();
    TestAttention();
    TestTransformerLayer();
    TestGpt2Model();
    TestLlama3Model();
    TestRopeUtils();
    TestStateDict();

    std::cout << "\n========================================" << std::endl;
    std::cout << "    All Tests Completed" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
