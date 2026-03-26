#include <iostream>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/core/models/decode_only_transformer/layer_specs.h"
#include "infini_train/include/core/models/decode_only_transformer/model.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/core/transformer/activations/gelu.h"
#include "infini_train/include/core/transformer/activations/swiglu.h"
#include "infini_train/include/core/transformer/attention/causal_self_attention.h"
#include "infini_train/include/core/transformer/mlp.h"
#include "infini_train/include/core/transformer/norms/layer_norm.h"
#include "infini_train/include/core/transformer/norms/rms_norm.h"
#include "infini_train/include/core/transformer/spec_utils.h"
#include "infini_train/include/core/transformer/transformer_builders.h"
#include "infini_train/include/core/transformer/transformer_config.h"
#include "infini_train/include/core/transformer/transformer_layer.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cpu/cpu_guard_impl.h"

using namespace infini_train;
namespace nn = infini_train::nn;

// ============================================================================
// Test 1: Basic Module Registration
// ============================================================================
void test_module_registry() {
    std::cout << "\n=== Test 1: Module Registration ===" << std::endl;

    bool all_registered = true;

    // Check all required modules are registeredP
    if (!nn::ModuleRegistry::Instance().Has(typeid(nn::CausalSelfAttention))) {
        std::cout << "FAIL: CausalSelfAttention not registered" << std::endl;
        all_registered = false;
    }

    if (!nn::ModuleRegistry::Instance().Has(typeid(nn::MLP))) {
        std::cout << "FAIL: MLP not registered" << std::endl;
        all_registered = false;
    }

    if (!nn::ModuleRegistry::Instance().Has(typeid(nn::TransformerLayer))) {
        std::cout << "FAIL: TransformerLayer not registered" << std::endl;
        all_registered = false;
    }

    if (!nn::ModuleRegistry::Instance().Has(typeid(nn::LayerNorm))) {
        std::cout << "FAIL: LayerNorm not registered" << std::endl;
        all_registered = false;
    }

    if (!nn::ModuleRegistry::Instance().Has(typeid(nn::RMSNorm))) {
        std::cout << "FAIL: RMSNorm not registered" << std::endl;
        all_registered = false;
    }

    if (!nn::ModuleRegistry::Instance().Has(typeid(nn::Embedding))) {
        std::cout << "FAIL: Embedding not registered" << std::endl;
        all_registered = false;
    }

    if (!nn::ModuleRegistry::Instance().Has(typeid(nn::parallel::ColumnParallelLinear))) {
        std::cout << "FAIL: ColumnParallelLinear not registered" << std::endl;
        all_registered = false;
    }

    if (!nn::ModuleRegistry::Instance().Has(typeid(nn::parallel::RowParallelLinear))) {
        std::cout << "FAIL: RowParallelLinear not registered" << std::endl;
        all_registered = false;
    }

    if (!nn::ModuleRegistry::Instance().Has(typeid(nn::parallel::VocabParallelEmbedding))) {
        std::cout << "FAIL: VocabParallelEmbedding not registered" << std::endl;
        all_registered = false;
    }

    if (all_registered) {
        std::cout << "SUCCESS: All required modules are registered!" << std::endl;
    }
}

// ============================================================================
// Test 2: GPT2 Spec Building
// ============================================================================
void test_gpt2_spec() {
    std::cout << "\n=== Test 2: GPT2 Spec Building ===" << std::endl;

    // Create GPT2 configuration
    nn::TransformerConfig config = nn::TransformerConfig::GPT2();
    config.block_size = 1024;
    config.vocab_size = 50257;
    config.n_layer = 12;
    config.n_head = 12;
    config.n_embd = 768;

    // Build GPT2 spec
    nn::ModuleSpec spec = nn::BuildGPT2Spec(config);

    // Verify spec structure
    bool test_passed = true;

    if (spec.submodules_.empty()) {
        std::cout << "FAIL: GPT2 spec has no submodules" << std::endl;
        test_passed = false;
    }

    if (!spec.submodules_.contains(nn::TransformerFirstStage::kType)) {
        std::cout << "FAIL: GPT2 spec missing 'first_stage'" << std::endl;
        test_passed = false;
    }

    if (!spec.submodules_.contains(nn::TransformerLayer::kType)) {
        std::cout << "FAIL: GPT2 spec missing 'block'" << std::endl;
        test_passed = false;
    }

    if (!spec.submodules_.contains(nn::TransformerLastStage::kType)) {
        std::cout << "FAIL: GPT2 spec missing 'last_stage'" << std::endl;
        test_passed = false;
    }

    // Verify first_stage submodules
    auto &first_stage = spec.submodules_[nn::TransformerFirstStage::kType];
    if (!first_stage.submodules_.contains("wte")) {
        std::cout << "FAIL: first_stage missing 'wte'" << std::endl;
        test_passed = false;
    }

    if (!first_stage.submodules_.contains("wpe")) {
        std::cout << "FAIL: first_stage missing 'wpe'" << std::endl;
        test_passed = false;
    }

    // Verify last_stage submodules
    auto &last_stage = spec.submodules_[nn::TransformerLastStage::kType];
    if (!last_stage.submodules_.contains("ln_f")) {
        std::cout << "FAIL: last_stage missing 'ln_f'" << std::endl;
        test_passed = false;
    }

    if (!last_stage.submodules_.contains("lm_head")) {
        std::cout << "FAIL: last_stage missing 'lm_head'" << std::endl;
        test_passed = false;
    }

    if (test_passed) {
        std::cout << "SUCCESS: GPT2 spec structure is correct!" << std::endl;
    }
}

// ============================================================================
// Test 3: LLaMA3 Spec Building
// ============================================================================
void test_llama3_spec() {
    std::cout << "\n=== Test 3: LLaMA3 Spec Building ===" << std::endl;

    // Create LLaMA3 configuration
    nn::TransformerConfig config = nn::TransformerConfig::LLaMA3();
    config.block_size = 8192;
    config.vocab_size = 128256;
    config.n_layer = 32;
    config.n_head = 32;
    config.n_kv_head = 8;
    config.n_embd = 4096;
    config.ffn_dim_multiplier = 1.3f;
    config.multiple_of = 256;

    // Build LLaMA3 spec
    nn::ModuleSpec spec = nn::BuildLLaMA3Spec(config);

    // Verify spec structure
    bool test_passed = true;

    if (spec.submodules_.empty()) {
        std::cout << "FAIL: LLaMA3 spec has no submodules" << std::endl;
        test_passed = false;
    }

    if (!spec.submodules_.contains(nn::TransformerFirstStage::kType)) {
        std::cout << "FAIL: LLaMA3 spec missing 'first_stage'" << std::endl;
        test_passed = false;
    }

    if (!spec.submodules_.contains(nn::TransformerLayer::kType)) {
        std::cout << "FAIL: LLaMA3 spec missing 'block'" << std::endl;
        test_passed = false;
    }

    if (!spec.submodules_.contains(nn::TransformerLastStage::kType)) {
        std::cout << "FAIL: LLaMA3 spec missing 'last_stage'" << std::endl;
        test_passed = false;
    }

    // Verify first_stage has only wte (LLaMA3 uses RoPE)
    auto &first_stage = spec.submodules_[nn::TransformerFirstStage::kType];
    if (!first_stage.submodules_.contains("wte")) {
        std::cout << "FAIL: first_stage missing 'wte'" << std::endl;
        test_passed = false;
    }

    if (first_stage.submodules_.contains("wpe")) {
        std::cout << "FAIL: first_stage should not have 'wpe' (LLaMA3 uses RoPE)" << std::endl;
        test_passed = false;
    }

    if (test_passed) {
        std::cout << "SUCCESS: LLaMA3 spec structure is correct!" << std::endl;
    }
}

// ============================================================================
// Test 4: GPT2 Model Instantiation
// ============================================================================
void test_gpt2_instantiation() {
    std::cout << "\n=== Test 4: GPT2 Model Instantiation ===" << std::endl;

    nn::TransformerConfig config = nn::TransformerConfig::GPT2();
    config.block_size = 1024;
    config.vocab_size = 50257;
    config.n_layer = 12;
    config.n_head = 12;
    config.n_embd = 768;

    try {
        auto model = std::make_shared<GPT2>(config);

        if (model == nullptr) {
            std::cout << "FAIL: Failed to create GPT2 model" << std::endl;
        } else if (model->Parameters().empty()) {
            std::cout << "FAIL: GPT2 model has no parameters" << std::endl;
        } else {
            std::cout << "SUCCESS: GPT2 model created with " << model->Parameters().size() << " parameters!"
                      << std::endl;
        }
    } catch (const std::exception &e) {
        std::cout << "FAIL: Exception during GPT2 model creation: " << e.what() << std::endl;
    }
}

// ============================================================================
// Test 5: LLaMA3 Model Instantiation
// ============================================================================
void test_llama3_instantiation() {
    std::cout << "\n=== Test 5: LLaMA3 Model Instantiation ===" << std::endl;

    nn::TransformerConfig config = nn::TransformerConfig::LLaMA3();

    try {
        auto model = std::make_shared<LLaMA3>(config);

        if (model == nullptr) {
            std::cout << "FAIL: Failed to create LLaMA3 model" << std::endl;
        } else if (model->Parameters().empty()) {
            std::cout << "FAIL: LLaMA3 model has no parameters" << std::endl;
        } else {
            std::cout << "SUCCESS: LLaMA3 model created with " << model->Parameters().size() << " parameters!"
                      << std::endl;
        }
    } catch (const std::exception &e) {
        std::cout << "FAIL: Exception during LLaMA3 model creation: " << e.what() << std::endl;
    }
}

// ============================================================================
// Test 6: Dimension Validation (Simplified)
// ============================================================================
void test_dimensions() {
    std::cout << "\n=== Test 6: Dimension Validation ===" << std::endl;

    nn::TransformerConfig config = nn::TransformerConfig::GPT2();
    config.block_size = 1024;
    config.vocab_size = 50257;
    config.n_layer = 12;
    config.n_head = 12;
    config.n_embd = 768;

    try {
        auto model = std::make_shared<GPT2>(config);

        // Create input tensor (batch, seq_len)
        std::vector<int64_t> input_shape = {2, 64};
        auto input = std::make_shared<Tensor>(input_shape, DataType::kINT64, Device());

        // Forward pass
        auto output = (*model)({input});

        // Verify output dimensions (batch, seq_len, vocab_size)
        if (output.empty()) {
            std::cout << "FAIL: Model produced no output" << std::endl;
        } else if (output[0]->Dims().size() != 3) {
            std::cout << "FAIL: Expected 3D output, got " << output[0]->Dims().size() << "D" << std::endl;
        } else if (output[0]->Dims()[0] != 2) {
            std::cout << "FAIL: Expected batch size 2, got " << output[0]->Dims()[0] << std::endl;
        } else if (output[0]->Dims()[1] != 64) {
            std::cout << "FAIL: Expected seq length 64, got " << output[0]->Dims()[1] << std::endl;
        } else if (output[0]->Dims()[2] != config.vocab_size) {
            std::cout << "FAIL: Expected vocab size " << config.vocab_size << ", got " << output[0]->Dims()[2]
                      << std::endl;
        } else {
            std::cout << "SUCCESS: Output dimensions are correct! (" << output[0]->Dims()[0] << ", "
                      << output[0]->Dims()[1] << ", " << output[0]->Dims()[2] << ")" << std::endl;
        }
    } catch (const std::exception &e) {
        std::cout << "FAIL: Exception during dimension test: " << e.what() << std::endl;
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    nn::parallel::global::GlobalEnv::Instance().Init(1, 1, false, 1, 1);

    std::cout << "========================================" << std::endl;
    std::cout << "    Transformer Spec Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_module_registry();
    test_gpt2_spec();
    test_llama3_spec();
    test_gpt2_instantiation();
    test_llama3_instantiation();
    test_dimensions();

    std::cout << "\n========================================" << std::endl;
    std::cout << "    All Tests Completed" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
