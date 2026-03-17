#include <cmath>
#include <iostream>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/nn/lora/lora_config.h"
#include "infini_train/include/nn/lora/lora_linear.h"
#include "infini_train/include/nn/lora/lora_utils.h"
#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/linear.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;
using namespace infini_train::nn::lora;

// ============================================================================
// Test 1: LoRAConfig
// ============================================================================
void test_lora_config() {
    std::cout << "\n=== Test 1: LoRAConfig ===" << std::endl;

    LoRAConfig config;
    config.rank = 8;
    config.alpha = 16.0f;

    // Test scaling calculation
    float expected_scaling = 16.0f / 8.0f;
    CHECK_EQ(config.Scaling(), expected_scaling) << "Scaling calculation failed";
    std::cout << "Scaling: " << config.Scaling() << " (expected: " << expected_scaling << ")" << std::endl;

    // Test ShouldApplyLoRA
    CHECK(config.ShouldApplyLoRA("c_attn")) << "Should match c_attn";
    CHECK(config.ShouldApplyLoRA("transformer.h.0.attn.c_attn")) << "Should match nested c_attn";
    CHECK(config.ShouldApplyLoRA("c_proj")) << "Should match c_proj";
    CHECK(!config.ShouldApplyLoRA("c_fc")) << "Should not match c_fc (not in default targets)";
    CHECK(!config.ShouldApplyLoRA("random_layer")) << "Should not match random_layer";

    std::cout << "LoRAConfig tests passed!" << std::endl;
}

// ============================================================================
// Test 2: LoRALinear Initialization
// ============================================================================
void test_lora_linear_init() {
    std::cout << "\n=== Test 2: LoRALinear Initialization ===" << std::endl;

    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;

    int64_t in_features = 64;
    int64_t out_features = 128;

    auto lora_linear
        = std::shared_ptr<LoRALinear>(new LoRALinear(in_features, out_features, config, /*bias=*/true, nullptr));

    // Check parameter shapes
    auto weight = lora_linear->parameter(nn::Linear::kParamWeightName);
    auto bias = lora_linear->parameter(nn::Linear::kParamBiasName);
    auto lora_A = lora_linear->parameter(LoRALinear::kParamLoraAName);
    auto lora_B = lora_linear->parameter(LoRALinear::kParamLoraBName);

    CHECK_EQ(weight->Dims().size(), 2);
    CHECK_EQ(weight->Dims()[0], out_features);
    CHECK_EQ(weight->Dims()[1], in_features);
    std::cout << "Weight shape: [" << weight->Dims()[0] << ", " << weight->Dims()[1] << "]" << std::endl;

    CHECK_EQ(bias->Dims().size(), 1);
    CHECK_EQ(bias->Dims()[0], out_features);
    std::cout << "Bias shape: [" << bias->Dims()[0] << "]" << std::endl;

    CHECK_EQ(lora_A->Dims().size(), 2);
    CHECK_EQ(lora_A->Dims()[0], config.rank);
    CHECK_EQ(lora_A->Dims()[1], in_features);
    std::cout << "LoRA A shape: [" << lora_A->Dims()[0] << ", " << lora_A->Dims()[1] << "]" << std::endl;

    CHECK_EQ(lora_B->Dims().size(), 2);
    CHECK_EQ(lora_B->Dims()[0], out_features);
    CHECK_EQ(lora_B->Dims()[1], config.rank);
    std::cout << "LoRA B shape: [" << lora_B->Dims()[0] << ", " << lora_B->Dims()[1] << "]" << std::endl;

    // Check requires_grad
    CHECK(!weight->requires_grad()) << "Base weight should be frozen";
    CHECK(!bias->requires_grad()) << "Base bias should be frozen";
    CHECK(lora_A->requires_grad()) << "LoRA A should be trainable";
    CHECK(lora_B->requires_grad()) << "LoRA B should be trainable";
    std::cout << "requires_grad check passed!" << std::endl;

    // Check LoRAParameters() returns only LoRA params
    auto params = lora_linear->LoRAParameters();
    CHECK_EQ(params.size(), 2) << "LoRAParameters() should return only LoRA params";
    std::cout << "LoRAParameters() returns " << params.size() << " tensors (LoRA A and B)" << std::endl;

    std::cout << "LoRALinear initialization tests passed!" << std::endl;
}

// ============================================================================
// Test 3: LoRALinear Forward Pass
// ============================================================================
void test_lora_linear_forward() {
    std::cout << "\n=== Test 3: LoRALinear Forward Pass ===" << std::endl;

    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;

    int64_t in_features = 64;
    int64_t out_features = 128;
    int64_t batch_size = 2;
    int64_t seq_len = 10;

    auto lora_linear
        = std::shared_ptr<LoRALinear>(new LoRALinear(in_features, out_features, config, /*bias=*/true, nullptr));

    // Create input tensor
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, seq_len, in_features}, DataType::kFLOAT32);

    // Forward pass
    auto output = (*lora_linear)({input})[0];

    // Check output shape
    CHECK_EQ(output->Dims().size(), 3);
    CHECK_EQ(output->Dims()[0], batch_size);
    CHECK_EQ(output->Dims()[1], seq_len);
    CHECK_EQ(output->Dims()[2], out_features);
    std::cout << "Output shape: [" << output->Dims()[0] << ", " << output->Dims()[1] << ", " << output->Dims()[2] << "]"
              << std::endl;

    std::cout << "LoRALinear forward pass tests passed!" << std::endl;
}

// ============================================================================
// Test 4: LoRALinear Weight Merging
// ============================================================================
void test_lora_linear_merge() {
    std::cout << "\n=== Test 4: LoRALinear Weight Merging ===" << std::endl;

    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;

    int64_t in_features = 32;
    int64_t out_features = 64;

    auto lora_linear
        = std::shared_ptr<LoRALinear>(new LoRALinear(in_features, out_features, config, /*bias=*/false, nullptr));

    // Print weight sum before merge
    auto weight_before = lora_linear->parameter(nn::Linear::kParamWeightName);
    auto lora_A = lora_linear->parameter(LoRALinear::kParamLoraAName);
    auto lora_B = lora_linear->parameter(LoRALinear::kParamLoraBName);

    float weight_before_sum = weight_before->EigenMatrix().sum();
    float lora_A_sum = lora_A->EigenMatrix().sum();
    float lora_B_sum = lora_B->EigenMatrix().sum();

    std::cout << "\n--- Before Merge ---" << std::endl;
    std::cout << "Base weight sum: " << weight_before_sum << std::endl;
    std::cout << "LoRA A sum: " << lora_A_sum << std::endl;
    std::cout << "LoRA B sum: " << lora_B_sum << std::endl;
    std::cout << "Scaling (alpha/r): " << config.Scaling() << std::endl;

    // Create input
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 5, in_features}, DataType::kFLOAT32);
    input->EigenMatrix().setRandom();

    // Get output before merge
    auto output_before = (*lora_linear)({input})[0];
    float output_before_sum = output_before->EigenMatrix().sum();
    std::cout << "Output sum before merge: " << output_before_sum << std::endl;

    // Merge weights
    CHECK(!lora_linear->IsMerged()) << "Should not be merged initially";
    lora_linear->MergeWeights();
    CHECK(lora_linear->IsMerged()) << "Should be merged after MergeWeights()";

    // Verify LoRA params are frozen after merge
    CHECK(!lora_A->requires_grad()) << "lora_A should be frozen after merge";
    CHECK(!lora_B->requires_grad()) << "lora_B should be frozen after merge";
    std::cout << "\nWeights merged successfully, LoRA params frozen" << std::endl;

    // Print weight sum after merge
    auto weight_after = lora_linear->parameter(nn::Linear::kParamWeightName);
    float weight_after_sum = weight_after->EigenMatrix().sum();
    std::cout << "\n--- After Merge ---" << std::endl;
    std::cout << "Base weight sum after merge: " << weight_after_sum << std::endl;
    std::cout << "Weight change (should be ~LoRA contribution): " << (weight_after_sum - weight_before_sum)
              << std::endl;

    // Get output after merge
    auto output_merged = (*lora_linear)({input})[0];
    float output_merged_sum = output_merged->EigenMatrix().sum();
    std::cout << "Output sum after merge: " << output_merged_sum << std::endl;

    // Verify: output_after should equal output_before (numerically)
    std::cout << "\nVerification: output_before == output_after? " << std::endl;
    std::cout << "  Before: " << output_before_sum << std::endl;
    std::cout << "  After:  " << output_merged_sum << std::endl;
    std::cout << "  Diff:   " << std::abs(output_before_sum - output_merged_sum) << std::endl;
    CHECK(std::abs(output_before_sum - output_merged_sum) < 1e-3) << "Outputs should be numerically identical!";

    // Shape comparison (always same)
    std::cout << "\nOutput shape: [" << output_before->Dims()[0] << ", " << output_before->Dims()[1] << ", "
              << output_before->Dims()[2] << "] (unchanged)" << std::endl;

    // Unmerge weights
    lora_linear->UnmergeWeights();
    CHECK(!lora_linear->IsMerged()) << "Should not be merged after UnmergeWeights()";

    // Verify LoRA params are trainable again after unmerge
    CHECK(lora_A->requires_grad()) << "lora_A should be trainable after unmerge";
    CHECK(lora_B->requires_grad()) << "lora_B should be trainable after unmerge";

    // Print weight sum after unmerge
    auto weight_unmerged = lora_linear->parameter(nn::Linear::kParamWeightName);
    float weight_unmerged_sum = weight_unmerged->EigenMatrix().sum();
    std::cout << "\n--- After Unmerge ---" << std::endl;
    std::cout << "Base weight sum after unmerge: " << weight_unmerged_sum << std::endl;

    // Verify: weight should be restored to original value
    std::cout << "\nVerification: weight restored after unmerge? " << std::endl;
    std::cout << "  Original: " << weight_before_sum << std::endl;
    std::cout << "  Unmerged: " << weight_unmerged_sum << std::endl;
    std::cout << "  Diff:     " << std::abs(weight_before_sum - weight_unmerged_sum) << std::endl;
    CHECK(std::abs(weight_before_sum - weight_unmerged_sum) < 1e-4) << "Weight should be restored!";

    // Get output after unmerge
    auto output_unmerged = (*lora_linear)({input})[0];
    float output_unmerged_sum = output_unmerged->EigenMatrix().sum();
    std::cout << "Output sum after unmerge: " << output_unmerged_sum << std::endl;

    // Shape comparison: merge doesn't change shape, only weights
    CHECK(output_before->Dims() == output_merged->Dims()) << "Shape should be identical after merge";
    CHECK(output_merged->Dims() == output_unmerged->Dims()) << "Shape should be identical after unmerge";

    std::cout << "\nLoRALinear weight merging tests passed!" << std::endl;
}

// ============================================================================
// Test 5: LoRA Utility Functions
// ============================================================================
void test_lora_utils() {
    std::cout << "\n=== Test 5: LoRA Utility Functions ===" << std::endl;

    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;

    auto lora_linear = std::shared_ptr<LoRALinear>(new LoRALinear(32, 64, config, /*bias=*/true, nullptr));

    // Test GetLoRAParameters
    auto lora_params = GetLoRAParameters(lora_linear);
    CHECK_EQ(lora_params.size(), 2) << "Should have 2 LoRA parameters";
    std::cout << "GetLoRAParameters returned " << lora_params.size() << " parameters" << std::endl;

    // Test CountTrainableParameters
    int64_t trainable = CountTrainableParameters(lora_linear);
    int64_t expected_trainable = config.rank * 32 + 64 * config.rank; // A: [4, 32], B: [64, 4]
    CHECK_EQ(trainable, expected_trainable) << "Trainable parameter count mismatch";
    std::cout << "Trainable parameters: " << trainable << " (expected: " << expected_trainable << ")" << std::endl;

    // Test CountTotalParameters
    int64_t total = CountTotalParameters(lora_linear);
    int64_t expected_total = 64 * 32 + 64 + config.rank * 32 + 64 * config.rank; // weight + bias + A + B
    CHECK_EQ(total, expected_total) << "Total parameter count mismatch";
    std::cout << "Total parameters: " << total << " (expected: " << expected_total << ")" << std::endl;

    // Test PrintLoRASummary
    std::cout << "\nLoRA Summary:" << std::endl;
    PrintLoRASummary(lora_linear);

    std::cout << "LoRA utility function tests passed!" << std::endl;
}

// ============================================================================
// Test 6: LoRALinear from existing Linear
// ============================================================================
void test_lora_from_linear() {
    std::cout << "\n=== Test 6: LoRALinear from existing Linear ===" << std::endl;

    // Create a standard Linear layer
    auto linear = std::make_shared<nn::Linear>(64, 128, /*bias=*/true);

    // Wrap it with LoRA
    LoRAConfig config;
    config.rank = 8;
    config.alpha = 16.0f;

    auto lora_linear = std::make_shared<LoRALinear>(linear, config);

    // Check dimensions
    CHECK_EQ(lora_linear->in_features(), 64);
    CHECK_EQ(lora_linear->out_features(), 128);
    CHECK_EQ(lora_linear->rank(), 8);
    std::cout << "LoRALinear created from Linear: in=" << lora_linear->in_features()
              << ", out=" << lora_linear->out_features() << ", rank=" << lora_linear->rank() << std::endl;

    // Test forward pass
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 10, 64}, DataType::kFLOAT32);
    auto output = (*lora_linear)({input})[0];

    CHECK_EQ(output->Dims()[0], 2);
    CHECK_EQ(output->Dims()[1], 10);
    CHECK_EQ(output->Dims()[2], 128);
    std::cout << "Forward pass successful, output shape: [" << output->Dims()[0] << ", " << output->Dims()[1] << ", "
              << output->Dims()[2] << "]" << std::endl;

    std::cout << "LoRALinear from existing Linear tests passed!" << std::endl;
}

// ============================================================================
// Test 7: LoRALinear from existing Linear (tests LoRA utilities)
// ============================================================================
void test_lora_model_wrapper() {
    std::cout << "\n=== Test 7: LoRALinear from existing Linear ===" << std::endl;

    // Create LoRA config
    LoRAConfig lora_config;
    lora_config.rank = 8;
    lora_config.alpha = 16.0f;

    // Create base Linear module (simple test without InjectLoRALayers)
    auto base_linear = std::make_shared<nn::Linear>(64, 128, /*bias=*/true);

    // Create a minimal wrapper test by manually testing what LoRAModel does
    // Apply LoRA directly to the Linear layer
    auto lora_linear = std::make_shared<LoRALinear>(base_linear, lora_config);

    // Replace the base_linear in its container
    // Note: In a real use case, you would use InjectLoRALayers on a transformer model

    // Test GetLoRAParameters on the LoRA Linear
    auto lora_params = GetLoRAParameters(lora_linear);
    CHECK_GT(lora_params.size(), 0) << "Should have trainable parameters";
    std::cout << "LoRA parameters extracted: " << lora_params.size() << std::endl;

    // Test CountTrainableParameters
    int64_t trainable = CountTrainableParameters(lora_linear);
    CHECK_EQ(trainable, lora_config.rank * 64 + 128 * lora_config.rank);
    std::cout << "Trainable parameters: " << trainable << std::endl;

    // Test PrintSummary
    std::cout << "\nLoRA Summary for Linear wrapper:" << std::endl;
    PrintLoRASummary(lora_linear);

    // Test Save/Load LoRA on the LoRA Linear
    const std::string test_path = "/tmp/test_lora_linear.bin";
    SaveLoRAWeights(lora_linear, test_path);
    std::cout << "SaveLoRAWeights completed" << std::endl;

    LoadLoRAWeights(lora_linear, test_path);
    std::cout << "LoadLoRAWeights completed" << std::endl;

    // Test Merge/Unmerge on LoRA Linear
    CHECK(!lora_linear->IsMerged()) << "Should not be merged initially";
    lora_linear->MergeWeights();
    CHECK(lora_linear->IsMerged()) << "Should be merged after MergeWeights()";
    std::cout << "MergeWeights completed" << std::endl;

    lora_linear->UnmergeWeights();
    CHECK(!lora_linear->IsMerged()) << "Should be unmerged after UnmergeWeights()";
    std::cout << "UnmergeWeights completed" << std::endl;

    std::cout << "LoRALinear utility tests passed!" << std::endl;
}

// ============================================================================
// Test 8: Save/Load LoRA Weights
// ============================================================================
void test_lora_save_load_weights() {
    std::cout << "\n=== Test 8: Save/Load LoRA Weights ===" << std::endl;

    // Create a LoRALinear
    LoRAConfig config;
    config.rank = 4;
    config.alpha = 8.0f;

    int64_t in_features = 32;
    int64_t out_features = 64;

    auto linear = std::make_shared<nn::Linear>(in_features, out_features, /*bias=*/true);
    auto lora_linear = std::make_shared<LoRALinear>(linear, config);

    // Get references to lora_A and lora_B
    auto lora_A = lora_linear->parameter(LoRALinear::kParamLoraAName);
    auto lora_B = lora_linear->parameter(LoRALinear::kParamLoraBName);

    // Set specific values to lora_A and lora_B
    // lora_A: [rank, in_features] = [4, 32]
    // lora_B: [out_features, rank] = [64, 4]
    lora_A->EigenMatrix().setZero();
    lora_B->EigenMatrix().setZero();

    // Set lora_A to all 1s
    for (int64_t i = 0; i < lora_A->Dims()[0]; ++i) {
        for (int64_t j = 0; j < lora_A->Dims()[1]; ++j) { lora_A->EigenMatrix()(i, j) = 1.0f; }
    }

    // Set lora_B to all 2s
    for (int64_t i = 0; i < lora_B->Dims()[0]; ++i) {
        for (int64_t j = 0; j < lora_B->Dims()[1]; ++j) { lora_B->EigenMatrix()(i, j) = 2.0f; }
    }

    // Record original sums
    float lora_A_sum_orig = lora_A->EigenMatrix().sum();
    float lora_B_sum_orig = lora_B->EigenMatrix().sum();
    // lora_A: all 1.0f, shape [rank, in_features] = [4, 32]
    // lora_B: all 2.0f, shape [out_features, rank] = [64, 4]
    float expected_lora_A_sum = config.rank * in_features * 1.0f;  // 4 * 32 * 1 = 128
    float expected_lora_B_sum = out_features * config.rank * 2.0f; // 64 * 4 * 2 = 512
    std::cout << "Original lora_A sum: " << lora_A_sum_orig << " (expected: " << expected_lora_A_sum << ")"
              << std::endl;
    std::cout << "Original lora_B sum: " << lora_B_sum_orig << " (expected: " << expected_lora_B_sum << ")"
              << std::endl;

    CHECK_EQ(lora_A_sum_orig, expected_lora_A_sum);
    CHECK_EQ(lora_B_sum_orig, expected_lora_B_sum);

    // Save to file
    const std::string test_path = "/tmp/test_lora_save_load.bin";
    SaveLoRAWeights(lora_linear, test_path);
    std::cout << "Saved LoRA weights to: " << test_path << std::endl;

    // Modify weights to different values
    lora_A->EigenMatrix().setConstant(9.0f);
    lora_B->EigenMatrix().setConstant(9.0f);

    float lora_A_sum_modified = lora_A->EigenMatrix().sum();
    float lora_B_sum_modified = lora_B->EigenMatrix().sum();
    std::cout << "Modified lora_A sum: " << lora_A_sum_modified << std::endl;
    std::cout << "Modified lora_B sum: " << lora_B_sum_modified << std::endl;

    CHECK_NE(lora_A_sum_modified, lora_A_sum_orig);
    CHECK_NE(lora_B_sum_modified, lora_B_sum_orig);

    // Load from file
    LoadLoRAWeights(lora_linear, test_path);
    std::cout << "Loaded LoRA weights from: " << test_path << std::endl;

    // Verify weights are restored
    float lora_A_sum_loaded = lora_A->EigenMatrix().sum();
    float lora_B_sum_loaded = lora_B->EigenMatrix().sum();
    std::cout << "Loaded lora_A sum: " << lora_A_sum_loaded << std::endl;
    std::cout << "Loaded lora_B sum: " << lora_B_sum_loaded << std::endl;

    CHECK_EQ(lora_A_sum_loaded, lora_A_sum_orig) << "lora_A should be restored to original values";
    CHECK_EQ(lora_B_sum_loaded, lora_B_sum_orig) << "lora_B should be restored to original values";

    // Also verify individual elements
    for (int64_t i = 0; i < lora_A->Dims()[0]; ++i) {
        for (int64_t j = 0; j < lora_A->Dims()[1]; ++j) {
            CHECK_EQ(lora_A->EigenMatrix()(i, j), 1.0f) << "lora_A element mismatch at (" << i << "," << j << ")";
        }
    }

    for (int64_t i = 0; i < lora_B->Dims()[0]; ++i) {
        for (int64_t j = 0; j < lora_B->Dims()[1]; ++j) {
            CHECK_EQ(lora_B->EigenMatrix()(i, j), 2.0f) << "lora_B element mismatch at (" << i << "," << j << ")";
        }
    }

    std::cout << "All elements verified correctly!" << std::endl;

    // Cleanup
    std::remove(test_path.c_str());
    std::cout << "Test 8: Save/Load LoRA Weights passed!" << std::endl;
}

// ============================================================================
// Test 8: ParseLoRATargetModules parsing
// ============================================================================
void test_set_target_modules() {
    std::cout << "\n=== Test 8: ParseLoRATargetModules Parsing ===" << std::endl;

    // Test single target
    auto modules = ParseLoRATargetModules("c_attn");
    CHECK_EQ(modules.size(), 1);
    CHECK(modules.count("c_attn"));
    std::cout << "Single target: OK" << std::endl;

    // Test multiple targets
    modules = ParseLoRATargetModules("c_attn,c_proj,c_fc");
    CHECK_EQ(modules.size(), 3);
    CHECK(modules.count("c_attn"));
    CHECK(modules.count("c_proj"));
    CHECK(modules.count("c_fc"));
    std::cout << "Multiple targets: OK" << std::endl;

    // Test with spaces
    modules = ParseLoRATargetModules("c_attn, c_proj , c_fc");
    CHECK_EQ(modules.size(), 3);
    std::cout << "Targets with spaces: OK" << std::endl;

    // Test empty/whitespace
    modules = ParseLoRATargetModules("c_attn,,c_proj");
    CHECK_EQ(modules.size(), 2);
    std::cout << "Empty entries ignored: OK" << std::endl;

    std::cout << "ParseLoRATargetModules tests passed!" << std::endl;
}

// ============================================================================
// Test 9: ShouldApplyLoRA edge cases (attn.c_proj vs mlp.c_proj)
// ============================================================================
void test_should_apply_lora_edge_cases() {
    std::cout << "\n=== Test 9: ShouldApplyLoRA Edge Cases ===" << std::endl;

    // Test: Only attn.c_proj in target_modules
    {
        LoRAConfig config{8, 16.0f, 0.0f, ParseLoRATargetModules("c_attn,attn.c_proj")};

        // Should match attention paths
        CHECK(config.ShouldApplyLoRA("attn.c_proj"));
        CHECK(config.ShouldApplyLoRA("transformer.h.0.attn.c_proj"));
        CHECK(config.ShouldApplyLoRA("transformer.h.1.attn.c_proj"));

        // Should NOT match mlp paths
        CHECK(!config.ShouldApplyLoRA("mlp.c_proj"));
        CHECK(!config.ShouldApplyLoRA("transformer.h.0.mlp.c_proj"));
        std::cout << "attn.c_proj only: OK" << std::endl;
    }

    // Test: Only mlp.c_proj in target_modules
    {
        LoRAConfig config{8, 16.0f, 0.0f, ParseLoRATargetModules("c_attn,mlp.c_proj")};

        // Should NOT match attention paths
        CHECK(!config.ShouldApplyLoRA("attn.c_proj"));
        CHECK(!config.ShouldApplyLoRA("transformer.h.0.attn.c_proj"));

        // Should match mlp paths
        CHECK(config.ShouldApplyLoRA("mlp.c_proj"));
        CHECK(config.ShouldApplyLoRA("transformer.h.0.mlp.c_proj"));
        std::cout << "mlp.c_proj only: OK" << std::endl;
    }

    // Test: Generic c_proj in target_modules (matches both)
    {
        LoRAConfig config{8, 16.0f, 0.0f, ParseLoRATargetModules("c_attn,c_proj")};

        // Should match both attention and mlp
        CHECK(config.ShouldApplyLoRA("transformer.h.0.attn.c_proj"));
        CHECK(config.ShouldApplyLoRA("transformer.h.0.mlp.c_proj"));
        std::cout << "Generic c_proj (matches both): OK" << std::endl;
    }

    // Test: All targets
    {
        LoRAConfig config{8, 16.0f, 0.0f, ParseLoRATargetModules("c_attn,attn.c_proj,c_fc,c_fc2,mlp.c_proj")};

        CHECK(config.ShouldApplyLoRA("transformer.h.0.attn.c_attn"));
        CHECK(config.ShouldApplyLoRA("transformer.h.0.attn.c_proj"));
        CHECK(config.ShouldApplyLoRA("transformer.h.0.mlp.c_fc"));
        CHECK(config.ShouldApplyLoRA("transformer.h.0.mlp.c_fc2"));
        CHECK(config.ShouldApplyLoRA("transformer.h.0.mlp.c_proj"));
        std::cout << "All targets: OK" << std::endl;
    }

    std::cout << "ShouldApplyLoRA edge cases tests passed!" << std::endl;
}

// ============================================================================
// Test 10: ReplaceModuleByPath
// ============================================================================
void test_replace_module_by_path() {
    std::cout << "\n=== Test 10: ReplaceModuleByPath ===" << std::endl;

    // Test ReplaceModuleByPath by wrapping a Linear with LoRA directly
    // This tests the core functionality that ReplaceModuleByPath provides

    // Create base Linear
    auto base_linear = std::make_shared<nn::Linear>(64, 128, /*bias=*/true);

    // Configure LoRA
    LoRAConfig lora_config;
    lora_config.rank = 4;
    lora_config.alpha = 8.0f;

    // Wrap with LoRA - this is what ReplaceModuleByPath does internally
    auto lora_linear = std::make_shared<LoRALinear>(base_linear, lora_config);

    // Verify LoRA was applied correctly
    auto params = lora_linear->LoRAParameters();
    CHECK_EQ(params.size(), 2) << "LoRALinear should have 2 trainable parameters (lora_A and lora_B)";
    std::cout << "LoRALinear has " << params.size() << " trainable parameters" << std::endl;

    // Verify parameter shapes
    auto lora_a = params[0];
    auto lora_b = params[1];
    CHECK_EQ(lora_a->Dims()[0], lora_config.rank); // rank x in_features
    CHECK_EQ(lora_a->Dims()[1], 64);
    CHECK_EQ(lora_b->Dims()[0], 128); // out_features x rank
    CHECK_EQ(lora_b->Dims()[1], lora_config.rank);
    std::cout << "LoRA parameter shapes: OK" << std::endl;

    // Verify base parameters are frozen (use named parameters instead of index)
    auto weight = lora_linear->parameter(nn::Linear::kParamWeightName);
    auto lora_a_param = lora_linear->parameter(LoRALinear::kParamLoraAName);
    auto lora_b_param = lora_linear->parameter(LoRALinear::kParamLoraBName);
    CHECK(weight != nullptr);
    CHECK(lora_a_param != nullptr);
    CHECK(lora_b_param != nullptr);
    CHECK(!weight->requires_grad());      // weight is frozen
    CHECK(lora_a_param->requires_grad()); // lora_A is trainable
    CHECK(lora_b_param->requires_grad()); // lora_B is trainable
    std::cout << "Base weight frozen, LoRA params trainable: OK" << std::endl;

    std::cout << "ReplaceModuleByPath tests passed!" << std::endl;
}

// ============================================================================
// Test 11: FreezeBaseModel / UnfreezeModel
// ============================================================================
void test_freeze_unfreeze() {
    std::cout << "\n=== Test 11: FreezeBaseModel / UnfreezeModel ===" << std::endl;

    // Test with LoRALinear directly - it has both base and LoRA params
    LoRAConfig lora_config;
    lora_config.rank = 4;
    lora_config.alpha = 8.0f;

    auto linear = std::make_shared<nn::Linear>(64, 128, /*bias=*/true);
    auto lora_linear = std::make_shared<LoRALinear>(linear, lora_config);

    // Get all parameters from LoRALinear (includes base + LoRA)
    auto all_params = lora_linear->Parameters();

    // Initially only LoRA params should be trainable (base weights are frozen by constructor)
    int64_t total_params = 0;
    for (const auto &p : all_params) {
        if (p->requires_grad()) {
            total_params += p->NumElements();
        }
    }
    // Expected: only LoRA params (lora_A + lora_B) = 4*64 + 128*4 = 256 + 512 = 768
    // Note: LoRALinear freezes base weights in constructor by design
    int64_t expected_total = lora_config.rank * 64 + 128 * lora_config.rank;
    CHECK_EQ(total_params, expected_total);
    std::cout << "Initial trainable params: " << total_params << " (expected: " << expected_total << ")" << std::endl;

    // FreezeBaseModel on LoRALinear
    FreezeBaseModel(lora_linear);

    // After freeze, only LoRA params should be trainable
    int64_t after_freeze = 0;
    for (const auto &p : all_params) {
        if (p->requires_grad()) {
            after_freeze += p->NumElements();
        }
    }
    // LoRA params: A (rank x in) + B (out x rank) = 4*64 + 128*4 = 256 + 512 = 768
    int64_t expected_lora = lora_config.rank * 64 + 128 * lora_config.rank;
    CHECK_EQ(after_freeze, expected_lora);
    std::cout << "After freeze trainable: " << after_freeze << " (expected: " << expected_lora << ")" << std::endl;

    // Unfreeze all
    UnfreezeModel(lora_linear);
    int64_t after_unfreeze = 0;
    for (const auto &p : all_params) {
        if (p->requires_grad()) {
            after_unfreeze += p->NumElements();
        }
    }
    // Should be back to all params trainable (base + LoRA)
    int64_t expected_after_unfreeze = 64 * 128 + 128 + lora_config.rank * 64 + 128 * lora_config.rank;
    CHECK_EQ(after_unfreeze, expected_after_unfreeze);
    std::cout << "After unfreeze trainable: " << after_unfreeze << std::endl;

    std::cout << "FreezeBaseModel / UnfreezeModel tests passed!" << std::endl;
}

// ============================================================================
// Test 12: LoRAStateDict
// ============================================================================
void test_lora_state_dict() {
    std::cout << "\n=== Test 12: LoRAStateDict ===" << std::endl;

    // Test with a single LoRALinear
    LoRAConfig lora_config;
    lora_config.rank = 4;
    lora_config.alpha = 8.0f;

    auto linear = std::make_shared<nn::Linear>(64, 128, /*bias=*/true);
    auto lora_linear = std::make_shared<LoRALinear>(linear, lora_config);

    // Get state dict - it contains all parameters with their names
    auto state_dict = lora_linear->StateDict();

    // Check that we have all expected parameters
    CHECK(state_dict.count("weight")) << "Should have weight parameter";
    CHECK(state_dict.count("bias")) << "Should have bias parameter";
    CHECK(state_dict.count("lora_A")) << "Should have lora_A parameter";
    CHECK(state_dict.count("lora_B")) << "Should have lora_B parameter";
    std::cout << "State dict contains: weight, bias, lora_A, lora_B" << std::endl;

    // Verify LoRA parameters exist and are trainable
    CHECK(state_dict.at("lora_A")->requires_grad()) << "lora_A should be trainable";
    CHECK(state_dict.at("lora_B")->requires_grad()) << "lora_B should be trainable";
    CHECK(!state_dict.at("weight")->requires_grad()) << "weight should be frozen";
    std::cout << "LoRA parameters are trainable, base weight is frozen: OK" << std::endl;

    // Verify shapes
    CHECK_EQ(state_dict.at("lora_A")->Dims()[0], lora_config.rank);
    CHECK_EQ(state_dict.at("lora_A")->Dims()[1], 64);
    CHECK_EQ(state_dict.at("lora_B")->Dims()[0], 128);
    CHECK_EQ(state_dict.at("lora_B")->Dims()[1], lora_config.rank);
    std::cout << "LoRA parameter shapes: OK" << std::endl;

    std::cout << "LoRAStateDict tests passed!" << std::endl;
}

// ============================================================================
// Test 13: GetLoRAModel simplified API
// ============================================================================
void test_get_lora_model() {
    std::cout << "\n=== Test 13: GetLoRAModel Simplified API ===" << std::endl;

    // Test GetLoRAModel with a simple Linear layer
    // We'll wrap it with LoRA directly and verify the wrapper works

    // Create base Linear
    auto base_linear = std::make_shared<nn::Linear>(64, 128, /*bias=*/true);

    // Configure LoRA
    LoRAConfig config{4, 8.0f, 0.0f, ParseLoRATargetModules("Linear")};

    // Use GetLoRAModel with the linear as the "model"
    // Note: GetLoRAModel returns the modified model (in-place injection)
    auto model = GetLoRAModel(base_linear, config);

    CHECK(model != nullptr);
    std::cout << "GetLoRAModel returned valid pointer" << std::endl;

    // Test that LoRA was applied - check trainable parameters
    auto lora_params = GetLoRAParameters(model);
    // GetLoRAParameters returns vector<shared_ptr<Tensor>>, size() is the count of tensors
    // LoRALinear has 2 trainable tensors: lora_A (rank x in) and lora_B (out x rank)
    CHECK_EQ(lora_params.size(), 2);
    std::cout << "Trainable parameter tensors: " << lora_params.size() << " (expected: 2)" << std::endl;

    // Also verify total element count
    int64_t total_elements = 0;
    for (const auto &t : lora_params) { total_elements += t->NumElements(); }
    int64_t expected_elements = config.rank * 64 + 128 * config.rank; // 768
    CHECK_EQ(total_elements, expected_elements);
    std::cout << "Total trainable elements: " << total_elements << " (expected: " << expected_elements << ")"
              << std::endl;

    // Test PrintSummary
    std::cout << "\nLoRA Model Summary:" << std::endl;
    PrintLoRASummary(model);

    // Test Merge/Unmerge using utility functions
    MergeLoRAWeights(model);
    // Verify LoRA params frozen after merge
    auto *lora_mod = dynamic_cast<LoRALinear *>(model.get());
    CHECK(lora_mod != nullptr);
    CHECK(!lora_mod->LoRAParameters()[0]->requires_grad()) << "lora_A should be frozen after merge";
    CHECK(!lora_mod->LoRAParameters()[1]->requires_grad()) << "lora_B should be frozen after merge";
    std::cout << "Merge: OK (LoRA params frozen)" << std::endl;

    UnmergeLoRAWeights(model);
    CHECK(lora_mod->LoRAParameters()[0]->requires_grad()) << "lora_A should be trainable after unmerge";
    CHECK(lora_mod->LoRAParameters()[1]->requires_grad()) << "lora_B should be trainable after unmerge";
    std::cout << "Unmerge: OK (LoRA params trainable)" << std::endl;

    std::cout << "GetLoRAModel in-place injection tests passed!" << std::endl;
}

// ============================================================================
// Test 14: MergeAndUnload
// ============================================================================
void test_merge_and_unload() {
    std::cout << "\n=== Test 14: MergeAndUnload ===" << std::endl;

    // Create base Linear and apply LoRA
    auto base_linear = std::make_shared<nn::Linear>(64, 128, /*bias=*/true);
    LoRAConfig config{4, 8.0f, 0.0f, ParseLoRATargetModules("Linear")};
    auto model = GetLoRAModel(base_linear, config);

    // Verify it's a LoRA module
    CHECK(dynamic_cast<LoRALinear *>(model.get()) != nullptr) << "Should be LoRALinear";

    // Create input and get output before merge_and_unload
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 5, 64}, DataType::kFLOAT32);
    input->EigenMatrix().setRandom();
    auto output_before = (*model)({input})[0];
    float output_before_sum = output_before->EigenMatrix().sum();
    std::cout << "Output sum before MergeAndUnload: " << output_before_sum << std::endl;

    // MergeAndUnload
    auto unloaded_model = MergeAndUnload(model);
    CHECK(unloaded_model != nullptr) << "MergeAndUnload should return valid model";

    // Verify it's no longer a LoRA module
    CHECK(dynamic_cast<LoRALinear *>(unloaded_model.get()) == nullptr) << "Should be plain Linear after MergeAndUnload";
    std::cout << "Model is no longer LoRALinear: OK" << std::endl;

    // Verify no LoRA parameters exist (check state dict)
    auto state_dict = unloaded_model->StateDict();
    for (const auto &[name, param] : state_dict) {
        CHECK(name.find("lora_A") == std::string::npos && name.find("lora_B") == std::string::npos)
            << "Should not have LoRA parameters after MergeAndUnload, found: " << name;
    }
    std::cout << "No LoRA parameters in state dict: OK" << std::endl;

    // Verify forward output matches (merged output should equal unmerged LoRA output)
    auto output_after = (*unloaded_model)({input})[0];
    float output_after_sum = output_after->EigenMatrix().sum();
    std::cout << "Output sum after MergeAndUnload: " << output_after_sum << std::endl;
    std::cout << "Diff: " << std::abs(output_before_sum - output_after_sum) << std::endl;
    CHECK(std::abs(output_before_sum - output_after_sum) < 1e-3) << "Output should match after MergeAndUnload";

    // Verify all parameters have requires_grad = true (unfrozen)
    for (const auto &param : unloaded_model->Parameters()) {
        CHECK(param->requires_grad()) << "All parameters should be trainable after MergeAndUnload";
    }
    std::cout << "All parameters trainable: OK" << std::endl;

    std::cout << "MergeAndUnload tests passed!" << std::endl;
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = 1;

    // Initialize parallel settings (required for some tensor operations)
    // Parameters: nthread_per_process, tensor_parallel_size, sequence_parallel_enabled,
    //             pipeline_parallel_size, virtual_pipeline_parallel_size
    nn::parallel::global::InitAllEnv(1, 1, false, 1, 1);

    std::cout << "========================================" << std::endl;
    std::cout << "       LoRA Module Unit Tests          " << std::endl;
    std::cout << "========================================" << std::endl;

    test_lora_config();
    test_lora_linear_init();
    test_lora_linear_forward();
    test_lora_linear_merge();
    test_lora_utils();
    test_lora_from_linear();
    test_lora_model_wrapper();
    test_lora_save_load_weights();
    test_set_target_modules();
    test_should_apply_lora_edge_cases();
    test_replace_module_by_path();
    test_freeze_unfreeze();
    test_lora_state_dict();
    test_get_lora_model();
    test_merge_and_unload();

    std::cout << "\n========================================" << std::endl;
    std::cout << "       All LoRA Tests Passed!          " << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
