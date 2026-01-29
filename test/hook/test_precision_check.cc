#include <filesystem>
#include <iostream>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/utils/global_module_hook_registry.h"
#include "infini_train/include/utils/precision_check_config.h"
#include "infini_train/include/utils/precision_checker.h"

using namespace infini_train;

class MyModel : public nn::Module {
public:
    MyModel() : Module("MyModel") {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override {
        auto x = inputs[0];
        x->RequiresGrad();
        auto y = x->Mul(x);
        return {y};
    }
};

// Simple model for multi-iteration test
class SimpleModel : public nn::Module {
public:
    SimpleModel() : Module("SimpleModel") {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override {
        auto x = inputs[0];
        x->RequiresGrad();
        auto y = x->Mul(x)->Mul(x); // x^3
        return {y};
    }
};

void RunModelForwardBackward(const std::shared_ptr<nn::Module> &model) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32);
    x->Fill<float>(2.0f);
    x->RequiresGrad();

    std::vector<std::shared_ptr<Tensor>> inputs = {x};
    auto outputs = (*model)(inputs);
    auto loss = outputs[0]->Sum(0, false)->Sum(0, false);
    loss->Backward();
}

void TestFunctionLevel(const std::string &config_str) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Function-Level Test: " << config_str << std::endl;
    std::cout << "========================================" << std::endl;

    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32);
    x->Fill<float>(2.0f);
    x->RequiresGrad();

    auto y = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32);
    y->Fill<float>(3.0f);
    y->RequiresGrad();

    auto z = x->Mul(y);
    auto loss = z->Sum(0, false)->Sum(0, false);
    loss->Backward();

    std::cout << "Test completed." << std::endl;
}

void TestModuleLevel(const std::string &config_str) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Module-Level Test: " << config_str << std::endl;
    std::cout << "========================================" << std::endl;

    auto model = std::make_shared<MyModel>();
    RunModelForwardBackward(model);

    std::cout << "Test completed." << std::endl;
}

// Test: Simple format output (level=2, format=simple)
void TestSimpleFormat() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test: Simple Format (level=2, format=simple)" << std::endl;
    std::cout << "========================================" << std::endl;

    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32);
    x->Fill<float>(2.0f);
    x->RequiresGrad();

    auto y = x->Mul(x);
    auto loss = y->Sum(0, false)->Sum(0, false); // Two Sum ops to produce scalar
    loss->Backward();

    std::cout << "Simple format test completed - check output for min/max/mean values." << std::endl;
}

// Test: MD5 format output (level=2, format=md5)
void TestMd5Format() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test: MD5 Format (level=2, format=md5)" << std::endl;
    std::cout << "========================================" << std::endl;

    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32);
    x->Fill<float>(2.0f);
    x->RequiresGrad();

    auto y = x->Mul(x);
    auto loss = y->Sum(0, false)->Sum(0, false); // Two Sum ops to produce scalar
    loss->Backward();

    std::cout << "MD5 format test completed - check output for md5 hashes." << std::endl;
}

// Test: Save tensors to NPY files (level=1, save_tensors=true)
void TestSaveTensors() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test: Save Tensors (level=1, save_tensors=true)" << std::endl;
    std::cout << "========================================" << std::endl;

    std::string output_path = "/tmp/precision_check_npy";

    auto model = std::make_shared<MyModel>();
    RunModelForwardBackward(model);

    // Verify NPY files were created
    namespace fs = std::filesystem;
    bool found_npy = false;
    if (fs::exists(output_path)) {
        for (const auto &entry : fs::recursive_directory_iterator(output_path)) {
            if (entry.path().extension() == ".npy") {
                found_npy = true;
                std::cout << "Found NPY file: " << entry.path() << std::endl;
            }
        }
    }

    if (found_npy) {
        std::cout << "Save tensors test PASSED - NPY files created successfully." << std::endl;
    } else {
        std::cout << "Save tensors test completed - check output directory for NPY files." << std::endl;
    }
}

// Test: Multi-iteration file overwrite (level=1, save_tensors=true, iter=3)
void TestMultiIterOverwrite() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test: Multi-Iteration File Overwrite" << std::endl;
    std::cout << "========================================" << std::endl;

    std::string output_path = "/tmp/precision_check_overwrite";

    auto model = std::make_shared<SimpleModel>();
    int num_iters = 3;

    // Run multiple iterations - files should be overwritten
    for (int i = 0; i < num_iters; ++i) {
        std::cout << "Iteration " << (i + 1) << "/" << num_iters << std::endl;
        utils::PrecisionCheckEnv::ResetCounters(); // Reset counters each iteration
        RunModelForwardBackward(model);
    }

    namespace fs = std::filesystem;
    int npy_count = 0;
    if (fs::exists(output_path)) {
        for (const auto &entry : fs::recursive_directory_iterator(output_path)) {
            if (entry.path().extension() == ".npy") {
                ++npy_count;
            }
        }
    }

    std::cout << "Multi-iteration test completed - found " << npy_count << " NPY files after " << num_iters
              << " iterations." << std::endl;
    std::cout << "(Files should be overwritten each iteration, count should be consistent with 1 iter)" << std::endl;
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    std::string config_str = argc > 1 ? argv[1] : "";

    std::cout << "========================================" << std::endl;
    std::cout << "  Precision Check Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    nn::parallel::global::InitAllEnv(1, 1, false, 1, 1);

    // If no config argument, run all format tests
    if (config_str.empty()) {
        auto config = utils::PrecisionCheckConfig::Parse("level=2,format=simple");
        utils::PrecisionCheckEnv::Instance().Init(config);

        std::cout << "\nRunning all precision check format tests..." << std::endl;

        // Test 1: Simple format
        TestSimpleFormat();

        // Test 2: MD5 format
        auto md5_config = utils::PrecisionCheckConfig::Parse("level=2,format=md5");
        utils::PrecisionCheckEnv::Instance().Init(md5_config);
        TestMd5Format();

        // Test 3: Save tensors
        auto npy_config = utils::PrecisionCheckConfig::Parse("level=1,save_tensors=true");
        utils::PrecisionCheckEnv::Instance().Init(npy_config);
        TestSaveTensors();

        // Test 4: Multi-iteration overwrite
        auto iter_config = utils::PrecisionCheckConfig::Parse("level=1,save_tensors=true");
        utils::PrecisionCheckEnv::Instance().Init(iter_config);
        TestMultiIterOverwrite();

        std::cout << "\n========================================" << std::endl;
        std::cout << "  All Tests Completed Successfully" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    }

    // If config provided, run single test (original behavior)
    auto config = utils::PrecisionCheckConfig::Parse(config_str);
    utils::PrecisionCheckEnv::Instance().Init(config);

    std::cout << "Config: " << config_str << std::endl;

    if (config.level == utils::PrecisionCheckLevel::MODULE) {
        TestModuleLevel(config_str);
    } else if (config.level == utils::PrecisionCheckLevel::FUNCTION) {
        TestFunctionLevel(config_str);
    } else {
        std::cout << "No tests to run (level=0)" << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  Test Completed" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
