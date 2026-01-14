#include <iostream>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/utils/precision_check_config.h"

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

void TestModuleLevel() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  Module-Level Test" << std::endl;
    std::cout << "========================================" << std::endl;

    auto model = std::make_shared<MyModel>();
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32);
    x->Fill<float>(2.0f);
    x->RequiresGrad();

    std::vector<std::shared_ptr<Tensor>> inputs = {x};
    auto outputs = (*model)(inputs);
    auto loss = outputs[0]->Sum(0, false)->Sum(0, false);
    loss->Backward();

    std::cout << "Test completed." << std::endl;
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    std::string config_str = argc > 1 ? argv[1] : "level=2";

    std::cout << "========================================" << std::endl;
    std::cout << "  Precision Check Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Config: " << config_str << std::endl;

    auto config = utils::PrecisionCheckConfig::Parse(config_str);
    nn::parallel::global::InitAllEnv(1, 1, false, 1, 1, config);

    if (config.level == 1) {
        TestModuleLevel();
    } else if (config.level == 2) {
        TestFunctionLevel(config_str);
    } else {
        std::cout << "No tests to run (level=0)" << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  All Tests Completed Successfully" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
