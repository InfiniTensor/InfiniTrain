#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/utils/precision_check_config.h"
#include "infini_train/include/utils/precision_checker.h"
#include "test_utils.h"

using namespace infini_train;

class PrecisionCheckTest : public infini_train::test::InfiniTrainTestP {};

class SimpleModel : public nn::Module {
public:
    SimpleModel() : Module("SimpleModel") {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) override {
        auto x = inputs[0];
        x->RequiresGrad();
        auto y = x->Mul(x)->Mul(x);
        return {y};
    }
};

TEST_P(PrecisionCheckTest, SimpleFormat) {
    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    x->Fill<float>(2.0f);
    x->RequiresGrad();

    auto y = x->Mul(x);
    auto loss = y->Sum(0, false)->Sum(0, false);
    loss->Backward();

    EXPECT_NE(x->DataPtr(), nullptr);
}

TEST_P(PrecisionCheckTest, ModuleForwardBackward) {
    auto model = std::make_shared<SimpleModel>();

    auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
    x->Fill<float>(2.0f);
    x->RequiresGrad();

    std::vector<std::shared_ptr<Tensor>> inputs = {x};
    auto outputs = (*model)(inputs);
    auto loss = outputs[0]->Sum(0, false)->Sum(0, false);
    loss->Backward();

    EXPECT_TRUE(x->requires_grad());
}

TEST_P(PrecisionCheckTest, MultiIteration) {
    auto model = std::make_shared<SimpleModel>();

    for (int i = 0; i < 3; ++i) {
        auto x = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice());
        x->Fill<float>(2.0f);
        x->RequiresGrad();

        std::vector<std::shared_ptr<Tensor>> inputs = {x};
        auto outputs = (*model)(inputs);
        auto loss = outputs[0]->Sum(0, false)->Sum(0, false);
        loss->Backward();
    }

    SUCCEED();
}

INFINI_TRAIN_REGISTER_TEST(PrecisionCheckTest);
