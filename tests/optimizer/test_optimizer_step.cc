#include <stdint.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/common/hook.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

class OptimizerStepTest : public infini_train::test::InfiniTrainTest {};

TEST_P(OptimizerStepTest, SGDStep) {
  auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3},
                                        DataType::kFLOAT32, GetDevice());
  param->set_requires_grad(true);
  param->Fill(1.0f);
  auto optimizer = std::make_shared<optimizers::SGD>(
      std::vector<std::shared_ptr<Tensor>>{param}, 0.01);
  optimizer->ZeroGrad();
  optimizer->Step();
}

TEST_P(OptimizerStepTest, AdamStep) {
  auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3},
                                        DataType::kFLOAT32, GetDevice());
  param->set_requires_grad(true);
  param->Fill(1.0f);
  auto optimizer = std::make_shared<optimizers::Adam>(
      std::vector<std::shared_ptr<Tensor>>{param}, 0.001);
  optimizer->ZeroGrad();
  optimizer->Step();
}

TEST_P(OptimizerStepTest, StepPostHookRunsAfterAdamStep) {
  auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3},
                                        DataType::kFLOAT32, GetDevice());
  param->set_requires_grad(true);
  param->Fill(1.0f);
  auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3},
                                       DataType::kFLOAT32, GetDevice());
  grad->Fill(0.25f);
  param->set_grad(grad);

  auto optimizer = std::make_shared<optimizers::Adam>(
      std::vector<std::shared_ptr<Tensor>>{param}, 0.001);
  int first_hook_calls = 0;
  int second_hook_calls = 0;
  std::vector<int> call_order;
  auto first_handle =
      optimizer->RegisterPostStepHook([&](Optimizer &hook_optimizer) {
        ++first_hook_calls;
        call_order.push_back(1);
        EXPECT_EQ(&hook_optimizer, optimizer.get());
        const auto state = hook_optimizer.StateDict();
        EXPECT_EQ(*static_cast<const int64_t *>(state.at("adam.t")->DataPtr()),
                  first_hook_calls);
      });
  auto second_handle =
      optimizer->RegisterPostStepHook([&](Optimizer &hook_optimizer) {
        ++second_hook_calls;
        call_order.push_back(2);
        EXPECT_EQ(&hook_optimizer, optimizer.get());
        const auto state = hook_optimizer.StateDict();
        EXPECT_EQ(*static_cast<const int64_t *>(state.at("adam.t")->DataPtr()),
                  second_hook_calls);
      });
  optimizer->Step();
  EXPECT_EQ(first_hook_calls, 1);
  EXPECT_EQ(second_hook_calls, 1);
  EXPECT_EQ(call_order, (std::vector<int>{1, 2}));

  first_handle->Remove();
  optimizer->Step();
  EXPECT_EQ(first_hook_calls, 1);
  EXPECT_EQ(second_hook_calls, 2);
  EXPECT_EQ(call_order, (std::vector<int>{1, 2, 2}));

  second_handle->Remove();
}

TEST_P(OptimizerStepTest, ZeroGrad) {
  auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3},
                                        DataType::kFLOAT32, GetDevice());
  param->set_requires_grad(true);
  auto optimizer = std::make_shared<optimizers::SGD>(
      std::vector<std::shared_ptr<Tensor>>{param}, 0.01);
  optimizer->ZeroGrad();
}

TEST_P(OptimizerStepTest, ZeroGradWithNone) {
  auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3},
                                        DataType::kFLOAT32, GetDevice());
  param->set_requires_grad(true);
  auto optimizer = std::make_shared<optimizers::SGD>(
      std::vector<std::shared_ptr<Tensor>>{param}, 0.01);
  optimizer->ZeroGrad(false);
}

TEST_P(OptimizerStepTest, SGDMultiParams) {
  std::vector<std::shared_ptr<Tensor>> params;
  for (int i = 0; i < 3; ++i) {
    auto param = std::make_shared<Tensor>(std::vector<int64_t>{2, 3},
                                          DataType::kFLOAT32, GetDevice());
    param->set_requires_grad(true);
    params.push_back(param);
  }
  auto optimizer = std::make_shared<optimizers::SGD>(params, 0.01);
  EXPECT_NE(optimizer, nullptr);
  optimizer->ZeroGrad();
}

INFINI_TRAIN_REGISTER_TEST(OptimizerStepTest);
