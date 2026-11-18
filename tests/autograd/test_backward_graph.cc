#include <stdint.h>

#include <initializer_list>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/autograd/topk.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/utils/checkpoint.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {

class StopGradient : public autograd::Function {
 public:
  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &inputs) override {
    return {inputs[0]->Mul(2.0f)};
  }

  std::vector<std::shared_ptr<Tensor>> Backward(
      const std::vector<std::shared_ptr<Tensor>> &) override {
    return {nullptr};
  }
};

}  // namespace

class BackwardGraphTest : public test::InfiniTrainTest {};

TEST_P(BackwardGraphTest, IgnoresUnusedLeafConsumers) {
  for (bool destroy_unused : {false, true}) {
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(2.0f);
    auto unused = input->Mul(3.0f);
    if (destroy_unused) {
      unused.reset();
    }
    input->Sum(0)->Backward();
    test::ExpectTensorFloatEqual(input->grad(), {1.0f, 1.0f});
  }
}

TEST_P(BackwardGraphTest, IgnoresUnusedIntermediateConsumers) {
  auto input = std::make_shared<Tensor>(std::vector<int64_t>{2},
                                        DataType::kFLOAT32, GetDevice(), true);
  input->Fill(3.0f);
  auto intermediate = input->Mul(2.0f);
  auto unused = intermediate->Mul(7.0f);
  intermediate->Sum(0)->Backward();
  test::ExpectTensorFloatEqual(input->grad(), {2.0f, 2.0f});
}

TEST_P(BackwardGraphTest, FillsUnusedSplitGradientsAfterOutputsAreDestroyed) {
  const std::vector<std::vector<float>> expected{
      {1.0f, 1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 0.0f, 1.0f}};
  for (int output = 0; output < 3; ++output) {
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{5}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(2.0f);
    auto selected = input->Split(2, 0)[output];
    selected->Sum(0)->Backward();
    test::ExpectTensorFloatEqual(input->grad(), expected[output]);
  }
}

TEST_P(BackwardGraphTest, AccumulatesAllUsedSplitOutputs) {
  auto input = std::make_shared<Tensor>(std::vector<int64_t>{5},
                                        DataType::kFLOAT32, GetDevice(), true);
  input->Fill(2.0f);
  auto outputs = input->Split(2, 0);
  auto loss = outputs[0]->Sum(0)->Mul(2.0f);
  loss = loss->Add(outputs[1]->Sum(0)->Mul(3.0f));
  loss = loss->Add(outputs[2]->Sum(0)->Mul(4.0f));
  loss->Backward();
  test::ExpectTensorFloatEqual(input->grad(), {2.0f, 2.0f, 3.0f, 3.0f, 4.0f});
}

TEST_P(BackwardGraphTest, CountsRepeatedEdgesToSharedIntermediate) {
  auto input = std::make_shared<Tensor>(std::vector<int64_t>{2},
                                        DataType::kFLOAT32, GetDevice(), true);
  input->Fill(3.0f);
  auto intermediate = input->Mul(2.0f);
  auto unused = intermediate->Mul(7.0f);
  intermediate->Mul(intermediate)->Sum(0)->Backward();
  test::ExpectTensorFloatEqual(input->grad(), {24.0f, 24.0f});
}

TEST_P(BackwardGraphTest, NullGradientCompletesAReachableDependency) {
  auto input = std::make_shared<Tensor>(std::vector<int64_t>{2},
                                        DataType::kFLOAT32, GetDevice(), true);
  input->Fill(3.0f);
  auto intermediate = input->Mul(2.0f);
  auto stopped = std::make_shared<StopGradient>()->Apply({intermediate})[0];
  intermediate->Mul(3.0f)->Add(stopped)->Sum(0)->Backward();
  test::ExpectTensorFloatEqual(input->grad(), {6.0f, 6.0f});
}

TEST_P(BackwardGraphTest, NullOnlyGradientDoesNotCreateLeafGradient) {
  std::weak_ptr<Tensor> weak_input;
  {
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice(), true);
    weak_input = input;
    input->Fill(3.0f);
    auto intermediate = input->Mul(2.0f);
    auto stopped = std::make_shared<StopGradient>()->Apply({intermediate})[0];
    stopped->Sum(0)->Backward();
    EXPECT_EQ(input->grad(), nullptr);
  }
  EXPECT_TRUE(weak_input.expired());
}

TEST_P(BackwardGraphTest, KeepsNonDifferentiableOutputGradientNull) {
  const float data[] = {1.0f, 3.0f, 2.0f};
  auto input = std::make_shared<Tensor>(data, std::vector<int64_t>{3},
                                        DataType::kFLOAT32, GetDevice());
  input->RequiresGrad();
  auto function = std::make_shared<autograd::TopK>(1, 0, true, true);
  bool called = false;
  auto handle = function->RegisterBackwardPreHook(
      [&called](autograd::Function *, const auto &gradients) {
        called = true;
        ASSERT_EQ(gradients.size(), 2);
        EXPECT_EQ(gradients[1], nullptr);
      });
  auto outputs = function->Apply({input});
  outputs[0]->Sum(0)->Backward();
  EXPECT_TRUE(called);
  test::ExpectTensorFloatEqual(input->grad(), {0.0f, 1.0f, 0.0f});
}

TEST_P(BackwardGraphTest, HandlesUnusedBranchesInCheckpointRecomputation) {
  for (bool reentrant : {false, true}) {
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(2.0f);
    auto forward = [](const std::vector<std::shared_ptr<Tensor>> &inputs) {
      auto unused = inputs[0]->Mul(7.0f);
      return std::vector<std::shared_ptr<Tensor>>{inputs[0]->Mul(inputs[0])};
    };
    auto outputs = utils::checkpoint::Checkpoint(forward, {input}, reentrant,
                                                 false, true, true);
    outputs[0]->Sum(0)->Backward();
    test::ExpectTensorFloatEqual(input->grad(), {4.0f, 4.0f});
  }
}

TEST_P(BackwardGraphTest, PreservesPendingGradientsDuringReentrantBackward) {
  auto input = std::make_shared<Tensor>(std::vector<int64_t>{},
                                        DataType::kFLOAT32, GetDevice(), true);
  auto weight = std::make_shared<Tensor>(std::vector<int64_t>{},
                                         DataType::kFLOAT32, GetDevice(), true);
  input->Fill(2.0f);
  weight->Fill(3.0f);
  auto before = weight->Mul(2.0f);
  auto forward = [weight](const std::vector<std::shared_ptr<Tensor>> &inputs) {
    return std::vector<std::shared_ptr<Tensor>>{inputs[0]->Mul(weight)};
  };
  auto outputs =
      utils::checkpoint::Checkpoint(forward, {input}, true, false, true, true);
  auto after = weight->Mul(5.0f);
  before->Add(outputs[0])->Add(after)->Backward();
  test::ExpectTensorFloatEqual(input->grad(), {3.0f});
  test::ExpectTensorFloatEqual(weight->grad(), {9.0f});
}

INFINI_TRAIN_REGISTER_TEST(BackwardGraphTest);
