#include <stddef.h>
#include <stdint.h>

#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/utils/checkpoint.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {
// A custom Function may implement its forward using other autograd Functions.
// Both saved tensors are needed to detect a shifted pack/unpack sequence.
class NestedSquare : public autograd::Function {
 public:
  explicit NestedSquare(bool nested_setup = false)
      : Function("NestedSquare"), nested_setup_(nested_setup) {}

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &inputs) override {
    return {inputs[0] * inputs[0]};
  }

  void SetupContext(
      const std::vector<std::shared_ptr<Tensor>> &inputs,
      const std::vector<std::shared_ptr<Tensor>> &outputs) override {
    auto square = nested_setup_ ? inputs[0] * inputs[0] : outputs[0];
    ctx_.SaveForBackward({inputs[0], square});
  }

  std::vector<std::shared_ptr<Tensor>> Backward(
      const std::vector<std::shared_ptr<Tensor>> &grads) override {
    auto saved = ctx_.GetSavedTensors();
    // Test inputs are nonzero. This equals 2*x and uses both saved tensors.
    return {grads[0] * (saved[1] / saved[0]) * 2.0f};
  }

 private:
  bool nested_setup_;
};

class SaveScalarAndNull : public autograd::Function {
 public:
  SaveScalarAndNull() : Function("SaveScalarAndNull") {}

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &inputs) override {
    auto out = std::make_shared<Tensor>(inputs[0]->Dims(), inputs[0]->Dtype(),
                                        inputs[0]->GetDevice());
    out->CopyFrom(inputs[0]);
    return {out};
  }

  void SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                    const std::vector<std::shared_ptr<Tensor>> &) override {
    ctx_.SaveForBackward({nullptr, inputs[0], nullptr});
  }

  std::vector<std::shared_ptr<Tensor>> Backward(
      const std::vector<std::shared_ptr<Tensor>> &grads) override {
    auto saved = ctx_.GetSavedTensors();
    EXPECT_EQ(saved.size(), 3);
    EXPECT_EQ(saved.at(0), nullptr);
    EXPECT_EQ(saved.at(2), nullptr);
    EXPECT_TRUE(saved.at(1)->Dims().empty());
    test::ExpectTensorFloatEqual(saved.at(1), 3.0f);
    return {grads[0]};
  }
};
}  // namespace

class ActivationCheckpointTest : public test::InfiniTrainTest {};

TEST_P(ActivationCheckpointTest,
       NestedFunctionsPreserveSavedTensorsAndGradients) {
  for (bool nested_setup : {false, true}) {
    for (bool early_stop : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "nested_setup=" << nested_setup
                                        << " early_stop=" << early_stop);
      const float data[] = {2.0f, 3.0f};
      auto input = std::make_shared<Tensor>(data, std::vector<int64_t>{2},
                                            DataType::kFLOAT32, GetDevice());
      input->RequiresGrad();
      auto forward = [nested_setup](const auto &inputs) {
        return std::make_shared<NestedSquare>(nested_setup)->Apply(inputs);
      };
      auto outputs = utils::checkpoint::Checkpoint(forward, {input}, false,
                                                   false, true, early_stop);
      test::ExpectTensorFloatEqual(outputs[0], {4.0f, 9.0f});
      auto gradient = std::make_shared<Tensor>(input->Dims(),
                                               DataType::kFLOAT32, GetDevice());
      gradient->Fill(1.0f);
      outputs[0]->Backward(gradient);
      test::ExpectTensorNear(input->grad(), {4.0f, 6.0f}, 1e-6f);
      EXPECT_TRUE(autograd::GradMode::IsEnabled());
      EXPECT_FALSE(autograd::GradMode::PropagateRequiresGrad());
    }
  }
}

TEST_P(ActivationCheckpointTest, ScalarAndNullSavedEntriesRemainDistinct) {
  for (bool determinism_check : {false, true}) {
    auto input = std::make_shared<Tensor>(
        std::vector<int64_t>{}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(3.0f);
    auto forward = [](const auto &inputs) {
      return std::make_shared<SaveScalarAndNull>()->Apply(inputs);
    };
    auto outputs = utils::checkpoint::Checkpoint(forward, {input}, false, false,
                                                 determinism_check);
    outputs[0]->Backward();
    test::ExpectTensorFloatEqual(input->grad(), 1.0f);
  }
}

TEST_P(ActivationCheckpointTest, PropagationDoesNotLeakIntoNestedFunctions) {
  for (bool grad_enabled : {false, true}) {
    for (bool propagate : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "grad_enabled=" << grad_enabled
                                        << " propagate=" << propagate);
      auto input = std::make_shared<Tensor>(
          std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice(), true);
      input->Fill(2.0f);
      size_t packed = 0;
      autograd::FunctionCtx::SavedTensorHooks hooks;
      hooks.pack = [&packed](const auto &tensor) -> std::shared_ptr<void> {
        ++packed;
        return tensor;
      };
      hooks.unpack = [](const auto &state) {
        return std::static_pointer_cast<Tensor>(state);
      };
      autograd::FunctionCtx::SavedTensorHooksGuard hooks_guard(
          std::move(hooks));
      autograd::NoGradGuard no_grad;
      autograd::GradMode::SetEnabled(grad_enabled);
      autograd::PropagateRequiresGradGuard propagation(propagate);
      auto outputs = std::make_shared<NestedSquare>(true)->Apply({input});
      EXPECT_EQ(packed, grad_enabled || propagate ? 2 : 0);
      EXPECT_EQ(outputs[0]->requires_grad(), grad_enabled || propagate);
      EXPECT_EQ(outputs[0]->grad_fn() != nullptr, grad_enabled);
    }
  }
}

TEST(ActivationCheckpointGuardTest, RestoresPropagationAfterException) {
  EXPECT_FALSE(autograd::GradMode::PropagateRequiresGrad());
  {
    autograd::PropagateRequiresGradGuard enabled;
    try {
      autograd::PropagateRequiresGradGuard disabled(false);
      EXPECT_FALSE(autograd::GradMode::PropagateRequiresGrad());
      throw std::runtime_error("test");
    } catch (const std::runtime_error &) {
    }
    EXPECT_TRUE(autograd::GradMode::PropagateRequiresGrad());
  }
  EXPECT_FALSE(autograd::GradMode::PropagateRequiresGrad());
}

INFINI_TRAIN_REGISTER_TEST(ActivationCheckpointTest);
