#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/autograd/activations.h"
#include "infini_train/include/autograd/elementwise.h"
#include "infini_train/include/autograd/function.h"
#include "infini_train/include/autograd/linear.h"
#include "infini_train/include/autograd/matmul.h"
#include "infini_train/include/autograd/normalization.h"
#include "infini_train/include/autograd/no_op.h"
#include "infini_train/include/autograd/outer.h"
#include "infini_train/include/autograd/reduction.h"
#include "infini_train/include/autograd/softmax.h"
#include "infini_train/include/autograd/transform.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

using namespace infini_train;

// ============================================================================
// Forward / Backward — CPU + CUDA
// ============================================================================

class AutogradForwardTest : public infini_train::test::InfiniTrainTest {};
class AutogradBackwardTest : public infini_train::test::InfiniTrainTest {};

class SaveOutputForBackwardFunction : public autograd::Function {
public:
    static constexpr char kType[] = "SaveOutputForBackwardFunction";

    SaveOutputForBackwardFunction() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        const auto &input = input_tensors[0];
        auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
        output->CopyFrom(input);
        return {output};
    }

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override {
        ctx_.SaveForBackward({output_tensors[0]});
    }

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        return {grad_outputs[0]};
    }

    const std::shared_ptr<Tensor> &saved_tensor() const { return ctx_.saved_tensors()[0]; }
};

class NeedsInputGradFunction : public autograd::Function {
public:
    static constexpr char kType[] = "NeedsInputGradFunction";

    NeedsInputGradFunction() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        const auto &input = input_tensors[0];
        auto output = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
        output->CopyFrom(input);
        return {output};
    }

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &) override {
        observed_needs_input_grad_ = ctx_.needs_input_grad();
        ctx_.SaveForBackward({input_tensors[0]});
    }

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        return {grad_outputs[0], nullptr};
    }

    const std::vector<bool> &observed_needs_input_grad() const { return observed_needs_input_grad_; }
    const std::shared_ptr<Tensor> &saved_tensor() const { return ctx_.saved_tensors()[0]; }

private:
    std::vector<bool> observed_needs_input_grad_;
};

class MarkNonDifferentiableFunction : public autograd::Function {
public:
    static constexpr char kType[] = "MarkNonDifferentiableFunction";

    MarkNonDifferentiableFunction() : autograd::Function(kType) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override {
        const auto &input = input_tensors[0];
        auto differentiable = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
        auto non_differentiable = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
        differentiable->CopyFrom(input);
        non_differentiable->CopyFrom(input);
        return {differentiable, non_differentiable};
    }

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override {
        ctx_.MarkNonDifferentiable({output_tensors[1]});
    }

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override {
        return {grad_outputs[0]};
    }
};

TEST_P(AutogradForwardTest, SavedOutputIsPackedWithoutAutogradMeta) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);

    auto fn = std::make_shared<SaveOutputForBackwardFunction>();
    auto outputs = fn->Apply({input});

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_NE(fn->saved_tensor(), nullptr);
    EXPECT_NE(fn->saved_tensor().get(), outputs[0].get());
    EXPECT_EQ(fn->saved_tensor()->DataPtr(), outputs[0]->DataPtr());
    EXPECT_FALSE(fn->saved_tensor()->requires_grad());
    EXPECT_EQ(fn->saved_tensor()->grad_fn(), nullptr);
}

TEST_P(AutogradForwardTest, FunctionCtxNeedsInputGradAndSaveForBackward) {
    auto requires_grad_input = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice(), true);
    auto no_grad_input = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice(), false);
    requires_grad_input->Fill(1.0f);
    no_grad_input->Fill(2.0f);

    auto fn = std::make_shared<NeedsInputGradFunction>();
    auto outputs = fn->Apply({requires_grad_input, no_grad_input});

    ASSERT_EQ(outputs.size(), 1);
    ASSERT_EQ(fn->observed_needs_input_grad().size(), 2);
    EXPECT_TRUE(fn->observed_needs_input_grad()[0]);
    EXPECT_FALSE(fn->observed_needs_input_grad()[1]);
    EXPECT_EQ(fn->saved_tensor().get(), requires_grad_input.get());
}

TEST_P(AutogradForwardTest, MarkNonDifferentiableOutput) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);

    auto outputs = std::make_shared<MarkNonDifferentiableFunction>()->Apply({input});

    ASSERT_EQ(outputs.size(), 2);
    EXPECT_TRUE(outputs[0]->requires_grad());
    EXPECT_NE(outputs[0]->grad_fn(), nullptr);
    EXPECT_FALSE(outputs[1]->requires_grad());
    EXPECT_EQ(outputs[1]->grad_fn(), nullptr);
    EXPECT_TRUE(outputs[1]->is_leaf());
}

TEST_P(AutogradForwardTest, AddForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(2.0f);
    auto result = std::make_shared<autograd::Add>()->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
}

TEST_P(AutogradForwardTest, SubForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(5.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(3.0f);
    auto result = std::make_shared<autograd::Sub>()->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, MulForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(2.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(3.0f);
    auto result = std::make_shared<autograd::Mul>()->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, DivForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(6.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(2.0f);
    auto result = std::make_shared<autograd::Div>()->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, NegForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(5.0f);
    auto result = std::make_shared<autograd::Neg>()->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, SinForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(0.0f);
    auto result = std::make_shared<autograd::Sin>()->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, CosForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(0.0f);
    auto result = std::make_shared<autograd::Cos>()->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, TanhForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(0.0f);
    auto result = std::make_shared<autograd::Tanh>()->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, ExpForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto result = std::make_shared<autograd::Exp>()->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, LogForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(2.0f);
    auto result = std::make_shared<autograd::Log>()->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, ReciprocalForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(2.0f);
    auto result = std::make_shared<autograd::Reciprocal>()->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, PowForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(2.0f);
    auto result = std::make_shared<autograd::Pow>(2.0f)->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, RsqrtForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(4.0f);
    auto result = std::make_shared<autograd::Rsqrt>()->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, SigmoidForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(0.0f);
    auto result = std::make_shared<autograd::Sigmoid>()->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, MatmulForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{3, 4}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(1.0f);
    auto result = std::make_shared<autograd::Matmul>()->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 4}));
}

TEST_P(AutogradForwardTest, SumForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto result = std::make_shared<autograd::Sum>(1, false)->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, MeanForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto result = std::make_shared<autograd::Mean>(1, false)->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, MaxForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto result = std::make_shared<autograd::Max>(1, false)->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, MinForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto result = std::make_shared<autograd::Min>(1, false)->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, SoftmaxForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto result = std::make_shared<autograd::Softmax>(1)->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
}

TEST_P(AutogradForwardTest, LayerNormForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3, 4}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.0f);
    auto result = std::make_shared<autograd::LayerNorm>(1e-5f)->Apply({a, weight, bias});
    EXPECT_EQ(result.size(), 3);
    EXPECT_FALSE(result[1]->requires_grad());
    EXPECT_EQ(result[1]->grad_fn(), nullptr);
    EXPECT_FALSE(result[2]->requires_grad());
    EXPECT_EQ(result[2]->grad_fn(), nullptr);
}

TEST_P(AutogradForwardTest, LinearForward) {
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    input->Fill(1.0f);
    auto weight = std::make_shared<Tensor>(std::vector<int64_t>{4, 3}, DataType::kFLOAT32, GetDevice(), true);
    weight->Fill(1.0f);
    auto bias = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    bias->Fill(0.0f);
    auto result = std::make_shared<autograd::Linear>()->Apply({input, weight, bias});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 4}));
}

TEST_P(AutogradForwardTest, TransposeForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto result = std::make_shared<autograd::Transpose>(0, 1)->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{3, 2}));
}

TEST_P(AutogradForwardTest, SliceForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{4, 4}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto result = std::make_shared<autograd::Slice>(std::vector<int64_t>{1, 1}, std::vector<int64_t>{3, 3},
                                                    std::vector<int64_t>{1, 1})
                      ->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, SplitForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{4, 4}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto result = std::make_shared<autograd::Split>(2, 0)->Apply({a});
    EXPECT_EQ(result.size(), 2);
}

TEST_P(AutogradForwardTest, ConcatForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 2}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(2.0f);
    auto result = std::make_shared<autograd::Concat>(0)->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{4, 2}));
}

TEST_P(AutogradForwardTest, StackForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(2.0f);
    auto result = std::make_shared<autograd::Stack>(0)->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 2, 3}));
}

TEST_P(AutogradForwardTest, TrilForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{3, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto result = std::make_shared<autograd::Tril>(0)->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, TriuForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{3, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto result = std::make_shared<autograd::Triu>(0)->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, OuterForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{4}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(1.0f);
    auto result = std::make_shared<autograd::Outer>()->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{3, 4}));
}

TEST_P(AutogradForwardTest, AddScalarForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto result = std::make_shared<autograd::AddScalar>(2.0f)->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, MulScalarForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(2.0f);
    auto result = std::make_shared<autograd::MulScalar>(3.0f)->Apply({a});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, LtForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(5.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(3.0f);
    auto result = std::make_shared<autograd::Lt>()->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, LeForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(3.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(3.0f);
    auto result = std::make_shared<autograd::Le>()->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, GtForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(5.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(3.0f);
    auto result = std::make_shared<autograd::Gt>()->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, GeForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(3.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(3.0f);
    auto result = std::make_shared<autograd::Ge>()->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, EqualsForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(3.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(3.0f);
    auto result = std::make_shared<autograd::Equals>()->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, AndForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(1.0f);
    auto result = std::make_shared<autograd::And>()->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, OrForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(0.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(1.0f);
    auto result = std::make_shared<autograd::Or>()->Apply({a, b});
    EXPECT_EQ(result.size(), 1);
}

TEST_P(AutogradForwardTest, NoOpForward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto result = std::make_shared<autograd::NoOp>(std::vector<int64_t>{2, 3})->Apply({a});
    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0]->Dims(), (std::vector<int64_t>{2, 3}));
}

TEST_P(AutogradBackwardTest, AddBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(1.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(2.0f);
    auto add_fn = std::make_shared<autograd::Add>();
    auto result = add_fn->Apply({a, b});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = add_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

TEST_P(AutogradBackwardTest, MulBackward) {
    auto a = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    a->Fill(2.0f);
    auto b = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    b->Fill(3.0f);
    auto mul_fn = std::make_shared<autograd::Mul>();
    auto result = mul_fn->Apply({a, b});
    auto grad = std::make_shared<Tensor>(std::vector<int64_t>{2, 3}, DataType::kFLOAT32, GetDevice(), true);
    grad->Fill(1.0f);
    auto grad_inputs = mul_fn->Backward({grad});
    EXPECT_EQ(grad_inputs.size(), 2);
}

INFINI_TRAIN_REGISTER_TEST(AutogradForwardTest);

INFINI_TRAIN_REGISTER_TEST(AutogradBackwardTest);
