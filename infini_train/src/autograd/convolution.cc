#include "infini_train/include/autograd/convolution.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Conv2d::Forward(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    CHECK(inputs.size() == 2 || inputs.size() == 3);
    CHECK_GT(stride_, 0);
    CHECK_GE(padding_, 0);
    for (const auto &t : inputs) {
        CHECK(t);
        CHECK(t->Dtype() == DataType::kFLOAT32) << "Conv2d supports FP32 only";
        CHECK(t->GetDevice() == inputs[0]->GetDevice()) << "Conv2d device mismatch";
        for (auto d : t->Dims()) { CHECK_GT(d, 0); }
    }
    const auto &x = inputs[0];
    const auto &w = inputs[1];
    CHECK_EQ(x->Dims().size(), 4) << "Conv2d expects NCHW input";
    CHECK_EQ(w->Dims().size(), 4) << "Conv2d expects OIHW weight";
    CHECK_EQ(x->Dims()[1], w->Dims()[1]);
    CHECK_GE(x->Dims()[2] + 2 * padding_, w->Dims()[2]);
    CHECK_GE(x->Dims()[3] + 2 * padding_, w->Dims()[3]);
    const auto bias = inputs.size() == 3 ? inputs[2] : nullptr;
    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], w->Dims()[0]);
    }
    return {Dispatcher::Instance().Call<std::shared_ptr<Tensor>>({x->GetDevice().type(), "Conv2dForward"}, x, w, bias,
                                                                 stride_, padding_)};
}

void Conv2d::SetupContext(const std::vector<std::shared_ptr<Tensor>> &inputs,
                          const std::vector<std::shared_ptr<Tensor>> &) {
    ctx_.SaveForBackward({inputs[0], inputs[1]});
    bias_ = inputs.size() == 3;
}

std::vector<std::shared_ptr<Tensor>> Conv2d::Backward(const std::vector<std::shared_ptr<Tensor>> &grads) {
    CHECK_EQ(grads.size(), 1);
    CHECK(grads[0]);
    const auto saved = ctx_.GetSavedTensors();
    const auto &x = saved[0];
    const auto &w = saved[1];
    const auto &dy = grads[0];
    const std::vector<int64_t> shape
        = {x->Dims()[0], w->Dims()[0], (x->Dims()[2] + 2 * padding_ - w->Dims()[2]) / stride_ + 1,
           (x->Dims()[3] + 2 * padding_ - w->Dims()[3]) / stride_ + 1};
    CHECK(dy->Dims() == shape);
    CHECK(dy->Dtype() == DataType::kFLOAT32);
    CHECK(dy->GetDevice() == x->GetDevice());
    const auto &needs = ctx_.needs_input_grad();
    CHECK_EQ(needs.size(), bias_ ? 3 : 2);
    auto result = Dispatcher::Instance().Call<std::vector<std::shared_ptr<Tensor>>>(
        {x->GetDevice().type(), "Conv2dBackward"}, x, w, dy, stride_, padding_, static_cast<bool>(needs[0]),
        static_cast<bool>(needs[1]), bias_ && needs[2]);
    if (!bias_) {
        result.resize(2);
    }
    return result;
}
} // namespace infini_train::autograd
