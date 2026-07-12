#include "infini_train/include/core/runtime/dropout_kernels.h"

#include "infini_train/include/core/runtime/dropout_stubs.h"
#include "infini_train/include/tensor.h"

namespace infini_train {

DEFINE_DISPATCH(dropout_forward_stub);
DEFINE_DISPATCH(dropout_backward_stub);

namespace {

void check_dropout_probability(double p) {
    CHECK_GE(p, 0.0) << "dropout probability has to be between 0 and 1, but got " << p;
    CHECK_LE(p, 1.0) << "dropout probability has to be between 0 and 1, but got " << p;
}

void check_dropout_tensors(const Tensor &output, const Tensor &mask, const Tensor &input) {
    CHECK(IsFloatingPointDType(input.Dtype())) << "Dropout supports floating-point tensors only";
    CHECK_EQ(static_cast<int>(output.Dtype()), static_cast<int>(input.Dtype()));
    CHECK(output.GetDevice() == input.GetDevice());
    CHECK(output.Dims() == input.Dims());
    CHECK_EQ(static_cast<int>(mask.Dtype()), static_cast<int>(DataType::kUINT8));
    CHECK(mask.GetDevice() == input.GetDevice());
    CHECK(mask.Dims() == input.Dims());
}

} // namespace

void dropout_forward_kernel(Tensor &output, Tensor &mask, const Tensor &input, double p,
                            const std::optional<Generator> &generator) {
    check_dropout_probability(p);
    check_dropout_tensors(output, mask, input);
    dropout_forward_stub(input.GetDevice().type(), output, mask, input, p, generator);
}

void dropout_backward_kernel(Tensor &grad_input, const Tensor &grad_output, const Tensor &mask, double p) {
    check_dropout_probability(p);
    check_dropout_tensors(grad_input, mask, grad_output);
    dropout_backward_stub(grad_output.GetDevice().type(), grad_input, grad_output, mask, p);
}

} // namespace infini_train
