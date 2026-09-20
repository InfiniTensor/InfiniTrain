#include "infini_train/include/autograd/accumulate.h"

#include "glog/logging.h"

#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/sparse_row_grad.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
AccumulateGrad::AccumulateGrad(std::shared_ptr<Tensor> tensor, float learning_rate)
    : tensor_(tensor), learning_rate_(learning_rate) {}

std::vector<std::shared_ptr<Tensor>> AccumulateGrad::Forward(const std::vector<std::shared_ptr<Tensor>> &) {
    LOG(FATAL) << "AccumulateGrad::Forward shall not be called directly!";
    return {};
}

std::vector<std::shared_ptr<Tensor>>
AccumulateGrad::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    auto grad_output = grad_outputs[0];

    auto device = tensor_->GetDevice();
    core::DeviceGuard guard(device);

    if (grad_output) {
        if (grad_output->Dtype() != tensor_->Dtype()) {
            LOG(WARNING) << "AccumulateGrad: grad dtype (" << kDataTypeToDesc.at(grad_output->Dtype())
                         << ") does not match parameter dtype (" << kDataTypeToDesc.at(tensor_->Dtype())
                         << "). This indicates a dtype mismatch in the autograd graph (e.g. autocast "
                            "running before autograd). The grad is not cast and will be used as-is.";
        }

        const bool overwrite = tensor_->ConsumeGradOverwriteFlag();
        auto pre_hook = tensor_->pre_accumulate_grad_hook();
        if (pre_hook) {
            if (pre_hook->TryBypassAccumulate(tensor_, grad_output, overwrite, learning_rate_)) {
                tensor_->ResetAccumulator();
                return {};
            }
            (*pre_hook)(grad_output);
        }

        auto grad = tensor_->grad();
        auto *sparse_state = GetSparseRowGradState(tensor_.get());
        if (grad) {
            if (sparse_state && grad->DataPtr() == grad_output->DataPtr()) {
                // Sparse embedding path: grad_output is a view of the persistent buffer the
                // embedding backward already scatter-added into (learning_rate_ is 1.0f for
                // AccumulateGrad nodes), so accumulating it again would double count.
            } else {
                if (overwrite) {
                    // If the tensor is marked to overrite its current grad on next grad update
                    // See notes in `infini_train::nn::parallel::Reducer::PrepareForBackward()`
                    // NOTE(zbl): must copy, cannot change grad buffer address
                    grad->CopyFrom(grad_output);
                } else {
                    auto kernel = Dispatcher::Instance().GetKernel({device.type(), "AccumulateGrad"});
                    kernel.Call<void>(grad_output, learning_rate_, grad);
                }
                if (sparse_state) {
                    // A dense grad from another producer (matmul on a tied weight, DDP bucket)
                    // landed in the storage; from here on the sparse optimizer shortcut is off.
                    sparse_state->poisoned = true;
                    if (grad->DataPtr() != sparse_state->grad_buffer->DataPtr()) {
                        // The accumulator belongs to that other producer: grad_output was the
                        // cumulative sparse buffer and the add/copy above merged it in full.
                        // Flush it so the next micro-batch contributes only its own delta.
                        ClearSparseRowGradRows(sparse_state);
                    }
                }
            }
        } else {
            // FIXME(zbl): check whether need to do copying instead of slicing
            auto new_grad = std::make_shared<Tensor>(*grad_output.get(), 0, grad_output->Dims());
            tensor_->set_grad(new_grad);
        }
        auto post_hook = tensor_->post_accumulate_grad_hook();
        if (post_hook != nullptr) {
            (*post_hook)(tensor_->grad());
        }
        tensor_->ResetAccumulator();
    }
    return {};
}
} // namespace infini_train::autograd
