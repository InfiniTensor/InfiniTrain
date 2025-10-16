#include "infini_train/include/autograd/function.h"

#include "glog/logging.h"

#include "infini_train/include/autograd/accumulate.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include <memory>

namespace infini_train::autograd {

std::vector<std::shared_ptr<Tensor>> Function::Apply(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_GE(input_tensors.size(), 1);
    const auto *device = input_tensors[0]->GetDevice();
    // TODO(dcj): Cache context information to reduce setDevice overhead.
    device->SetDevice();

    std::vector<std::shared_ptr<Tensor>> output_tensors;
    {
        autograd::NoGradGuard no_grad;
        // no_grad in autograd.Function.Forward()
        output_tensors = Forward(input_tensors);
        SetupContext(input_tensors, output_tensors);
    }

    if (!autograd::GradMode::IsEnabled()) {
        // with no_grad: block graph building operations
        return output_tensors;
    }

    bool output_requires_grad = false;
    for (int idx = 0; idx < input_tensors.size(); ++idx) {
        const auto &input_tensor = input_tensors[idx];
        if (input_tensor->requires_grad() && input_tensor->is_leaf()) {
            next_functions_.emplace_back(input_tensor->grad_accumulator(), input_tensor->output_idx());
            input_tensor->grad_accumulator()->IncreaseDependenciesNumber();
        } else {
            next_functions_.emplace_back(input_tensor->grad_fn(), input_tensor->output_idx());
            if (input_tensor->grad_fn()) {
                input_tensor->grad_fn()->IncreaseDependenciesNumber();
            }
        }
        output_requires_grad |= input_tensor->requires_grad();
    }

    grad_outputs_reached_ = 0;
    grad_outputs_.resize(output_tensors.size(), nullptr);
    for (int output_idx = 0; output_idx < output_tensors.size(); ++output_idx) {
        auto &output_tensor = output_tensors[output_idx];
        // TODO(dcj): Mark if an output tensor need differentiable or not.
        output_tensor->set_requires_grad(output_requires_grad);
        output_tensor->set_is_leaf(false);
        output_tensor->set_grad_fn(output_requires_grad ? shared_from_this() : nullptr);
        output_tensor->set_output_idx(output_idx);
    }

    return output_tensors;
}

void Function::BackwardPartial(const std::shared_ptr<Tensor> &grad_output, int grad_output_idx) {
    const auto *device = grad_output->GetDevice();
    device->SetDevice();
    // std::cout << "Start BackwardPartial of " << type_ << " for output idx " << grad_output_idx << std::endl;

    // NOTE(dcj): The accumulate autograd function has no grad_outputs.
    // Temporarily resize the vector to hold one nullptr as a buffer.
    if (grad_outputs_.empty()) {
        grad_outputs_.resize(1, nullptr);
    }
    if (!grad_outputs_.at(grad_output_idx)) {
        grad_outputs_[grad_output_idx] = grad_output;
        ++grad_outputs_reached_;
    } else {
        auto kernel = Dispatcher::Instance().GetKernel({device->Type(), "AccumulateGrad"});
        kernel.Call<void>(grad_output, 1.0f, grad_outputs_.at(grad_output_idx));
    }
    ++dependencies_reached_;
    if (grad_outputs_reached_ == grad_outputs_.size()
        && (dependencies_reached_ == dependencies_number_ || dependencies_number_ == 0)) {

        // cudaDeviceSynchronize();
        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess) {
        //     fprintf(stderr, "CUDA kernel launch error: %s (at %s:%d)\n", cudaGetErrorString(err), __FILE__,
        //     __LINE__);
        // }
        // std::cout << "----------------------------------------------------------------------------------------------"
        //           << std::endl;
        // std::cout << "Start BackwardPartial of " << type_ << " for output idx " << grad_output_idx << std::endl;
        // std::cout << " - grad_output: " << std::endl;
        // if (grad_output) {
        // std::shared_ptr<Tensor> tensor_to_print;

        // if (grad_output->Dtype() == DataType::kBFLOAT16) {
        //     tensor_to_print = std::make_shared<Tensor>(grad_output->To(DataType::kFLOAT32));
        // } else {
        //     tensor_to_print = grad_output;
        // }

        // // // Print as usual
        // tensor_to_print->Print();

        //             if (tensor_to_print->Dtype() == DataType::kFLOAT32) {
        //                 const size_t num_elements = tensor_to_print->NumElements();
        //                 const size_t num_bytes = num_elements * sizeof(float);

        //                 std::vector<float> host_buffer(num_elements);

        //                 if (tensor_to_print->GetDevice()->Type() == DeviceType::kCPU) {
        //                     std::memcpy(host_buffer.data(), tensor_to_print->DataPtr(), num_bytes);
        //                 }
        // #ifdef USE_CUDA
        //                 else if (tensor_to_print->GetDevice()->Type() == DeviceType::kCUDA) {
        //                     cudaDeviceSynchronize();
        //                     cudaError_t err
        //                         = cudaMemcpy(host_buffer.data(), tensor_to_print->DataPtr(), num_bytes,
        //                         cudaMemcpyDeviceToHost);
        //                     CHECK_EQ(err, cudaSuccess) << "cudaMemcpy failed: " << cudaGetErrorString(err);
        //                 }
        // #endif
        //                 else {
        //                     LOG(FATAL) << "Unsupported device type for Print.";
        //                 }

        //                 // Cast raw pointer to float* (since we ensured float32)
        //                 float min_val = std::numeric_limits<float>::infinity();
        //                 float max_val = -std::numeric_limits<float>::infinity();

        //                 for (size_t i = 0; i < num_elements; i++) {
        //                     float v = host_buffer[i];
        //                     if (v < min_val) {
        //                         min_val = v;
        //                     }
        //                     if (v > max_val) {
        //                         max_val = v;
        //                     }
        //                 }

        //                 std::cout << "   min value: " << min_val << ", max value: " << max_val << std::endl;

        //                 float abs_max = std::max(std::fabs(min_val), std::fabs(max_val));
        //                 if (abs_max > 15.0f) {
        //                     std::cerr << "Detected value with |v| > 15 (abs_max=" << abs_max << "). Exiting..."
        //                     << std::endl;
        //                     // std::exit(0);
        //                 }
        //             }
        // }
        // if (type_ == "MaskFunction") {
        //     std::exit(0);
        // }

        std::vector<std::shared_ptr<Tensor>> grad_inputs;
        {
            autograd::NoGradGuard no_grad;
            // no_grad in autograd.Function.Backward()
            grad_inputs = Backward(grad_outputs_);
        }
        saved_tensors_.clear();
        grad_outputs_.clear();
        grad_outputs_reached_ = 0;
        dependencies_reached_ = 0;

        CHECK_EQ(grad_inputs.size(), next_functions_.size());
        for (int idx = 0; idx < grad_inputs.size(); ++idx) {
            auto &grad_input = grad_inputs[idx];
            auto &[next_function, output_idx] = next_functions_[idx];
            if (grad_input && next_function) {
                next_function->BackwardPartial(grad_input, output_idx);
            }
        }
    }
}

void Function::IncreaseDependenciesNumber() { ++dependencies_number_; }
} // namespace infini_train::autograd
