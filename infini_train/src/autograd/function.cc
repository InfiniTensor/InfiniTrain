#include "infini_train/include/autograd/function.h"

#include "glog/logging.h"

#include "infini_train/include/autograd/accumulate.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

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

    // === 添加打印：记录前向传播的Function类型和地址 ===
    // std::cout << "[FORWARD] Function: " << typeid(*this).name() << " @ " << this << ", Device: " <<
    // device->ToString()
    //           << ", Inputs: " << input_tensors.size() << ", Outputs: " << output_tensors.size() << std::endl;
    // ===================================================
    auto input_requires_grad_flags = InputRequiresGrad(input_tensors);

    bool output_requires_grad = false;
    for (int idx = 0; idx < input_tensors.size(); ++idx) {
        if (!input_requires_grad_flags[idx]) {
            continue;
        }
        const auto &input_tensor = input_tensors[idx];
        // std::cout << "[FORWARD] Function: " << typeid(*this).name() << " " << input_tensor->requires_grad()
        //   << "  " << input_tensor->is_leaf() << " 梯度函数： "<< input_tensor->grad_fn() << std::endl;
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

    auto output_requires_grad_flags = OutputRequiresGrad(input_tensors);
    grad_outputs_reached_ = 0;

    grad_outputs_.clear();
    // grad_outputs_.resize(output_tensors.size(), nullptr);
    for (int output_idx = 0; output_idx < output_tensors.size(); ++output_idx) {
        auto &output_tensor = output_tensors[output_idx];

        // TODO(dcj): Mark if an output tensor need differentiable or not.
        // bool need_grad = output_requires_grad;
        bool need_grad = output_requires_grad_flags[output_idx];
        output_tensor->set_requires_grad(need_grad);
        output_tensor->set_is_leaf(false);
        output_tensor->set_grad_fn(need_grad ? shared_from_this() : nullptr);
        output_tensor->set_output_idx(output_idx);

        if (need_grad) {
            grad_outputs_.push_back(nullptr);
        }
        // // === 添加打印：标记每个输出张量的grad_fn ===
        // std::cout << "  [OUTPUT] Tensor[" << output_idx
        //           << "] grad_fn=" << (output_requires_grad ? shared_from_this().get() : nullptr) << std::endl;
        // // ================================================
    }
    return output_tensors;
}

void Function::BackwardPartial(const std::shared_ptr<Tensor> &grad_output, int grad_output_idx) {
    const auto *device = grad_output->GetDevice();
    device->SetDevice();

    // === 添加打印：进入反向传播的详细信息 ===
    // std::cout << "[BACKWARD-ENTER] Function: " << typeid(*this).name() << " @ " << this
    //           << ", grad_output_idx=" << grad_output_idx << ", Device: " << device->ToString()
    //           << ", grad_output=" << grad_output.get() << ", size=" << (grad_output ? grad_output->SizeInBytes() : 0)
    //           << std::endl;
    // ===========================================

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

    // === 添加打印：当前状态 ===
    // std::cout << "  [STATE] grad_outputs_reached=" << grad_outputs_reached_ << "/" << grad_outputs_.size()
    //           << ", dependencies_reached=" << dependencies_reached_ << "/" << dependencies_number_ << std::endl;
    // ==========================

    if (grad_outputs_reached_ == grad_outputs_.size()
        && (dependencies_reached_ == dependencies_number_ || dependencies_number_ == 0)) {

        // === 添加打印：开始执行Backward逻辑 ===
        // std::cout << "  [BACKWARD-EXEC] Executing Backward() for Function: " << typeid(*this).name() << " @ " << this
        //           << std::endl;
        // std::cout << "  [BACKWARD-EXEC] 当前: " << grad_outputs_reached_ << " " << dependencies_reached_ << " " <<
        // dependencies_number_
        //           << std::endl;
        // ======================================

        std::vector<std::shared_ptr<Tensor>> grad_inputs;
        {
            autograd::NoGradGuard no_grad;
            // no_grad in autograd.Function.Backward()
            grad_inputs = Backward(grad_outputs_);
            // std::cout << "----------------FLAG0--------------  "<< grad_inputs.size()<< std::endl;
        }

        // std::cout << "----------------FLAG1--------------"<< std::endl;
        saved_tensors_.clear();
        grad_outputs_.clear();
        grad_outputs_reached_ = 0;
        dependencies_reached_ = 0;
        // std::cout << "----------------FLAG2--------------"<< std::endl;
        // usleep(2000);
        CHECK_EQ(grad_inputs.size(), next_functions_.size());
        for (int idx = 0; idx < grad_inputs.size(); ++idx) {
            auto &grad_input = grad_inputs[idx];
            auto &[next_function, output_idx] = next_functions_[idx];
            if (grad_input && next_function) {
                // std::cout << "------------------------------"<< std::endl;
                // === 添加打印：准备调用下一个反向函数 ===
                // std::cout << "  [NEXT-BWD] SENDING grad_input[" << idx << "]=" << grad_input.get()
                //           << " to Function: " << (next_function ? typeid(*next_function).name() : "nullptr") << " @ "
                //           << next_function.get() << ", output_idx=" << output_idx << std::endl;
                // ========================================

                next_function->BackwardPartial(grad_input, output_idx);
            }
        }
    }
}

void Function::IncreaseDependenciesNumber() { ++dependencies_number_; }
} // namespace infini_train::autograd
