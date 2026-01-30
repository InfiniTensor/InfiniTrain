#include "infini_train/include/autograd/accumulate.h"

#include "glog/logging.h"

#include <atomic>
#include <iostream>

#include "infini_train/include/autograd/function_hook.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/infiniccl.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::autograd {
std::atomic<uint64_t> AccumulateGrad::global_id_counter_{0};

AccumulateGrad::AccumulateGrad(std::shared_ptr<Tensor> tensor, float learning_rate)
    : tensor_(tensor), learning_rate_(learning_rate) {

    if (tensor && tensor->GetDevice()->rank().thread_rank() == 0) {
        id_ = global_id_counter_.fetch_add(1, std::memory_order_relaxed);
    }
}

std::vector<std::shared_ptr<Tensor>> AccumulateGrad::Forward(const std::vector<std::shared_ptr<Tensor>> &) {
    LOG(FATAL) << "AccumulateGrad::Forward shall not be called directly!";
    return {};
}

std::vector<std::shared_ptr<Tensor>>
AccumulateGrad::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(grad_outputs.size(), 1);
    auto grad_output = grad_outputs[0];

    auto grad = tensor_->grad();
    auto device = tensor_->GetDevice();
    device->SetDevice();

    if (grad_output) {
        if (grad) {
            if (tensor_->ConsumeGradOverwriteFlag()) {
                // If the tensor is marked to overrite its current grad on next grad update
                // See notes in `infini_train::nn::parallel::Reducer::PrepareForBackward()`
                // NOTE(zbl): must copy, cannot change grad buffer address
                grad->CopyFrom(grad_output);
            } else {
                auto kernel = Dispatcher::Instance().GetKernel({device->Type(), "AccumulateGrad"});
                kernel.Call<void>(grad_output, learning_rate_, grad);
            }
        } else {
            // FIXME(zbl): check whether need to do copying instead of slicing
            auto new_grad = std::make_shared<Tensor>(*grad_output.get(), 0, grad_output->Dims());
            tensor_->set_grad(new_grad);
        }
        auto hook = tensor_->post_accumulate_grad_hook();
        if (hook != nullptr) {
            (*hook)(tensor_->grad());
            std::shared_ptr<Tensor> comm_grad = nullptr;

            if (infini_train::nn::parallel::global::IsHeterogeneous()) {
                if (tensor_->GetDevice()->rank().thread_rank() == 0) {
                    // LOG(ERROR) << std::format("id: {}, output_idx: {}, dims: ", std::to_string(id_),
                    //                           std::to_string(tensor_->output_idx()));
                    // auto dims = tensor_->Dims();
                    // for (int i = 0; i < dims.size(); ++i) { LOG(ERROR) << dims[i]; }
                    // #ifdef USE_MACA
                    //                     // 1. write tensor_->grad() as npy, grad_tensor_functionid_outputid
                    //                     infini_train::nn::parallel::WriteTensor(
                    //                         tensor_->grad(),
                    //                         std::format("{}/muxi/tensor_{}_{}",
                    //                         infini_train::nn::parallel::kSharedPathPrefix,
                    //                                     std::to_string(id_), std::to_string(tensor_->output_idx())));
                    //                     // 2. read another node's tensor_->grad() from npy
                    //                     comm_grad = infini_train::nn::parallel::ReadTensor(
                    //                         std::format("{}/nv/tensor_{}_{}",
                    //                         infini_train::nn::parallel::kSharedPathPrefix,
                    //                                     std::to_string(id_), std::to_string(tensor_->output_idx())),
                    //                         tensor_->GetDevice());
                    // #endif
                    // #ifdef USE_CUDA
                    //                     // 1. write tensor_->grad() as npy, grad_tensor_functionid_outputid
                    //                     infini_train::nn::parallel::WriteTensor(
                    //                         tensor_->grad(),
                    //                         std::format("{}/nv/tensor_{}_{}",
                    //                         infini_train::nn::parallel::kSharedPathPrefix,
                    //                                     std::to_string(id_), std::to_string(tensor_->output_idx())));
                    //                     // 2. read another node's tensor_->grad() from npy
                    //                     comm_grad = infini_train::nn::parallel::ReadTensor(
                    //                         std::format("{}/muxi/tensor_{}_{}",
                    //                         infini_train::nn::parallel::kSharedPathPrefix,
                    //                                     std::to_string(id_), std::to_string(tensor_->output_idx())),
                    //                         tensor_->GetDevice());
                    // #endif

                    // // 3. do inter-node allreduce
                    // tensor_->set_grad((tensor_->grad() + comm_grad) / 2);

                    infiniAllReduce(tensor_->grad()->DataPtr(), tensor_->grad()->DataPtr(),
                                    tensor_->grad()->NumElements(), infiniFloat32, infiniSum,
                                    nn::parallel::global::GetInfinicclComm(), nullptr);
                    tensor_->set_grad(tensor_->grad() / 2);
                } else {
                    tensor_->ZeroGrad(false);
                }
                // 4. dp intra-node broadcast
                auto ddp_pg = infini_train::nn::parallel::ProcessGroupFactory::Instance()->Get(
                    infini_train::nn::parallel::GetDataParallelProcessGroupName(
                        tensor_->GetDevice()->rank().thread_rank()));
                infini_train::nn::parallel::function::AllReduce(
                    tensor_->grad(), infini_train::nn::parallel::function::ReduceOpType::kSum, ddp_pg);
            }
        }
        tensor_->ResetAccumulator();
    }
    return {};
}
} // namespace infini_train::autograd
