// pipeline_schedule.cc
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"

#include "glog/logging.h"
#include <cuda_runtime.h>

#include "infini_train/include/tensor.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"

namespace infini_train::nn::pipeline {

//TODO(jym):需要提前预知 output_shape，本次要获知上一层的输出形状
std::shared_ptr<Tensor> PipelineSchedule::AllocateForwardRecvTensor() {
    return std::make_shared<Tensor>(output_shape, dtype_, device_);
}

std::shared_ptr<Tensor> PipelineSchedule::AllocateBackwardRecvTensor() {
    // 本个stage前向输出的 shape
    auto output_shape = stage_->GetOutputShape(); //TODO(jym):前向保存在stage_
    return std::make_shared<Tensor>(output_shape, dtype_, device_);
}

std::vector<std::vector<std::shared_ptr<Tensor>>> PipelineSchedule::SplitTensor(
    const std::vector<std::shared_ptr<Tensor>>& full_inputs) {

    if (full_inputs.empty()) {
        LOG(FATAL) << "SplitTensor: no input tensors provided.";
    }

    const auto& first_dims = full_inputs[0]->Dims();
    if (first_dims.empty()) {
        LOG(FATAL) << "SplitTensor: tensor has no dimensions.";
    }
    int64_t batch_size = first_dims[0];

    int microbatch_size = batch_size / num_microbatches_;
    int remainder = batch_size % num_microbatches_;

    std::vector<std::vector<std::shared_ptr<Tensor>>> micro_batches(num_microbatches_);

    int start_idx = 0;
    for (int mb_idx = 0; mb_idx < num_microbatches_; ++mb_idx) {
        int current_size = microbatch_size + (mb_idx == num_microbatches_ - 1 ? remainder : 0);

        for (const auto& tensor : full_inputs) {
            if (tensor->Dims() != batch_size) {
                LOG(FATAL) << "SplitTensor: tensor size mismatch on dim 0.";
            }
            // 切片：返回 view（共享数据，支持反向传播）
            auto sliced = tensor->Slice(0, start_idx, current_size);
            micro_batches[mb_idx].push_back(sliced);
        }

        start_idx += current_size;
    }

    return micro_batches;
}

void PipelineSchedule::Step(const std::vector<std::shared_ptr<Tensor>>& input,
    const std::vector<std::shared_ptr<Tensor>>& target,
    const std::shared_ptr<Module>& loss_fn,
    std::vector<std::shared_ptr<Tensor>>* losses) {

    auto arg_mbs = SplitTensor(input);
    auto target_mbs = target ? SplitTensor(target)[0] : std::vector<std::shared_ptr<Tensor>>();

    // 验证 microbatch 数量
    if (arg_mbs.empty() || arg_mbs[0].empty()) {
        LOG(FATAL) << "No microbatches to process.";
    }

    // // 3. 初始化当前 stage（如果需要）
    // if (!stage_->is_initialized()) {
    //     stage_->Initialize(arg_mbs[0]);
    // }

    StepMicrobatches(arg_mbs, target_mbs, loss_fn, losses);

    // 合并输出（如果是最后一 stage）
    if (stage_->IsLastStage() && losses) {
        // TODO(jym):计算损失
    }
}

void Schedule1F1B::StepMicrobatches(const std::vector<std::vector<std::shared_ptr<Tensor>>>& microbatch_inputs,
    const std::vector<std::shared_ptr<Tensor>>& microbatch_targets, const std::shared_ptr<Module>& loss_fn,
    std::vector<std::shared_ptr<Tensor>>* losses) {
    const int n_microbatches = NumMicrobatches();
    if (n_microbatches == 0)
        return;

    std::vector<std::shared_ptr<Tensor>> outputs(n_microbatches);

    int fwd_mb_index = 0;  //前向mrcro_batch索引
    int bwd_mb_index = 0;  //反向mrcro_batch索引

    // Warmup 阶段  只做前向，让所有stage都能启动
    int warmup_steps = stage_->IsLastStage() 
        ? n_microbatches 
        : std::min(n_microbatches, stage_->num_stages_ - stage_->stage_index_);

    for (int i = 0; i < warmup_steps; ++i) {
        std::vector<std::shared_ptr<Tensor>> input_tensors;
        if (stage_->IsFirstStage()) {
            input_tensors = microbatch_inputs[fwd_mb_index];
        } else {
            auto recv_tensor = AllocateForwardRecvTensor();  //TODO(jym):分配接收变量的内存
            Irecv(recv_tensor, stage_->prev_rank_);

            const auto *dev = dynamic_cast<const CudaDevice*>(recv_tensor[0]->GetDevice());
            cudaEvent_t done;
            cudaEventRecord(done, dev->Stream());
            cudaStreamWaitEvent(dev->Stream(), done, 0);

            input_tensors = {recv_tensor};
        }

        auto output_tensors = stage_->ForwardOneChunk(input_tensors);

        outputs[fwd_mb_index] = output_tensors[0];  // 假设每个stage都是单输出

        // 不是最后一层的话，发送给下一stage
        if (!stage_->IsLastStage()) {
            Isend(outputs[fwd_mb_index], stage_->next_rank_);
        }

        ++fwd_mb_index;
    }

    // 1F1B 主循环
    while (true) {
        // 先进行反向，最后一个stage直接反向，非最后一个stage先接收后续节点的梯度
        std::shared_ptr<Tensor> output_grad;

        if (stage_->IsLastStage()) {
            auto& target = microbatch_targets[bwd_mb_index];
            auto& output = outputs[bwd_mb_index];
            
            auto loss = loss_fn->Forward({output, target})[0];
            if (losses) losses->push_back(loss);

            output_grad = std::make_shared<Tensor>(
                output->Dims(), output->Dtype(), output->GetDevice());
            output_grad->Fill(1.0f);
            output_grad->set_requires_grad(false);
        } else {
            output_grad = AllocateBackwardRecvTensor(); //TODO(jym):
            Irecv(output_grad, stage_->next_rank_);

            const auto *dev = dynamic_cast<const CudaDevice*>(output_grad->GetDevice());
            cudaEvent_t done;
            cudaEventRecord(done, dev->Stream());
            cudaStreamWaitEvent(dev->Stream(), done, 0);
        }

        // 反向
        auto input_grad = stage_->BackwardOneChunk(output_grad);
        if (!stage_->IsFirstStage() && input_grad) {
            Isend(input_grad, stage_->prev_rank_);
        }
        ++bwd_mb_index;

        if (fwd_mb_index == n_microbatches) {
            break;
        }

        // 再进行前向
        std::vector<std::shared_ptr<Tensor>> new_input;

        if (stage_->IsFirstStage()) {
            new_input = microbatch_inputs[fwd_mb_index];
        } else {
            auto recv_tensor = AllocateForwardRecvTensor();  //TODO(jym):
            Irecv(recv_tensor, stage_->prev_rank_);

            const auto *dev = dynamic_cast<const CudaDevice*>(recv_tensor->GetDevice());
            cudaEvent_t done;
            cudaEventRecord(done, dev->Stream());
            cudaStreamWaitEvent(dev->Stream(), done, 0);

            new_input = {recv_tensor};
        }

        auto output_tensors = stage_->ForwardOneChunk(new_input);
        outputs[fwd_mb_index] = output_tensors[0];

        if (!stage_->IsLastStage()) {
            Isend(outputs[fwd_mb_index], stage_->next_rank_);
        }

        ++fwd_mb_index;
    }

    // Cooldown: 剩余反向传播
    while (bwd_mb_index < n_microbatches) {
        std::shared_ptr<Tensor> output_grad;
        if (stage_->IsLastStage()) {
            auto& target = microbatch_targets[bwd_mb_index];
            auto& output = outputs[bwd_mb_index];
            
            auto loss = loss_fn->Forward({output, target})[0];
            if (losses) losses->push_back(loss);

            output_grad = std::make_shared<Tensor>(
                output->Dims(), output->Dtype(), output->GetDevice());
            output_grad->Fill(1.0f);
            output_grad->set_requires_grad(false);
        } else {
            output_grad = AllocateBackwardRecvTensor(); //TODO(jym):
            Irecv(output_grad, stage_->next_rank_);

            const auto *dev = dynamic_cast<const CudaDevice*>(output_grad->GetDevice());
            cudaEvent_t done;
            cudaEventRecord(done, dev->Stream());
            cudaStreamWaitEvent(dev->Stream(), done, 0);

        }
       
        auto input_grad = stage_->BackwardOneChunk(output_grad);
        if (!stage_->IsFirstStage() && input_grad) {
            Isend(input_grad, stage_->prev_rank_);
        }
        ++bwd_mb_index;
    }

    // 缩放梯度（micro-batch 累加）
    stage_->ScaleGrads(1.0f / static_cast<float>(n_microbatches));
}

} // namespace infini_train::nn::pipeline