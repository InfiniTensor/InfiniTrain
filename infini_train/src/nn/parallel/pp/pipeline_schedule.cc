// pipeline_schedule.cc
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"

#include "glog/logging.h"
#include <cuda_runtime.h>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/nn/parallel/pp/send_recv.h"

namespace infini_train::nn::pipeline {

std::vector<std::shared_ptr<Tensor>> PipelineSchedule::SplitTensor(std::shared_ptr<Tensor> full_inputs) {
    // full_inputs (bs, seq_len)
    //  printf("SplitTensor entry! %ld\n", full_inputs->SizeInBytes());
    //  if (full_inputs.empty()) {
    //      LOG(FATAL) << "SplitTensor: no input tensors provided.";
    //  }

    const auto &first_dims = full_inputs->Dims();
    if (first_dims.empty()) {
        LOG(FATAL) << "SplitTensor: tensor has no dimensions.";
    }
    int64_t batch_size = first_dims[0];

    int microbatch_size = batch_size / num_microbatches_;
    int remainder = batch_size % num_microbatches_;

    std::vector<std::shared_ptr<Tensor>> micro_batches;

    int start_idx = 0;
    int end_idx = 0;
    for (int mb_idx = 0; mb_idx < num_microbatches_; ++mb_idx) {
        int current_size = microbatch_size + (mb_idx == num_microbatches_ - 1 ? remainder : 0);
        end_idx = start_idx + current_size;

        if (start_idx < 0 || end_idx > batch_size || start_idx >= end_idx) {
            printf("Invalid slice range: [%d, %d), batch_size=%ld\n", start_idx, end_idx, batch_size);
            abort();
        }

        // printf("SplitTensor mb_idx=%d  start_idx=%d  current_size=%d  batch_size=%ld\n",
        //     mb_idx, start_idx, current_size, batch_size);
        // printf("SplitTensor start_idx %d, end_idx %d, \n", start_idx, end_idx);

        if (full_inputs->Dims()[0] != batch_size) {
            LOG(FATAL) << "SplitTensor: tensor size mismatch on dim 0.";
        }

        // printf("[stage 0] SplitTensor sliced start\n");

        auto sliced = full_inputs->Slice(0, start_idx, end_idx);
        // printf("[stage 0] SplitTensor sliced after\n");
        micro_batches.push_back(sliced);

        start_idx = end_idx;
    }
    // printf("[stage 0] SplitTensor exit OK! %ld\n", micro_batches.size());
    return micro_batches;
}

float PipelineSchedule::Step(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
                             const std::shared_ptr<Module> &loss_fn) {
    std::vector<std::shared_ptr<Tensor>> micro_batches;
    std::vector<std::shared_ptr<Tensor>> target_mbs;
    if (stage_index_ == 0) {
        micro_batches = SplitTensor(input);
        target_mbs = SplitTensor(target);
        // printf("[stage %d] SplitTensor OK!  %ld\n", stage_index_, micro_batches.size());
    }

    float lossf = StepMicrobatches(micro_batches, target_mbs, loss_fn);

    // OptimizerStep();

    return lossf;
}

void PipelineSchedule::OptimizerStep() {
    auto optim = stage_->optimizer();
    optim->ZeroGrad();
    optim->Step();
}

float Schedule1F1B::StepMicrobatches(const std::vector<std::shared_ptr<Tensor>> &microbatch_inputs,
                                     const std::vector<std::shared_ptr<Tensor>> &microbatch_targets,
                                     const std::shared_ptr<Module> &loss_fn) {
    const int n_microbatches = NumMicrobatches();
    if (n_microbatches == 0) {
        return 0.0;
    }

    std::vector<std::shared_ptr<Tensor>> outputs(n_microbatches);

    int mb_idx = 0;       // 前向mrcro_batch索引
    int bwd_mb_index = 0; // 反向mrcro_batch索引

    // Warmup 阶段
    int warmup_steps = stage_->IsLastStage() ? n_microbatches
                                             : std::min(n_microbatches, stage_->num_stages() - stage_->stage_index());

    for (int i = 0; i < warmup_steps; ++i) {
        std::shared_ptr<Tensor> input_tensors;
        if (stage_->IsFirstStage()) {
            input_tensors = microbatch_inputs[mb_idx];
        } else {
            auto shape = stage_->recv_shape();
            auto recv_tensor
                = std::make_shared<Tensor>(std::vector<int64_t>{shape.batch_size, shape.seq_len, shape.hidden_size},
                                           DataType::kFLOAT32, stage_->device());
            auto output = IRecv({recv_tensor}, stage_->device(), stage_->stage_index(), stage_->prev_rank());
        }

        auto output_tensors = stage_->ForwardOneChunk({input_tensors});
        outputs[mb_idx] = output_tensors[0];

        if (!stage_->IsLastStage()) {
            ISend({outputs[mb_idx]}, stage_->device(), stage_->stage_index(), stage_->next_rank());
        }

        ++mb_idx;
    }

    // 1F1B 主循环
    while (true) {
        std::shared_ptr<Tensor> output_grad;

        if (stage_->IsLastStage()) {
            for (int mb_idx = 0; mb_idx < n_microbatches; ++mb_idx) {
                auto &target = microbatch_targets[mb_idx];
                auto &output = outputs[mb_idx];
                auto loss = loss_fn->Forward({output, target})[0];

                loss->Backward();
            }
        }

        // 再进行前向
        std::shared_ptr<Tensor> new_input;

        if (stage_->IsFirstStage()) {
            new_input = microbatch_inputs[mb_idx];
        } else {
            auto shape = stage_->recv_shape();
            auto recv_tensor
                = std::make_shared<Tensor>(std::vector<int64_t>{shape.batch_size, shape.seq_len, shape.hidden_size},
                                           DataType::kFLOAT32, stage_->device());
            auto output = IRecv({recv_tensor}, stage_->device(), stage_->stage_index(), stage_->prev_rank());
        }

        auto output_tensors = stage_->ForwardOneChunk({new_input});
        outputs[mb_idx] = output_tensors[0];

        if (!stage_->IsLastStage()) {
            ISend({outputs[mb_idx]}, stage_->device(), stage_->stage_index(), stage_->next_rank());
        }

        ++mb_idx;
    }

    // Cooldown: 剩余反向传播
    while (bwd_mb_index < n_microbatches) {
        std::shared_ptr<Tensor> output_grad;
        if (stage_->IsLastStage()) {
            for (int mb_idx = 0; mb_idx < n_microbatches; ++mb_idx) {
                auto &target = microbatch_targets[mb_idx];
                auto &output = outputs[mb_idx];
                auto loss = loss_fn->Forward({output, target})[0];

                loss->Backward();
            }
        }
    }

    // 缩放梯度（micro-batch 累加）
    // stage_->ScaleGrads(1.0f / static_cast<float>(n_microbatches));
}

float ScheduleGPipe::StepMicrobatches(const std::vector<std::shared_ptr<Tensor>> &microbatch_inputs,
                                      const std::vector<std::shared_ptr<Tensor>> &microbatch_targets,
                                      const std::shared_ptr<Module> &loss_fn) {
    const int n_microbatches = NumMicrobatches();
    if (n_microbatches == 0) {
        return 0.0;
    }
    printf("ScheduleGPipe::StepMicrobatches stage %d n_microbatches %d\n", stage_index_, n_microbatches);
    std::vector<std::shared_ptr<Tensor>> outputs(n_microbatches);
    std::vector<std::shared_ptr<Tensor>> output_grads(n_microbatches);

    for (int mb_idx = 0; mb_idx < n_microbatches; ++mb_idx) {
        std::shared_ptr<Tensor> input_tensors;
        if (stage_->IsFirstStage()) {
            auto &tensor_ref = microbatch_inputs[mb_idx];

            input_tensors = tensor_ref;
        } else {
            auto shape = stage_->recv_shape();
            auto recv_tensor
                = std::make_shared<Tensor>(std::vector<int64_t>{shape.batch_size, shape.seq_len, shape.hidden_size},
                                           DataType::kFLOAT32, stage_->device());
            printf("[stage %d] recv_tensor shape %d %d %d 接收的rank:%d! \n", stage_index_, recv_tensor->Dims()[0],
                   recv_tensor->Dims()[1], recv_tensor->Dims()[2], stage_->prev_rank());
            auto output = IRecv({recv_tensor}, stage_->device(), stage_->stage_index(),
            stage_->prev_rank());
            printf("[stage %d] Recv OK!!! \n", stage_index_);
            // input_tensors = output[0];
            infini_train::nn::init::Uniform(recv_tensor, -1.0, 1.0);
            printf("%f\n", static_cast<const float *>(recv_tensor->DataPtr()));
            input_tensors = recv_tensor;
        }

        if (input_tensors == nullptr) {
            printf("[stage %d] input_tensors is null\n", stage_index_);
        }
        printf("[stage %d] stage_->ForwardOneChunk start\n", stage_index_);
        auto output_tensors = stage_->ForwardOneChunk({input_tensors});
        outputs[mb_idx] = output_tensors[0];

        if (!stage_->IsLastStage()) {
            printf("[stage %d] start send!! %d, %d\n", stage_index_, stage_->stage_index(), stage_->next_rank());
            ISend({outputs[mb_idx]}, stage_->device(), stage_->stage_index(), stage_->next_rank());
            printf("[stage %d] after send!! microbatch: %d\n", stage_index_, mb_idx);
        }

        // std::cout << "一个microbatch循环完成 ForwardOneChunk:" << mb_idx << std::endl;
    }

    float lossf = 0.0;
    if (stage_->IsLastStage()) {
        for (int mb_idx = 0; mb_idx < n_microbatches; ++mb_idx) {
            // std::cout << "[DEBUG] Processing microbatch " << mb_idx << "/" << n_microbatches << std::endl;
            auto &target = microbatch_targets[mb_idx];
            auto &output = outputs[mb_idx];
            auto loss = loss_fn->Forward({output, target})[0];
            if (!loss) {
                LOG(INFO) << "[ERROR] loss is nullptr at mb_idx = " << mb_idx;
                continue;
            }

            LOG(INFO) << "finish loss forward";
            auto loss_cpu = loss->To(DeviceManager::Instance()->GetDefaultDevice());
            lossf += static_cast<const float *>(loss_cpu.DataPtr())[0];

            loss->Backward();
            LOG(INFO) << "finish backward";
            // std::cout << "[DEBUG] Backward done for mb_idx = " << mb_idx << std::endl;
        }
    }

    return lossf / n_microbatches;
}
} // namespace infini_train::nn::pipeline