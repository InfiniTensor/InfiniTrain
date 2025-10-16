// pipeline_schedule.cc
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"

#include "glog/logging.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <unistd.h>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/nn/parallel/pp/send_recv.h"

namespace infini_train::nn::pipeline {

std::vector<std::shared_ptr<Tensor>> PipelineSchedule::SplitTensor(std::shared_ptr<Tensor> full_inputs) {
    // full_inputs (bs, seq_len)
    //  printf("SplitTensor entry! %ld\n", full_inputs->SizeInBytes());
    //  if (full_inputs.empty()) {
    //      LOG(FATAL) << "SplitTensor: no input tensors provided.";
    //  }
    if (num_microbatches_ == 1) {
        return {full_inputs};
    }

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
    std::vector<std::shared_ptr<Tensor>> micro_batches(num_microbatches_);
    std::vector<std::shared_ptr<Tensor>> target_mbs(num_microbatches_);

    if (stage_->IsFirstStage()) {
        // micro_batches = input->Split(NumMicrobatches(), 0);
        micro_batches = SplitTensor(input);
    }
    if (stage_->IsLastStage()) {
        // target_mbs = target->Split(NumMicrobatches(), 0);
        target_mbs = SplitTensor(target);
    }

    auto optim = stage_->optimizer();
    optim->ZeroGrad();

    float lossf = StepMicrobatches(micro_batches, target_mbs, loss_fn);

    optim->Step();

    return lossf;
}

float ScheduleGPipe::StepMicrobatches(const std::vector<std::shared_ptr<Tensor>> &microbatch_inputs,
                                      const std::vector<std::shared_ptr<Tensor>> &microbatch_targets,
                                      const std::shared_ptr<Module> &loss_fn) {
    const int n_microbatches = NumMicrobatches();
    if (n_microbatches == 0) {
        return 0.0;
    }
    // printf("ScheduleGPipe::StepMicrobatches stage %d n_microbatches %d\n", stage_index_, n_microbatches);
    std::vector<std::vector<std::shared_ptr<Tensor>>> outputs(n_microbatches);
    std::vector<std::shared_ptr<Tensor>> output_grads(n_microbatches);

    for (int mb_idx = 0; mb_idx < n_microbatches; ++mb_idx) {
        std::vector<std::shared_ptr<Tensor>> input_tensors;
        if (stage_->IsFirstStage()) {
            auto &tensor_ref = microbatch_inputs[mb_idx];

            input_tensors = {tensor_ref};
        } else {
            auto shapes = stage_->recv_shape();
            std::vector<std::shared_ptr<Tensor>> recv_tensors;
            for (int shape_i = 0; shape_i < shapes.size(); ++shape_i) {
                auto r_tensor = std::make_shared<Tensor>(shapes[shape_i], DataType::kFLOAT32, stage_->device());

                // TODO:只给第一个设置梯度
                if (shape_i == 0) {
                    r_tensor->set_requires_grad(true);
                    r_tensor->set_is_leaf(false);
                }
                recv_tensors.push_back(r_tensor);
            }
            auto outputs = IRecv(recv_tensors, stage_->device(), stage_->stage_index(), stage_->prev_rank());

            input_tensors = outputs;
        }

        auto output_tensors = stage_->ForwardOneChunk(input_tensors);
        outputs[mb_idx] = output_tensors;

        if (!stage_->IsLastStage()) {
            // printf("[stage %d] start send!! %d, %d, 当前是microbatch: %d\n", stage_index_, stage_->stage_index(),
            //        stage_->next_rank(), mb_idx);

            // PrintTensorSummary(outputs[mb_idx][0], "stage" + std::to_string(stage_index_) + "_mb____"
            //                                            + std::to_string(mb_idx) + "_send_pre");
            for (int i = 0; i < outputs[mb_idx].size(); i++) {
                if (outputs[mb_idx][i] == nullptr) {
                    outputs[mb_idx][i]
                        = std::make_shared<Tensor>((std::vector<int64_t>){}, DataType::kFLOAT32, stage_->device());
                }
            }
            outputs[mb_idx] = ISend(outputs[mb_idx], stage_->device(), stage_->stage_index(), stage_->next_rank(),
                                    stage_->recv_shape());
            // printf("[stage %d] 单stage正向发送OK %d\n", stage_index_, mb_idx);
        }
    }

    float lossf = 0.0;
    if (stage_->IsLastStage()) {
        for (int mb_idx = 0; mb_idx < n_microbatches; ++mb_idx) {
            auto &target = microbatch_targets[mb_idx];
            auto &output = outputs[mb_idx];

            if (!target) {
                LOG(FATAL) << "[ERROR] target is nullptr for mb_idx = " << mb_idx;
            }
            if (!output[0]) {
                LOG(FATAL) << "[ERROR] output is nullptr for mb_idx = " << mb_idx;
            }

            auto target_copy = target->To(output[0]->GetDevice());
            auto loss = loss_fn->Forward({output[0], std::make_shared<Tensor>(target_copy)})[0];

            if (!loss) {
                LOG(INFO) << "[ERROR] loss is nullptr at mb_idx = " << mb_idx;
                continue;
            }
            loss = loss / n_microbatches;
            // LOG(INFO) << "finish loss forward";
            auto loss_cpu = loss->To(DeviceManager::Instance()->GetDefaultDevice());

            lossf += static_cast<const float *>(loss_cpu.DataPtr())[0];

            // LOG(INFO) << "before backward";
            loss->Backward();
            // LOG(INFO) << "finish backward";
        }
    }

    if (!stage_->IsLastStage()) {
        for (int mb_idx = 0; mb_idx < n_microbatches; ++mb_idx) {
            auto &out_tensor = outputs[mb_idx][0];
            // printf("[stage %d] 触发单stage的反向 %d, 构造的接收梯度的张量维度:", stage_index_, mb_idx);
            // for (auto i : out_tensor->Dims()) { printf("%ld ", i); }
            // printf("\n");

            auto gradient = std::make_shared<Tensor>(out_tensor->Dims(), out_tensor->Dtype(), out_tensor->GetDevice());
            usleep(2000);
            // std::cout << "=====输出张量========, 梯度函数：" << out_tensor->grad_fn() << std::endl;
            out_tensor->Backward(gradient);
        }
    }
    return lossf;
}

} // namespace infini_train::nn::pipeline
