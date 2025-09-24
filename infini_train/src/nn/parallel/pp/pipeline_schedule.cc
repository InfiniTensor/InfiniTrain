// pipeline_schedule.cc
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"

#include "glog/logging.h"
#include <cuda_runtime.h>
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
    std::vector<std::shared_ptr<Tensor>> micro_batches;
    std::vector<std::shared_ptr<Tensor>> target_mbs;

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
    std::vector<std::shared_ptr<Tensor>> outputs(n_microbatches);
    // std::vector<std::shared_ptr<Tensor>> output_grads(n_microbatches);

    for (int mb_idx = 0; mb_idx < n_microbatches; ++mb_idx) {
        std::shared_ptr<Tensor> input_tensors;
        if (stage_->IsFirstStage()) {
            auto &tensor_ref = microbatch_inputs[mb_idx];

            input_tensors = tensor_ref;
        } else {
            auto shape = stage_->recv_shape();
            auto recv_tensor = std::make_shared<Tensor>(
                std::vector<int64_t>{shape.batch_size / n_microbatches, shape.seq_len, shape.hidden_size},
                DataType::kFLOAT32, stage_->device());
            // printf("[stage %d] recv_tensor shape %ld %ld %ld 接收的rank:%d! \n", stage_index_, recv_tensor->Dims()[0],
            //        recv_tensor->Dims()[1], recv_tensor->Dims()[2], stage_->prev_rank());
            auto output = IRecv({recv_tensor}, stage_->device(), stage_->stage_index(), stage_->prev_rank());

            input_tensors = recv_tensor;
        }

        if (input_tensors == nullptr) {
            printf("[stage %d] input_tensors is null\n", stage_index_);
        }

        auto output_tensors = stage_->ForwardOneChunk({input_tensors});
        outputs[mb_idx] = output_tensors[0];

        if (!stage_->IsLastStage()) {
            // printf("[stage %d] start send!! %d, %d\n", stage_index_, stage_->stage_index(), stage_->next_rank());

            // PrintTensorSummary(outputs[mb_idx], "stage" + std::to_string(stage_index_) + "_mb" +
            // std::to_string(mb_idx) + "_send_pre");
            ISend({outputs[mb_idx]}, stage_->device(), stage_->stage_index(), stage_->next_rank());
            // printf("[stage %d] after send!! microbatch: %d\n", stage_index_, mb_idx);
        }
    }

    float lossf = 0.0;
    if (stage_->IsLastStage()) {
        for (int mb_idx = 0; mb_idx < n_microbatches; ++mb_idx) {
            std::cout << "[DEBUG] Processing microbatch " << mb_idx << "/" << n_microbatches << std::endl;

            auto &target = microbatch_targets[mb_idx];
            auto &output = outputs[mb_idx];

            if (!target) {
                LOG(FATAL) << "[ERROR] target is nullptr for mb_idx = " << mb_idx;
            }
            if (!output) {
                LOG(FATAL) << "[ERROR] output is nullptr for mb_idx = " << mb_idx;
            }

            // LOG(INFO) << "output dims: " << output->Dims().size();
            // for (int i = 0; i < output->Dims().size(); ++i) {
            //     LOG(INFO) << "output dim[" << i << "] = " << output->Dims()[i];
            // }

            // LOG(INFO) << "target dims: " << target->Dims().size();
            // for (int i = 0; i < target->Dims().size(); ++i) {
            //     LOG(INFO) << "target dim[" << i << "] = " << target->Dims()[i];
            // }

            // LOG(INFO) << "mb_idx=" << mb_idx << " output_dev=" << output->GetDevice()->ToString()
            //           << " target_dev=" << target->GetDevice()->ToString();

            auto target_copy = target->To(output->GetDevice());
            auto loss = loss_fn->Forward({output, std::make_shared<Tensor>(target_copy)})[0];
            if (!loss) {
                LOG(INFO) << "[ERROR] loss is nullptr at mb_idx = " << mb_idx;
                continue;
            }
            loss = loss / n_microbatches;
            // LOG(INFO) << "finish loss forward";
            auto loss_cpu = loss->To(DeviceManager::Instance()->GetDefaultDevice());

            lossf += static_cast<const float *>(loss_cpu.DataPtr())[0];

            loss->Backward();
            // LOG(INFO) << "finish backward";
        }
    }

    // // 打印参数验证
    // if (stage_->layers_.empty()) {
    //     printf("stage_->layers_ is empty!\n");
    //     return 0;
    // }

    // for (size_t li = 0; li < stage_->layers_.size(); ++li) {
    //     auto layer = stage_->layers_[li];
    //     auto params = layer->Parameters();

    //     if (params.empty()) {
    //         printf("Layer %s has no parameters!\n", layer->type().c_str());
    //         continue;
    //     }

    //     auto w = params[0];
    //     if (!w) {
    //         printf("Parameter tensor is null!\n");
    //         continue;
    //     }

    //     if (!w->DataPtr()) {
    //         printf("Parameter data is not allocated! shape=%ld\n", w->Dims()[0]);
    //         continue;
    //     }

        
    //     auto w_cpu = w->To(DeviceManager::Instance()->GetDefaultDevice());
    //     if (!w_cpu.DataPtr()) {
    //         printf("Failed to copy tensor to CPU!\n");
    //         continue;
    //     }

    //     float *data = static_cast<float *>(w_cpu.DataPtr());
    //     printf("Weight[0][0] = %f\n", data[0]);
    // }
    return lossf;
}
} // namespace infini_train::nn::pipeline