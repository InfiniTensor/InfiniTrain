// pipeline_schedule.cc
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"

#include "glog/logging.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <unistd.h>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/pp/send_recv.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

std::vector<std::shared_ptr<Tensor>> PipelineSchedule::SplitTensor(std::shared_ptr<Tensor> full_inputs) {
    const auto n = num_microbatches_;
    if (n == 1) {
        return {full_inputs};
    }

    const auto &first_dims = full_inputs->Dims();
    if (first_dims.empty()) {
        LOG(FATAL) << "SplitTensor: tensor has no dimensions.";
    }
    int64_t batch_size = first_dims[0];
    int microbatch_size = batch_size / n;
    int remainder = batch_size % n;

    std::vector<std::shared_ptr<Tensor>> micro_batches;

    int start_idx = 0;
    int end_idx = 0;
    for (int mb = 0; mb < n; ++mb) {
        int current_size = microbatch_size + (mb == n - 1 ? remainder : 0);
        end_idx = start_idx + current_size;

        if (start_idx < 0 || end_idx > batch_size || start_idx >= end_idx) {
            LOG(FATAL) << "Invalid slice range: [%d, %d), batch_size=%ld" << start_idx << end_idx << batch_size;
        }

        if (full_inputs->Dims()[0] != batch_size) {
            LOG(FATAL) << "SplitTensor: tensor size mismatch on dim 0.";
        }

        auto sliced = full_inputs->Slice(0, start_idx, end_idx);

        micro_batches.push_back(sliced);

        start_idx = end_idx;
    }

    return micro_batches;
}

float PipelineSchedule::Step(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
                             const std::shared_ptr<Module> &loss_fn) {
    std::vector<std::shared_ptr<Tensor>> micro_batches(num_microbatches_);
    std::vector<std::shared_ptr<Tensor>> target_mbs(num_microbatches_);
    if (stage_->IsFirstStage()) {
        micro_batches = SplitTensor(input);
    }
    if (stage_->IsLastStage()) {
        target_mbs = SplitTensor(target);
    }

    const auto &optim = stage_->optimizer();

    optim->ZeroGrad();

    float lossf = StepMicrobatches(micro_batches, target_mbs, loss_fn);

    optim->Step();

    return lossf;
}

std::vector<std::shared_ptr<Tensor>> PipelineSchedule::ReceiveFromPrev() {
    std::vector<std::shared_ptr<Tensor>> recv_tensors;
    if (!stage_->IsFirstStage()) {
        auto shapes = stage_->recv_shape();
        for (size_t i = 0; i < shapes.size(); ++i) {
            auto tensor = std::make_shared<Tensor>(shapes[i], DataType::kFLOAT32, stage_->device());
            tensor->set_requires_grad(true);
            tensor->set_is_leaf(false);
            recv_tensors.push_back(tensor);
        }
        return IRecv(recv_tensors, stage_->device(), stage_->stage_index(), stage_->prev_rank());
    }
    return recv_tensors;
}

std::vector<std::shared_ptr<Tensor>> PipelineSchedule::SendToNext(const std::vector<std::shared_ptr<Tensor>> &tensors) {
    if (!stage_->IsLastStage()) {
        return ISend(tensors, stage_->device(), stage_->stage_index(), stage_->next_rank(), stage_->recv_shape());
    }
    return tensors;
}

float ScheduleGPipe::StepMicrobatches(const std::vector<std::shared_ptr<Tensor>> &microbatch_inputs,
                                      const std::vector<std::shared_ptr<Tensor>> &microbatch_targets,
                                      const std::shared_ptr<Module> &loss_fn) {
    const auto n = num_microbatches_;
    if (n == 0) {
        return 0.0f;
    }
    std::vector<std::vector<std::shared_ptr<Tensor>>> outputs(n);

    // ======== Forward Pass ========
    for (int mb = 0; mb < n; ++mb) {
        std::vector<std::shared_ptr<Tensor>> inputs;

        if (stage_->IsFirstStage()) {
            inputs = {microbatch_inputs[mb]};
        } else {
            inputs = ReceiveFromPrev();
        }

        outputs[mb] = stage_->ForwardOneChunk(inputs);

        for (auto &t : outputs[mb]) {
            if (!t) {
                t = std::make_shared<Tensor>((std::vector<int64_t>){}, DataType::kFLOAT32, stage_->device());
            }
        }

        outputs[mb] = SendToNext(outputs[mb]);
    }

    // ======== Backward Pass ========
    float total_loss = 0.0f;
    if (!stage_->IsLastStage()) {
        for (int mb = 0; mb < n; ++mb) {
            auto out_tensor = outputs[mb][0];

            auto gradient = std::make_shared<Tensor>(out_tensor->Dims(), out_tensor->Dtype(), out_tensor->GetDevice());

            out_tensor->Backward(gradient);
        }
    } else {
        for (int mb = 0; mb < n; ++mb) {
            auto target = microbatch_targets[mb];
            auto output = outputs[mb][0];

            if (!target || !output) {
                LOG(FATAL) << "Output or target is null at mb=" << mb;
            }

            auto target_on_device = target->To(output->GetDevice());
            auto loss = loss_fn->Forward({output, std::make_shared<Tensor>(target_on_device)})[0];
            if (!loss) {
                LOG(INFO) << "[ERROR] loss is nullptr at mb = " << mb;
                continue;
            }

            loss = loss / n;
            auto loss_cpu = loss->To(DeviceManager::Instance()->GetDefaultDevice());
            total_loss += static_cast<const float *>(loss_cpu.DataPtr())[0];

            loss->Backward();
        }
    }

    return total_loss;
}

} // namespace infini_train::nn::parallel
