// pipeline_schedule.cc
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autocast.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"
#include "infini_train/include/nn/parallel/pp/send_recv.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

float PipelineSchedule::Step(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> target,
                             const std::shared_ptr<Module> &loss_fn, DataType dtype) {
    std::vector<std::shared_ptr<Tensor>> micro_batches(num_micro_batches_);
    std::vector<std::shared_ptr<Tensor>> target_mbs(num_micro_batches_);
    if (stage_->IsFirstStage()) {
        micro_batches = input->Split(input->Dims()[0] / num_micro_batches_);
    }

    if (stage_->IsLastStage()) {
        target_mbs = target->Split(target->Dims()[0] / num_micro_batches_);
    }

    const auto &optimizer = stage_->optimizer();

    optimizer->ZeroGrad();

    float lossf = StepMicroBatches(micro_batches, target_mbs, loss_fn, dtype);

    optimizer->Step();

    return lossf;
}

std::vector<std::shared_ptr<Tensor>> PipelineSchedule::ReceiveFromPrev() {
    std::vector<std::shared_ptr<Tensor>> recv_tensors;
    auto &shapes = stage_->recv_shape();
    for (size_t i = 0; i < shapes.size(); ++i) {
        // FIXME(jym): The data type between stages is not float32, which will cause a crash
        auto tensor = std::make_shared<Tensor>(shapes[i], DataType::kFLOAT32, stage_->device());
        tensor->set_requires_grad(true);
        recv_tensors.push_back(tensor);
    }

    return IRecv(recv_tensors, stage_->device(), stage_->stage_index(), stage_->prev_rank());
}

std::vector<std::shared_ptr<Tensor>> PipelineSchedule::SendToNext(const std::vector<std::shared_ptr<Tensor>> &tensors) {
    return ISend(tensors, stage_->device(), stage_->stage_index(), stage_->next_rank(), stage_->recv_shape());
}

float ScheduleGPipe::StepMicroBatches(const std::vector<std::shared_ptr<Tensor>> &microbatch_inputs,
                                      const std::vector<std::shared_ptr<Tensor>> &microbatch_targets,
                                      const std::shared_ptr<Module> &loss_fn, DataType dtype) {
    const auto n = num_micro_batches_;
    if (n == 0) {
        return 0.0f;
    }

    std::vector<std::vector<std::shared_ptr<Tensor>>> outputs(n);

    // ======== Forward Pass ========
    for (int mb = 0; mb < n; ++mb) {
        infini_train::AutocastGuard autocast_guard(stage_->device()->Type(), dtype);

        std::vector<std::shared_ptr<Tensor>> inputs;
        if (stage_->IsFirstStage()) {
            inputs = {microbatch_inputs[mb]};
        } else {
            inputs = ReceiveFromPrev();
        }

        outputs[mb] = stage_->ForwardOneChunk(inputs);

        if (!stage_->IsLastStage()) {
            outputs[mb] = SendToNext(outputs[mb]);
        }
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

            std::shared_ptr<Tensor> loss;
            {
                infini_train::AutocastGuard autocast_guard(stage_->device()->Type(), dtype);

                auto target_on_device = target->To(output->GetDevice());
                loss = loss_fn->Forward({output, std::make_shared<Tensor>(target_on_device)})[0];
                if (!loss) {
                    LOG(FATAL) << "[ERROR] loss is nullptr at mb = " << mb;
                }
                loss = loss / n;
            }

            auto loss_cpu = loss->To(DeviceManager::Instance()->GetDefaultDevice());
            total_loss += static_cast<const float *>(loss_cpu.DataPtr())[0];

            loss->Backward();
        }
    }

    return total_loss;
}

} // namespace infini_train::nn::parallel
