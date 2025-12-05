#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

#include <memory>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/process_group.h"

namespace infini_train::nn::parallel {

PipelineStage::PipelineStage(const std::shared_ptr<Module> &model, int stage_index /* pp_rank */,
                             int num_stages /* pp_size */, const std::vector<std::vector<int64_t>> &recv_shape,
                             std::shared_ptr<Optimizer> optimizer, int device_id)
    : model_(model), stage_index_(stage_index), num_stages_(num_stages),
      prev_rank_(stage_index > 0 ? stage_index - 1 : -1),
      next_rank_(stage_index < num_stages - 1 ? stage_index + 1 : -1), recv_shape_(recv_shape),
      optimizer_(std::move(optimizer)),
      device_(DeviceManager::Instance()->GetAllAvailableDevices(DeviceType::kCUDA).at(device_id)) {}

std::vector<std::shared_ptr<Tensor>>
PipelineStage::ForwardOneChunk(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    return model_->ForwardChunk(0, inputs);
}

} // namespace infini_train::nn::parallel
