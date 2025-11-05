#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

#include "glog/logging.h"

#include <memory>

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/init.h"

namespace infini_train::nn::parallel {

PipelineStage::PipelineStage(const std::vector<std::shared_ptr<Module>> &layers, int stage_index, int num_stages,
                             const std::vector<std::vector<int64_t>> &recvShape, std::shared_ptr<Optimizer> optim)
    : stage_index_(stage_index), num_stages_(num_stages), layers_(layers),
      prev_rank_(stage_index > 0 ? stage_index - 1 : -1),
      next_rank_(stage_index < num_stages - 1 ? stage_index + 1 : -1), recv_shape_(recvShape), optim_(std::move(optim)),
      device_(DeviceManager::Instance()->GetAllAvailableDevices(DeviceType::kCUDA).at(stage_index)) {}

std::vector<std::shared_ptr<Tensor>>
PipelineStage::ForwardOneChunk(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    std::vector<std::shared_ptr<Tensor>> current = inputs;
    int i = 0;
    for (const auto &layer : layers_) {
        current = layer->Forward(current);
        ++i;
    }

    return current;
}

} // namespace infini_train::nn::parallel
