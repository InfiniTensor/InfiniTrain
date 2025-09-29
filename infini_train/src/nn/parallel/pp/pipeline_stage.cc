#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

#include "glog/logging.h"

#include <memory>

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/init.h"

namespace infini_train::nn::pipeline {
void PrintTensorSummary(const std::shared_ptr<Tensor>& tensor, const std::string& tag) {
    // printf("PrintTensorSummary FLAG1!!!!\n");
    if (!tensor) {
        printf("[TENSOR] %s: NULL tensor\n", tag.c_str());
        return;
    }

    // 确保是 CUDA tensor
    // auto cuda_device = dynamic_cast<CudaDevice*>(tensor->GetDevice());
    // if (cuda_device) {
    //     // 同步 stream，确保数据 ready
    //     cudaStreamSynchronize(cuda_device->Stream());
    // }

    // 拷贝到 CPU
    auto cpu_tensor = tensor->To(DeviceManager::Instance()->GetDefaultDevice());

    float* data = static_cast<float*>(cpu_tensor.DataPtr());
    size_t num_elements = cpu_tensor.NumElements();

    // 计算 min, max, mean
    float min_val = data[0], max_val = data[0], sum = 0.0f;
    int nan_count = 0, inf_count = 0;

    for (size_t i = 0; i < num_elements; ++i) {
        float v = data[i];
        if (std::isnan(v)) nan_count++;
        else if (std::isinf(v)) inf_count++;
        else {
            min_val = std::min(min_val, v);
            max_val = std::max(max_val, v);
            sum += v;
        }
    }

    float mean_val = (num_elements - nan_count - inf_count) > 0 ? 
                     sum / (num_elements - nan_count - inf_count) : 0.0f;

    std::string shape_str = VectorToString(cpu_tensor.Dims());
    printf("[TENSOR] %s: shape=[%s], "
           "min=%.6f, max=%.6f, mean=%.6f, "
           "nan=%d, inf=%d, total=%zu\n",
           tag.c_str(),
           shape_str.c_str(),
           min_val, max_val, mean_val, nan_count, inf_count, num_elements);
}

PipelineStage::PipelineStage(std::vector<std::shared_ptr<Module>> &layers, int stage_index, int num_stages,
                            const std::vector<std::vector<int64_t>> &recvShape, std::shared_ptr<Optimizer> optim)
    : stage_index_(stage_index), num_stages_(num_stages), layers_(layers),
      prev_rank_(stage_index > 0 ? stage_index - 1 : -1),
      next_rank_(stage_index < num_stages - 1 ? stage_index + 1 : -1),
      devices_(DeviceManager::Instance()->GetAllAvailableDevices(DeviceType::kCUDA)), recv_shape_(recvShape),
      optim_(optim) {
    printf("PipelineStage entry!! %d %ld\n", stage_index, devices_.size());
    device_ = devices_.at(stage_index);
}

std::vector<std::shared_ptr<Tensor>>
PipelineStage::ForwardOneChunk(const std::vector<std::shared_ptr<Tensor>> &inputs) {
    std::vector<std::shared_ptr<Tensor>> current = inputs;

    for (int i = 0; i < layers_.size(); ++i) {
        printf("[stage %d] PipelineStage::  single layer Forward! layer %d\n", stage_index_, i);
        auto outputs = layers_[i]->Forward(current);

        current = outputs;

        // PrintTensorSummary(outputs[0], "stage" + std::to_string(stage_index_) + "_mb" + std::to_string(i) + "_send_pre");
    }

    // printf("[stage %d] PipelineStage::ForwardOneChunk OK! FLAG: %ld\n", stage_index_, current.size());
    return current;
}

} // namespace infini_train::nn::pipeline