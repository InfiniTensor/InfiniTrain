#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

#include "glog/logging.h"

#include <memory>

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/init.h"

namespace infini_train::nn::pipeline {

PipelineStage::PipelineStage(std::vector<std::shared_ptr<Module>> &layers, int stage_index, int num_stages,
                             const ActivationShape &recvShape, std::shared_ptr<Optimizer> optim)
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

    forward_outputs_.clear();
    printf("[stage %d] PipelineStage::ForwardOneChunk ENTRY!\n", stage_index_);
    if (IsFirstStage() && !layers_.empty()) {
        auto &idx = current[0]; // (bs, seq_len), dtype=INT64

        // 检查 dtype 判断是否是原始 token 输入
        if (idx->Dtype() == infini_train::DataType::kINT64) {
            const auto device = idx->GetDevice();
            const int64_t bs = idx->Dims()[0];
            const int64_t seq_len = idx->Dims()[1];

            // position ids
            auto pos = infini_train::nn::init::Arange(0, seq_len, infini_train::DataType::kINT64, device); // (seq_len)

            auto tok_emb = layers_[0]->Forward({idx})[0]; // Layer 0: WTE (token embedding)  (bs, seq_len, n_embd)
            auto pos_emb = layers_[1]->Forward({pos})[0]; // Layer 1: WPE (position embedding) (seq_len, n_embd)
            current[0] = tok_emb + pos_emb;
            for (int i = 2; i < layers_.size(); ++i) {
                auto outputs = layers_[i]->Forward(current);
                current = outputs;
            }
            return current;
        }
    }

    printf("[stage %d] before Forward %lf\n", stage_index_, static_cast<const float *>(current[0]->DataPtr()));
    for (int i = 0; i < layers_.size(); ++i) {
        printf("[stage %d] PipelineStage::  single layer Forward! layer %d\n", stage_index_, i);
        auto outputs = layers_[i]->Forward(current);
        printf("[stage %d] Forward OK!\n", stage_index_);
        current = outputs;
    }

    printf("[stage %d] PipelineStage::ForwardOneChunk OK! FLAG: %ld\n", stage_index_, current.size());
    return current;
}

} // namespace infini_train::nn::pipeline