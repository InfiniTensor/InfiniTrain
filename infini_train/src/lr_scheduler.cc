#include "infini_train/include/lr_scheduler.h"

#include "glog/logging.h"

#include "infini_train/include/optimizer.h"

namespace infini_train {

LRScheduler::LRScheduler(std::shared_ptr<Optimizer> optimizer,
                           int64_t last_step)
    : optimizer_(std::move(optimizer)),
      last_step_(last_step),
      current_lr_(0.0f),
      base_lr_(0.0f) {
    CHECK(optimizer_) << "LRScheduler: optimizer must not be null.";
    base_lr_ = optimizer_->GetLearningRate();
    current_lr_ = base_lr_;
}

void LRScheduler::Step() {
    ++last_step_;
    current_lr_ = ComputeLR();
    optimizer_->SetLearningRate(current_lr_);
}

float LRScheduler::GetLR() const { return current_lr_; }

int64_t LRScheduler::LastStep() const { return last_step_; }

StateDict LRScheduler::State() const {
    return {
        {"last_step", last_step_},
        {"current_lr", current_lr_},
        {"base_lr", base_lr_},
    };
}

void LRScheduler::LoadState(const StateDict &state) {
    last_step_ = std::get<int64_t>(state.at("last_step"));
    current_lr_ = std::get<float>(state.at("current_lr"));
    base_lr_ = std::get<float>(state.at("base_lr"));
    optimizer_->SetLearningRate(current_lr_);
}

}  // namespace infini_train