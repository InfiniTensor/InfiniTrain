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

namespace lr_schedulers {
ConstantLR::ConstantLR(std::shared_ptr<Optimizer> optimizer, float factor, int total_iters, int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), factor_(factor), total_iters_(total_iters) {
    Step();
}

float ConstantLR::ComputeLR() {
    if(last_step_ < total_iters_) {
        return base_lr_ * factor_;
    }
    return base_lr_;
}

StepLR::StepLR(std::shared_ptr<Optimizer> optimizer, int64_t step_size, float gamma , int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), step_size_(step_size), gamma_(gamma) {
    Step();
}

float StepLR::ComputeLR() {
    return base_lr_ * static_cast<float>(std::pow(static_cast<double>(gamma_), 
                                                    static_cast<double>(last_step_ / step_size_)));
}

LinearWarmupLR::LinearWarmupLR(std::shared_ptr<Optimizer> optimizer, int64_t warmup_steps, float start_factor, int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), warmup_steps_(warmup_steps), start_factor_(start_factor) {
    Step();
}

float LinearWarmupLR::ComputeLR() {
    if (last_step_ >= warmup_steps_) {
        return base_lr_;
    }
    float alpha = static_cast<float>(last_step_) / static_cast<float>(warmup_steps_);
    return base_lr_ * ( start_factor_ + (1.0f - start_factor_) * alpha);
}


}  // namespace lr_schedulers
}  // namespace infini_train