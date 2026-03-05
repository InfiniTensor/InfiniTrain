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
    optimizer_->SetInitialLearningRate(optimizer_->GetLearningRate());
    base_lr_ = optimizer_->GetInitialLearningRate();
    current_lr_ = base_lr_;
}

void LRScheduler::Step() {
    ++last_step_;
    ApplyLR(ComputeLR());
}

void LRScheduler::ApplyLR(float lr) {
    current_lr_ = lr;
    optimizer_->SetLearningRate(current_lr_);
}

float LRScheduler::GetLR() const { return current_lr_; }

float LRScheduler::BaseLR() const { return base_lr_; }

int64_t LRScheduler::LastStep() const { return last_step_; }

void LRScheduler::ResetStep(int64_t step) { last_step_ = step; }

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



// Concrete LR Schedulers

namespace lr_schedulers {

// --- ConstantLR --- 

ConstantLR::ConstantLR(std::shared_ptr<Optimizer> optimizer, 
                       float factor, 
                       int total_iters, 
                       int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), 
      factor_(factor), 
      total_iters_(total_iters) {
    Step();
}

float ConstantLR::ComputeLR() const {
    return last_step_ < total_iters_ ? base_lr_ * factor_ : base_lr_;
}

// --- StepLR ---

StepLR::StepLR(std::shared_ptr<Optimizer> optimizer,
               int64_t step_size,
               float gamma,
               int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step),
      step_size_(step_size),
      gamma_(gamma) {
    Step();
}

float StepLR::ComputeLR() const {
  return base_lr_ * static_cast<float>(std::pow(
             static_cast<double>(gamma_),
             static_cast<double>(last_step_ / step_size_)));
}

LinearWarmupLR::LinearWarmupLR(std::shared_ptr<Optimizer> optimizer, 
                               int64_t warmup_steps, 
                               float start_factor, 
                               int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), 
                  warmup_steps_(warmup_steps), 
                  start_factor_(start_factor) {
    Step();
}

float LinearWarmupLR::ComputeLR() const {
  if (last_step_ >= warmup_steps_) {
    return base_lr_;
  }
  float alpha =
      static_cast<float>(last_step_) / static_cast<float>(warmup_steps_);
  return base_lr_ * (start_factor_ + (1.0f - start_factor_) * alpha);
}

LambdaLR::LambdaLR(std::shared_ptr<Optimizer> optimizer, 
                   std::function<float(int64_t)> lr_lambda, 
                   int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), 
                  lr_lambda_(std::move(lr_lambda)) {
    Step();
}

float LambdaLR::ComputeLR() const {
  return base_lr_ * lr_lambda_(last_step_);
}

SequentialLR::SequentialLR(std::shared_ptr<Optimizer> optimizer,
                           std::vector<std::shared_ptr<LRScheduler>> schedulers, 
                           std::vector<int64_t>milestones, int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), 
                  schedulers_(std::move(schedulers)), 
                  milestones_(std::move(milestones)) {
    CHECK(!schedulers_.empty()) 
        << "SequentialLR requires at least one scheduler.";
    CHECK_EQ(milestones_.size(), schedulers_.size() - 1)
        << "SequentialLR: milestones count must be schedulers count - 1.";

    for(size_t i = 1; i < milestones_.size(); ++i) {
        CHECK_GT(milestones_[i], milestones_[i-1]) 
            << "Milestones must be strictly increasing.";
    }

    optimizer_->SetLearningRate(schedulers_[0]->BaseLR());

    // Reset all schedulers to the same last_step so they are in sync when Step() is called.
    for (auto &sched : schedulers_) {
        sched->ResetStep(sched->LastStep()-1);
    }

    Step();
}

void SequentialLR::Step() {
    ++last_step_;
    size_t idx = 0;
    for (size_t i = 0; i < milestones_.size(); ++i) {
        if (last_step_ >= milestones_[i]) {
            idx = i + 1;
        } else {
            break;
        }
    }

    auto &scheduler = schedulers_[idx];

    if (idx > 0 && milestones_[idx - 1] == last_step_) {
        scheduler->ResetStep(-1);
        scheduler->Step();
    } else {
        scheduler->Step();
    }

    current_lr_ = scheduler->GetLR();
}

StateDict SequentialLR::State() const {
    StateDict state;
    state["last_step"] = last_step_;
    state["current_lr"] = current_lr_;
    state["base_lr"] = base_lr_;
    for (size_t i = 0; i < schedulers_.size(); ++i) {
        auto sub_state = schedulers_[i]->State();
        for (const auto &[key, value] : sub_state) {
            state["scheduler_" + std::to_string(i) + "." + key] = value;
        }
    }
    return state;
}

void SequentialLR::LoadState(const StateDict &state) {
    last_step_ = std::get<int64_t>(state.at("last_step"));
    current_lr_ = std::get<float>(state.at("current_lr"));
    base_lr_ = std::get<float>(state.at("base_lr"));

    for (size_t i = 0; i < schedulers_.size(); ++i) {
        StateDict sub_state;
        std::string prefix = "scheduler_" + std::to_string(i) + ".";
        for (const auto &[key, value] : state) {
            if (key.substr(0, prefix.size()) == prefix) {
                sub_state[key.substr(prefix.size())] = value;
            }
        }
        if(!sub_state.empty())
            schedulers_[i]->LoadState(sub_state);
    }
    optimizer_->SetLearningRate(current_lr_);
}


}  // namespace lr_schedulers
}  // namespace infini_train