#include "infini_train/include/lr_scheduler.h"

#include "glog/logging.h"

#include "infini_train/include/optimizer.h"

namespace infini_train {

std::shared_ptr<LRScheduler> CreateLRScheduler(std::shared_ptr<Optimizer> optimizer, const LRSchedulerConfig &config) {
    if (config.type == "none") {
        return nullptr;
    }

    auto create_main = [&](std::shared_ptr<Optimizer> opt) -> std::shared_ptr<LRScheduler> {
        if (config.type == "constant") {
            return LRScheduler::Create<lr_schedulers::ConstantLR>(opt, config.constant_factor,
                                                                  config.constant_total_iters);
        }
        if (config.type == "step") {
            return LRScheduler::Create<lr_schedulers::StepLR>(opt, config.step_size, config.step_gamma);
        }
        if (config.type == "linear") {
            return LRScheduler::Create<lr_schedulers::LinearLR>(opt, config.linear_start_factor,
                                                                config.linear_end_factor, config.linear_total_iters);
        }
        if (config.type == "lambda") {
            return LRScheduler::Create<lr_schedulers::LambdaLR>(opt, config.lambda_fn);
        }
        if (config.type == "sequential") {
            std::vector<std::shared_ptr<LRScheduler>> schedulers;
            std::vector<int64_t> milestones = config.sequential_milestones;
            for (const auto &sub_config : config.sequential_configs) {
                auto sub_sched = CreateLRScheduler(opt, sub_config);
                if (sub_sched) {
                    schedulers.push_back(sub_sched);
                }
            }
            return LRScheduler::Create<lr_schedulers::SequentialLR>(opt, schedulers, milestones);
        }
        if (config.type == "chained") {
            std::vector<std::shared_ptr<LRScheduler>> schedulers;
            for (const auto &sub_config : config.chained_configs) {
                auto sub_sched = CreateLRScheduler(opt, sub_config);
                if (sub_sched) {
                    schedulers.push_back(sub_sched);
                }
            }
            return LRScheduler::Create<lr_schedulers::ChainedScheduler>(opt, schedulers);
        }
        LOG(FATAL) << "Unsupported LR scheduler type: " << config.type;
        return nullptr;
    };

    if (config.warmup_steps <= 0) {
        return create_main(optimizer);
    }

    auto warmup_scheduler = LRScheduler::Create<lr_schedulers::LinearLR>(optimizer,
                                                                         /*start_factor=*/config.warmup_start_factor,
                                                                         /*end_factor=*/config.warmup_end_factor,
                                                                         /*total_iters=*/config.warmup_steps);

    auto main_scheduler = create_main(optimizer);

    return LRScheduler::Create<lr_schedulers::SequentialLR>(
        optimizer, std::vector<std::shared_ptr<LRScheduler>>{warmup_scheduler, main_scheduler},
        std::vector<int64_t>{config.warmup_steps});
};

LRScheduler::LRScheduler(std::shared_ptr<Optimizer> optimizer, int64_t last_step)
    : optimizer_(std::move(optimizer)), last_step_(last_step), base_lr_(0.0f) {
    CHECK(optimizer_) << "LRScheduler: optimizer must not be null.";
    optimizer_->SetInitialLearningRate(optimizer_->GetLearningRate());
    base_lr_ = optimizer_->GetInitialLearningRate();
}

void LRScheduler::Step() {
    ++last_step_;
    ApplyLR(GetChainedFormLR());
}

void LRScheduler::Step(int64_t epoch) {
    last_step_ = epoch;
    ApplyLR(GetClosedFormLR());
}

void LRScheduler::InitialStep() {
    is_initial_ = true;
    Step();
    is_initial_ = false;
}

void LRScheduler::ApplyLR(float lr) {
    optimizer_->SetLearningRate(lr);
}

float LRScheduler::GetChainedFormLR() const { return GetClosedFormLR(); }

float LRScheduler::GetLR() const { return optimizer_->GetLearningRate(); }

float LRScheduler::BaseLR() const { return base_lr_; }

int64_t LRScheduler::LastStep() const { return last_step_; }

void LRScheduler::ResetStep(int64_t step) { last_step_ = step; }

StateDict LRScheduler::State() const {
    return {
        {"last_step", last_step_},
        {"recover_lr", optimizer_->GetLearningRate()},
        {"base_lr", base_lr_},
    };
}

void LRScheduler::LoadState(const StateDict &state) {
    last_step_ = std::get<int64_t>(state.at("last_step"));
    recover_lr_ = std::get<float>(state.at("recover_lr"));
    base_lr_ = std::get<float>(state.at("base_lr"));
    optimizer_->SetLearningRate(recover_lr_);
}

// Concrete LR Schedulers

namespace lr_schedulers {

// --- ConstantLR ---

ConstantLR::ConstantLR(std::shared_ptr<Optimizer> optimizer, float factor, int total_iters, int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), factor_(factor), total_iters_(total_iters) {}

float ConstantLR::GetClosedFormLR() const { return last_step_ < total_iters_ ? base_lr_ * factor_ : base_lr_; }

float ConstantLR::GetChainedFormLR() const {
    const float lr = optimizer_->GetLearningRate();
    if (last_step_ == 0) {
        return lr * factor_;
    } else if (last_step_ < total_iters_) {
        return lr;
    } else if (last_step_ == total_iters_) {
        return lr / factor_;
    }
    return lr;
}

// --- StepLR ---

StepLR::StepLR(std::shared_ptr<Optimizer> optimizer, int64_t step_size, float gamma, int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), step_size_(step_size), gamma_(gamma) {}

float StepLR::GetClosedFormLR() const {
    return base_lr_
         * static_cast<float>(std::pow(static_cast<double>(gamma_), static_cast<double>(last_step_ / step_size_)));
}

float StepLR::GetChainedFormLR() const {
    const float lr = optimizer_->GetLearningRate();
    if (last_step_ == 0 || (last_step_ % step_size_) != 0) {
        return lr;
    }
    return lr * gamma_;
}

LinearLR::LinearLR(std::shared_ptr<Optimizer> optimizer, float start_factor, float end_factor, int64_t total_iters,
                   int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), start_factor_(start_factor), end_factor_(end_factor),
      total_iters_(total_iters) {}

float LinearLR::GetClosedFormLR() const {
    if (last_step_ >= total_iters_) {
        return base_lr_ * end_factor_;
    }
    return base_lr_
         * (start_factor_
            + (end_factor_ - start_factor_) * static_cast<float>(last_step_) / static_cast<float>(total_iters_));
}

float LinearLR::GetChainedFormLR() const {
    const float lr = optimizer_->GetLearningRate();
    if (last_step_ == 0) {
        return lr * start_factor_;
    }
    if (last_step_ > total_iters_ || is_initial_) {
        return lr;
    }
    if (last_step_ == total_iters_) {
        const float prev_factor
            = start_factor_
            + (end_factor_ - start_factor_) * static_cast<float>(total_iters_ - 1) / static_cast<float>(total_iters_);
        return lr * (end_factor_ / prev_factor);
    }

    const float numerator = end_factor_ - start_factor_;
    const float denominator
        = start_factor_ * static_cast<float>(total_iters_) + static_cast<float>(last_step_ - 1) * numerator;
    return lr * (1.0f + numerator / denominator);
}

LambdaLR::LambdaLR(std::shared_ptr<Optimizer> optimizer, std::function<float(int64_t)> lr_lambda, int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), lr_lambda_(std::move(lr_lambda)) {}

float LambdaLR::GetClosedFormLR() const { return base_lr_ * lr_lambda_(last_step_); }

SequentialLR::SequentialLR(std::shared_ptr<Optimizer> optimizer, std::vector<std::shared_ptr<LRScheduler>> schedulers,
                           std::vector<int64_t> milestones, int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), schedulers_(std::move(schedulers)),
      milestones_(std::move(milestones)) {}

void SequentialLR::InitialStep() {
    CHECK(!schedulers_.empty()) << "SequentialLR requires at least one scheduler.";
    CHECK_EQ(milestones_.size(), schedulers_.size() - 1)
        << "SequentialLR: milestones count must be schedulers count - 1.";

    for (size_t i = 1; i < milestones_.size(); ++i) {
        CHECK_GT(milestones_[i], milestones_[i - 1]) << "Milestones must be strictly increasing.";
    }

    optimizer_->SetLearningRate(schedulers_[0]->BaseLR());

    UndoChildInitialSteps();

    ++last_step_;
    schedulers_[0]->InitialStep();
}

void SequentialLR::UndoChildInitialSteps() {
    for (auto &sched : schedulers_) {
        if (auto nested = std::dynamic_pointer_cast<SequentialLR>(sched)) {
            nested->UndoChildInitialSteps();
        }
        sched->ResetStep(sched->LastStep() - 1);
    }
}

void SequentialLR::Step() {
    ++last_step_;
    size_t idx = std::upper_bound(milestones_.begin(), milestones_.end(), last_step_) - milestones_.begin();

    auto &scheduler = schedulers_[idx];

    if (idx > 0 && milestones_[idx - 1] == last_step_) {
        scheduler->Step(0);
    } else {
        scheduler->Step();
    }

}

StateDict SequentialLR::State() const {
    StateDict state;
    state["last_step"] = last_step_;
    state["recover_lr"] = optimizer_->GetLearningRate();
    state["base_lr"] = base_lr_;
    for (size_t i = 0; i < schedulers_.size(); ++i) {
        auto sub_state = schedulers_[i]->State();
        for (const auto &[key, value] : sub_state) { state["scheduler_" + std::to_string(i) + "." + key] = value; }
    }
    return state;
}

void SequentialLR::LoadState(const StateDict &state) {
    last_step_ = std::get<int64_t>(state.at("last_step"));
    recover_lr_ = std::get<float>(state.at("recover_lr"));
    base_lr_ = std::get<float>(state.at("base_lr"));

    for (size_t i = 0; i < schedulers_.size(); ++i) {
        StateDict sub_state;
        std::string prefix = "scheduler_" + std::to_string(i) + ".";
        for (const auto &[key, value] : state) {
            if (key.substr(0, prefix.size()) == prefix) {
                sub_state[key.substr(prefix.size())] = value;
            }
        }
        if (!sub_state.empty()) {
            schedulers_[i]->LoadState(sub_state);
        }
    }
    optimizer_->SetLearningRate(recover_lr_);
}

ChainedScheduler::ChainedScheduler(std::shared_ptr<Optimizer> optimizer,
                                   std::vector<std::shared_ptr<LRScheduler>> schedulers, int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), schedulers_(std::move(schedulers)) {}

void ChainedScheduler::InitialStep() {
    CHECK(!schedulers_.empty()) << "ChainedScheduler requires at least one scheduler.";

}

void ChainedScheduler::Step() {
    ++last_step_;
    for (auto &sched : schedulers_) { sched->Step(); }
}

StateDict ChainedScheduler::State() const {
    StateDict state = LRScheduler::State();
    for (size_t i = 0; i < schedulers_.size(); ++i) {
        auto sub_state = schedulers_[i]->State();
        for (const auto &[key, value] : sub_state) { state["scheduler_" + std::to_string(i) + "." + key] = value; }
    }
    return state;
}

void ChainedScheduler::LoadState(const StateDict &state) {
    LRScheduler::LoadState(state);
    for (size_t i = 0; i < schedulers_.size(); ++i) {
        StateDict sub_state;
        std::string prefix = "scheduler_" + std::to_string(i) + ".";
        for (const auto &[key, value] : state) {
            if (key.substr(0, prefix.size()) == prefix) {
                sub_state[key.substr(prefix.size())] = value;
            }
        }
        if (!sub_state.empty()) {
            schedulers_[i]->LoadState(sub_state);
        }
    }
}

} // namespace lr_schedulers
} // namespace infini_train
