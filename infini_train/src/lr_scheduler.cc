#include "infini_train/include/lr_scheduler.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/optimizer.h"

namespace infini_train {

std::shared_ptr<LRScheduler> CreateLRScheduler(std::shared_ptr<Optimizer> optimizer,
                                               const TrainingLRSchedulerConfig &config) {
    if (config.lr_decay_style == "none") {
        return nullptr;
    }

    CHECK(optimizer) << "CreateLRScheduler: optimizer must not be null.";
    const float max_lr = config.lr != 0.0f ? config.lr : optimizer->learning_rate();
    CHECK_GT(max_lr, 0.0f) << "CreateLRScheduler: max_lr must be > 0.";
    CHECK_GE(config.lr_warmup_init, 0.0f) << "CreateLRScheduler: lr_warmup_init must be >= 0.";
    CHECK_GE(config.min_lr, 0.0f) << "CreateLRScheduler: min_lr must be >= 0.";
    CHECK_GE(max_lr, config.min_lr) << "CreateLRScheduler: max_lr must be >= min_lr.";
    CHECK_LE(config.lr_warmup_init, max_lr) << "CreateLRScheduler: lr_warmup_init must be <= max_lr.";
    CHECK_GE(config.lr_warmup_iters, 0) << "CreateLRScheduler: lr_warmup_iters must be >= 0.";
    CHECK_GT(config.lr_decay_iters, 0) << "CreateLRScheduler: lr_decay_iters must be > 0.";
    CHECK_LT(config.lr_warmup_iters, config.lr_decay_iters)
        << "CreateLRScheduler: lr_warmup_iters must be < lr_decay_iters.";
    CHECK(config.lr_decay_style == "constant" || config.lr_decay_style == "linear" || config.lr_decay_style == "cosine"
          || config.lr_decay_style == "inverse-square-root")
        << "CreateLRScheduler: unsupported lr_decay_style: " << config.lr_decay_style;

    std::shared_ptr<LRScheduler> main_scheduler;
    const int64_t decay_iters_after_warmup = config.lr_decay_iters - config.lr_warmup_iters;
    if (config.lr_decay_style == "constant") {
        main_scheduler = LRScheduler::Create<lr_schedulers::LambdaLR>(optimizer, [](int64_t) { return 1.0f; });
    } else if (config.lr_decay_style == "linear") {
        main_scheduler = LRScheduler::Create<lr_schedulers::LinearLR>(optimizer, 1.0f, config.min_lr / max_lr,
                                                                      decay_iters_after_warmup);
    } else if (config.lr_decay_style == "cosine") {
        main_scheduler = LRScheduler::Create<lr_schedulers::LambdaLR>(
            optimizer, [max_lr, min_lr = config.min_lr, decay_iters_after_warmup](int64_t step) {
                if (step > decay_iters_after_warmup) {
                    return min_lr / max_lr;
                }
                const float decay_ratio = static_cast<float>(step) / static_cast<float>(decay_iters_after_warmup);
                CHECK_GE(decay_ratio, 0.0f) << "CreateLRScheduler: decay "
                                               "ratio must be >= 0.";
                CHECK_LE(decay_ratio, 1.0f) << "CreateLRScheduler: decay "
                                               "ratio must be <= 1.";
                const float coeff = 0.5f * (std::cos(std::numbers::pi_v<float> * decay_ratio) + 1.0f);
                return (min_lr + coeff * (max_lr - min_lr)) / max_lr;
            });
    } else if (config.lr_decay_style == "inverse-square-root") {
        main_scheduler = LRScheduler::Create<lr_schedulers::LambdaLR>(
            optimizer, [max_lr, min_lr = config.min_lr, lr_warmup_iters = config.lr_warmup_iters,
                        lr_decay_iters = config.lr_decay_iters](int64_t step) {
                const int64_t global_step = step + lr_warmup_iters;
                if (global_step > lr_decay_iters) {
                    return min_lr / max_lr;
                }
                const auto warmup = static_cast<float>(std::max<int64_t>(lr_warmup_iters, 1));
                const auto current = static_cast<float>(std::max<int64_t>(global_step, 1));
                return std::max(min_lr, max_lr * std::sqrt(warmup) / std::sqrt(current)) / max_lr;
            });
    }

    CHECK(main_scheduler) << "CreateLRScheduler: failed to create scheduler.";
    if (config.lr_warmup_iters == 0) {
        return main_scheduler;
    }

    auto warmup_scheduler = LRScheduler::Create<lr_schedulers::LambdaLR>(
        optimizer,
        [lr_warmup_init = config.lr_warmup_init, max_lr, lr_warmup_iters = config.lr_warmup_iters](int64_t step) {
            const float warmup_ratio = static_cast<float>(step) / static_cast<float>(lr_warmup_iters);
            return (lr_warmup_init + (max_lr - lr_warmup_init) * warmup_ratio) / max_lr;
        });
    return LRScheduler::Create<lr_schedulers::SequentialLR>(
        std::move(optimizer), std::vector<std::shared_ptr<LRScheduler>>{warmup_scheduler, main_scheduler},
        std::vector<int64_t>{config.lr_warmup_iters});
}

LRScheduler::LRScheduler(std::shared_ptr<Optimizer> optimizer, int64_t last_step)
    : optimizer_(std::move(optimizer)), last_step_(last_step), base_lr_(0.0f) {
    CHECK(optimizer_) << "LRScheduler: optimizer must not be null.";
    optimizer_->set_initial_learning_rate(optimizer_->learning_rate());
    base_lr_ = optimizer_->initial_learning_rate();
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

void LRScheduler::ApplyLR(float lr) { optimizer_->set_learning_rate(lr); }

float LRScheduler::GetChainedFormLR() const { return GetClosedFormLR(); }

float LRScheduler::GetLR() const { return optimizer_->learning_rate(); }

float LRScheduler::BaseLR() const { return base_lr_; }

int64_t LRScheduler::LastStep() const { return last_step_; }

bool LRScheduler::SharesOptimizerWith(const std::shared_ptr<Optimizer> &opt) const { return optimizer_ == opt; }

void LRScheduler::ResetStep(int64_t step) { last_step_ = step; }

StateDict LRScheduler::State() const {
    return {
        {"last_step", last_step_},
        {"recover_lr", optimizer_->learning_rate()},
        {"base_lr", base_lr_},
    };
}

void LRScheduler::LoadState(const StateDict &state) {
    last_step_ = std::get<int64_t>(state.at("last_step"));
    recover_lr_ = std::get<float>(state.at("recover_lr"));
    base_lr_ = std::get<float>(state.at("base_lr"));
    optimizer_->set_learning_rate(recover_lr_);
}

// Concrete LR Schedulers

namespace lr_schedulers {

// --- ConstantLR ---

ConstantLR::ConstantLR(std::shared_ptr<Optimizer> optimizer, float factor, int total_iters, int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), factor_(factor), total_iters_(total_iters) {
    CHECK_GT(factor_, 0.0f) << "ConstantLR: factor must be > 0.";
    CHECK_LE(factor_, 1.0f) << "ConstantLR: factor must be <= 1.";
}

float ConstantLR::GetClosedFormLR() const { return last_step_ < total_iters_ ? base_lr_ * factor_ : base_lr_; }

float ConstantLR::GetChainedFormLR() const {
    const float lr = optimizer_->learning_rate();
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
    : LRScheduler(std::move(optimizer), last_step), step_size_(step_size), gamma_(gamma) {
    CHECK_GT(step_size_, 0) << "StepLR: step_size must be > 0.";
    CHECK_GT(gamma_, 0.0f) << "StepLR: gamma must be > 0.";
}

float StepLR::GetClosedFormLR() const {
    return base_lr_
         * static_cast<float>(std::pow(static_cast<double>(gamma_), static_cast<double>(last_step_ / step_size_)));
}

float StepLR::GetChainedFormLR() const {
    const float lr = optimizer_->learning_rate();
    if (last_step_ == 0 || (last_step_ % step_size_) != 0) {
        return lr;
    }
    return lr * gamma_;
}

LinearLR::LinearLR(std::shared_ptr<Optimizer> optimizer, float start_factor, float end_factor, int64_t total_iters,
                   int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), start_factor_(start_factor), end_factor_(end_factor),
      total_iters_(total_iters) {
    CHECK_GT(start_factor_, 0.0f) << "LinearLR: start_factor must be > 0.";
    CHECK_LE(start_factor_, 1.0f) << "LinearLR: start_factor must be <= 1.";
    CHECK_GE(end_factor_, 0.0f) << "LinearLR: end_factor must be >= 0.";
    CHECK_LE(end_factor_, 1.0f) << "LinearLR: end_factor must be <= 1.";
    CHECK_GT(total_iters_, 0) << "LinearLR: total_iters must be > 0.";
}

float LinearLR::GetClosedFormLR() const {
    if (last_step_ >= total_iters_) {
        return base_lr_ * end_factor_;
    }
    return base_lr_
         * (start_factor_
            + (end_factor_ - start_factor_) * static_cast<float>(last_step_) / static_cast<float>(total_iters_));
}

float LinearLR::GetChainedFormLR() const {
    const float lr = optimizer_->learning_rate();
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
    : LRScheduler(std::move(optimizer), last_step), lr_lambda_(std::move(lr_lambda)) {
    CHECK(lr_lambda_) << "LambdaLR: lr_lambda must not be null.";
}

float LambdaLR::GetClosedFormLR() const { return base_lr_ * lr_lambda_(last_step_); }

float SequentialLR::GetClosedFormLR() const {
    LOG(FATAL) << "SequentialLR does not support closed-form LR. Use Step() without an explicit epoch.";
    return base_lr_;
}

SequentialLR::SequentialLR(std::shared_ptr<Optimizer> optimizer, std::vector<std::shared_ptr<LRScheduler>> schedulers,
                           std::vector<int64_t> milestones, int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), schedulers_(std::move(schedulers)),
      milestones_(std::move(milestones)) {
    CHECK(!schedulers_.empty()) << "SequentialLR requires at least one scheduler.";

    for (size_t i = 0; i < schedulers_.size(); ++i) {
        CHECK(schedulers_[i]) << "SequentialLR: scheduler at index " << i << " must not be null.";
        CHECK(schedulers_[i]->SharesOptimizerWith(optimizer_))
            << "SequentialLR: scheduler at index " << i << " must share the same optimizer.";
    }

    CHECK_EQ(milestones_.size(), schedulers_.size() - 1)
        << "SequentialLR: milestones count must be schedulers count - 1.";

    for (size_t i = 1; i < milestones_.size(); ++i) {
        CHECK_GT(milestones_[i], milestones_[i - 1]) << "Milestones must be strictly increasing.";
    }
}

void SequentialLR::InitialStep() {

    optimizer_->set_learning_rate(schedulers_[0]->BaseLR());

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
    state["recover_lr"] = optimizer_->learning_rate();
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
    optimizer_->set_learning_rate(recover_lr_);
}

ChainedScheduler::ChainedScheduler(std::shared_ptr<Optimizer> optimizer,
                                   std::vector<std::shared_ptr<LRScheduler>> schedulers, int64_t last_step)
    : LRScheduler(std::move(optimizer), last_step), schedulers_(std::move(schedulers)) {
    CHECK(!schedulers_.empty()) << "ChainedScheduler requires at least one scheduler.";

    for (size_t i = 0; i < schedulers_.size(); ++i) {
        CHECK(schedulers_[i]) << "ChainedScheduler: scheduler at index " << i << " must not be null.";
        CHECK(schedulers_[i]->SharesOptimizerWith(optimizer_))
            << "ChainedScheduler: scheduler at index " << i << " must share the same optimizer.";
    }
}

float ChainedScheduler::GetClosedFormLR() const {
    LOG(FATAL) << "ChainedScheduler does not support closed-form LR. Use Step() without an explicit epoch.";
    return base_lr_;
}

void ChainedScheduler::InitialStep() { last_step_ = 0; }

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
