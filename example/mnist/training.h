#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "example/mnist/net.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/dataset.h"
#include "infini_train/include/nn/init.h"
#include "infini_train/include/nn/modules/loss.h"

namespace mnist {
// Keep the shared DataLoader API unchanged. Reset from identity so each epoch's
// sample order depends only on its seed, not the previous epoch's permutation.
class ShuffledDataset : public infini_train::Dataset {
public:
    explicit ShuffledDataset(std::shared_ptr<infini_train::Dataset> source, int rank = 0, int world_size = 1,
                             bool equal_shards = false)
        : source_(std::move(source)), rank_(rank), world_size_(world_size), equal_shards_(equal_shards) {
        CHECK(source_);
        CHECK_GT(world_size_, 0);
        CHECK_GE(rank_, 0);
        CHECK_LT(rank_, world_size_);
        Reset(0, false);
    }
    void Reset(uint32_t seed, bool shuffle = true) {
        std::vector<size_t> global_order(source_->Size());
        std::iota(global_order.begin(), global_order.end(), size_t{0});
        if (shuffle) {
            std::mt19937 rng(seed);
            std::shuffle(global_order.begin(), global_order.end(), rng);
        }
        // Training drops at most world_size-1 samples, so every rank performs
        // the same number of collectives and local mean losses have equal weight.
        // Evaluation keeps every sample exactly once, including uneven shards.
        const size_t size = equal_shards_ ? global_order.size() / world_size_ * world_size_ : global_order.size();
        order_.clear();
        for (size_t i = rank_; i < size; i += world_size_) { order_.push_back(global_order[i]); }
    }
    size_t Size() const override { return order_.size(); }
    std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
    operator[](size_t i) const override {
        return (*source_)[order_.at(i)];
    }
    const std::vector<size_t> &order() const { return order_; }

private:
    std::shared_ptr<infini_train::Dataset> source_;
    std::vector<size_t> order_;
    int rank_;
    int world_size_;
    bool equal_shards_;
};

// Explicit CPU initialization avoids dependence on the global RNG or OpenMP
// worker count. The distribution matches the existing Conv2d/Linear defaults.
inline void Initialize(MNIST &net, uint32_t seed) {
    using namespace infini_train;
    std::mt19937 rng(seed);
    for (const auto *layer_name : {"conv1", "conv2", "fc"}) {
        auto &layer = net.mutable_module(layer_name);
        const auto [fan_in, fan_out] = nn::init::CalculateFanInAndFanOut(layer->parameter("weight"));
        const float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
        for (const auto *name : {"weight", "bias"}) {
            const auto &param = layer->parameter(name);
            CHECK(param->GetDevice().IsCPU()) << "Initialize MNIST before moving it to CUDA";
            auto *p = static_cast<float *>(param->DataPtr());
            for (size_t i = 0; i < param->NumElements(); ++i) {
                const float uniform = static_cast<float>(rng() >> 8) / 16777216.0f;
                p[i] = (2.0f * uniform - 1.0f) * bound;
            }
        }
    }
}

struct Metrics {
    double loss_sum = 0;
    int64_t correct = 0;
    int64_t samples = 0;
    void Add(float mean_loss, int64_t batch_correct, int64_t batch_size) {
        CHECK(std::isfinite(mean_loss)) << "Non-finite loss";
        CHECK_GT(batch_size, 0);
        CHECK_GE(batch_correct, 0);
        CHECK_LE(batch_correct, batch_size);
        loss_sum += static_cast<double>(mean_loss) * batch_size;
        correct += batch_correct;
        samples += batch_size;
    }
    double Loss() const {
        CHECK_GT(samples, 0);
        return loss_sum / samples;
    }
    double Accuracy() const {
        CHECK_GT(samples, 0);
        return static_cast<double>(correct) / samples;
    }
};

inline void Accumulate(Metrics &metrics, const std::shared_ptr<infini_train::Tensor> &logits,
                       const std::shared_ptr<infini_train::Tensor> &loss,
                       const std::shared_ptr<infini_train::Tensor> &cpu_labels) {
    using namespace infini_train;
    CHECK(cpu_labels->GetDevice().IsCPU());
    CHECK(cpu_labels->Dtype() == DataType::kUINT8);
    auto y = logits->To(Device());
    auto l = loss->To(Device());
    const auto device = logits->GetDevice();
    core::GetDeviceGuardImpl(device.type())->SynchronizeDevice(device);
    const auto n = y.Dims()[0];
    const auto classes = y.Dims()[1];
    const auto *values = static_cast<const float *>(y.DataPtr());
    const auto *labels = static_cast<const uint8_t *>(cpu_labels->DataPtr());
    CHECK_EQ(cpu_labels->NumElements(), n);
    int64_t correct = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float *row = values + i * classes;
        for (int64_t j = 0; j < classes; ++j) { CHECK(std::isfinite(row[j])) << "Non-finite logits"; }
        CHECK_LT(labels[i], classes);
        correct += (std::max_element(row, row + classes) - row) == labels[i];
    }
    metrics.Add(*static_cast<const float *>(l.DataPtr()), correct, n);
}

inline Metrics Evaluate(MNIST &net, const infini_train::DataLoader &loader, infini_train::Device device) {
    using namespace infini_train;
    autograd::NoGradGuard no_grad;
    nn::CrossEntropyLoss loss_fn;
    Metrics result;
    for (const auto &[image, label] : loader) {
        auto x = std::make_shared<Tensor>(image->To(device));
        auto target = std::make_shared<Tensor>(label->To(device));
        auto logits = net({x})[0];
        Accumulate(result, logits, loss_fn({logits, target})[0], label);
    }
    return result;
}
} // namespace mnist
