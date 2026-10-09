// Real two-process NCCL integration test, including a partial final batch.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "example/mnist/distributed.h"
#include "gflags/gflags.h"
#include "infini_train/include/optimizer.h"

DEFINE_bool(ddp_buckets, true, "Test bucketed or per-parameter DDP reduction");
using namespace infini_train;

namespace {
std::vector<float> Values(const std::shared_ptr<Tensor> &tensor) {
    CHECK(tensor);
    auto cpu = tensor->To(Device());
    const auto device = tensor->GetDevice();
    core::GetDeviceGuardImpl(device.type())->SynchronizeDevice(device);
    const auto *p = static_cast<const float *>(cpu.DataPtr());
    return {p, p + cpu.NumElements()};
}

void Compare(const std::vector<float> &candidate, const std::vector<float> &reference, float atol, float rtol,
             const std::string &name) {
    CHECK_EQ(candidate.size(), reference.size());
    for (size_t i = 0; i < candidate.size(); ++i) {
        CHECK(std::isfinite(candidate[i]) && std::isfinite(reference[i])) << name;
        CHECK_LE(std::abs(candidate[i] - reference[i]), atol + rtol * std::abs(reference[i]))
            << name << " element=" << i << " candidate=" << candidate[i] << " reference=" << reference[i];
    }
}
} // namespace

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    mnist::DistributedContext context(true, true);
    CHECK_EQ(context.world_size, 2);
    auto net = std::make_shared<MNIST>();
    auto reference = std::make_shared<MNIST>();
    mnist::Initialize(*net, 42 + context.rank); // Deliberately different before broadcast.
    mnist::Initialize(*reference, 42);
    net->To(context.device);
    reference->To(context.device);
    auto model = context.Wrap(net, FLAGS_ddp_buckets);
    const auto params = net->NamedParameters(), ref_params = reference->NamedParameters();
    for (size_t i = 0; i < params.size(); ++i) {
        CHECK_EQ(params[i].first, ref_params[i].first);
        Compare(Values(params[i].second), Values(ref_params[i].second), 0, 0, "broadcast");
    }
    optimizers::SGD optimizer(net->Parameters(), 0.01f), ref_optimizer(reference->Parameters(), 0.01f);
    nn::CrossEntropyLoss criterion;
    for (int batch : {6, 2, 6}) {
        const int local = batch / 2;
        std::vector<float> data(batch * 784), targets(batch);
        for (size_t i = 0; i < data.size(); ++i) { data[i] = static_cast<float>((i * 17 + 5) % 251) / 255; }
        for (int i = 0; i < batch; ++i) { targets[i] = (i * 3 + 1) % 10; }
        auto x = std::make_shared<Tensor>(data.data() + context.rank * local * 784, std::vector<int64_t>{local, 784},
                                          DataType::kFLOAT32, context.device)
                     ->RequiresGrad();
        auto all_x = std::make_shared<Tensor>(data.data(), std::vector<int64_t>{batch, 784}, DataType::kFLOAT32,
                                              context.device)
                         ->RequiresGrad();
        auto labels_float
            = std::make_shared<Tensor>(targets.data(), std::vector<int64_t>{batch}, DataType::kFLOAT32, context.device);
        auto all_labels = std::make_shared<Tensor>(labels_float->To(DataType::kINT64));
        auto labels = std::make_shared<Tensor>(*all_labels, context.rank * local * sizeof(int64_t),
                                               std::vector<int64_t>{local});
        optimizer.ZeroGrad();
        ref_optimizer.ZeroGrad();
        auto logits = (*model)({x})[0];
        auto ref_logits = (*reference)({all_x})[0];
        auto ref_values = Values(ref_logits);
        std::vector<float> slice(ref_values.begin() + context.rank * local * 10,
                                 ref_values.begin() + (context.rank + 1) * local * 10);
        Compare(Values(logits), slice, 1e-5f, 1e-4f, "logits");
        auto loss = criterion({logits, labels})[0], ref_loss = criterion({ref_logits, all_labels})[0];
        mnist::Metrics local_metrics;
        local_metrics.Add(Values(loss)[0], 0, local);
        auto global_metrics = context.Sum(local_metrics);
        CHECK_EQ(global_metrics.samples, batch);
        CHECK_LE(std::abs(global_metrics.Loss() - Values(ref_loss)[0]), 1e-6);
        loss->Backward();
        ref_loss->Backward();
        auto local_grad = Values(x->grad()), global_grad = Values(all_x->grad());
        for (auto &v : local_grad) { v /= 2; }
        slice.assign(global_grad.begin() + context.rank * local * 784,
                     global_grad.begin() + (context.rank + 1) * local * 784);
        Compare(local_grad, slice, 1e-6f, 1e-3f, "input gradient");
        for (size_t i = 0; i < params.size(); ++i) {
            Compare(Values(params[i].second->grad()), Values(ref_params[i].second->grad()), 1e-6f, 1e-3f,
                    params[i].first + " gradient");
        }
        optimizer.Step();
        ref_optimizer.Step();
        for (size_t i = 0; i < params.size(); ++i) {
            Compare(Values(params[i].second), Values(ref_params[i].second), 1e-6f, 1e-5f, params[i].first + " updated");
        }
    }
    // Weighted reduction also handles unequal and empty evaluation shards.
    auto uneven = context.Sum(context.rank == 0 ? mnist::Metrics{6.0, 2, 3} : mnist::Metrics{4.0, 1, 1});
    CHECK_EQ(uneven.samples, 4);
    CHECK_EQ(uneven.correct, 3);
    CHECK_EQ(uneven.Loss(), 2.5);
    auto empty = context.Sum(context.rank == 0 ? mnist::Metrics{6.0, 2, 3} : mnist::Metrics{});
    CHECK_EQ(empty.samples, 3);
    CHECK_EQ(empty.Loss(), 2.0);
    context.Barrier();
    std::cout << "PASS rank=" << context.rank << " device=" << static_cast<int>(context.device.index())
              << " buckets=" << FLAGS_ddp_buckets << " broadcast/logits/loss/gradients/SGD/tails/metrics" << std::endl;
    return 0;
}
