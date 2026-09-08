#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <format>
#include <memory>
#include <thread>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#if defined(USE_CUDA)
#include <cuda_runtime_api.h>
#endif

#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/optimizer.h"

#include "example/mnist/dataset.h"
#include "example/mnist/net.h"

DEFINE_string(dataset, "", "MNIST dataset path");
DEFINE_int32(bs, 64, "Per-rank batch size");
DEFINE_int32(num_epoch, 5, "Number of training epochs");
DEFINE_double(lr, 0.01, "Learning rate");
DEFINE_string(device, "cpu", "Device type (cpu/cuda)");
DEFINE_int32(nthread_per_process, 1, "Training threads per process; values greater than one enable CUDA DDP");

using namespace infini_train;

namespace {
constexpr int kNumItersOfOutputDuration = 10;
constexpr int kNumClasses = 10;

constexpr char kDeviceCPU[] = "cpu";
constexpr char kDeviceCUDA[] = "cuda";

std::array<float, 3> ReduceMetrics(std::array<float, 3> metrics, Device device,
                                   const nn::parallel::ProcessGroup *ddp_pg) {
    if (ddp_pg == nullptr) {
        return metrics;
    }

    auto device_metrics = std::make_shared<Tensor>(metrics.data(), std::vector<int64_t>{3}, DataType::kFLOAT32, device);
    nn::parallel::function::AllReduce(device_metrics, nn::parallel::function::ReduceOpType::kSum, ddp_pg);
    const Tensor host_metrics = device_metrics->To(Device());
    const auto *values = static_cast<const float *>(host_metrics.DataPtr());
    return {values[0], values[1], values[2]};
}

void Train(const nn::parallel::Rank &rank, const std::shared_ptr<MNIST> &network,
           const std::shared_ptr<MNISTDataset> &train_dataset, const std::shared_ptr<MNISTDataset> &test_dataset) {
    using namespace nn::parallel;

    global::thread_global_rank = rank.GlobalRank();
    const int ddp_world_size = global::GetDataParallelSize();
    int ddp_rank = 0;
    const ProcessGroup *ddp_pg = nullptr;

    Device device;
    if (rank.IsParallel()) {
        device = Device(Device::DeviceType::kCUDA, global::GetDeviceIndex(rank.thread_rank()));
        if (ddp_world_size > 1) {
            auto *pg_factory = ProcessGroupFactory::Instance(device.type());
            ddp_pg = pg_factory->GetOrCreate(GetDataParallelProcessGroupName(rank.GlobalRank()),
                                             GetDataParallelGroupRanks(rank.GlobalRank()));
            ddp_rank = ddp_pg->GetGroupRank(rank.GlobalRank());
        }
    } else {
        device = FLAGS_device == kDeviceCPU ? Device() : Device(Device::DeviceType::kCUDA, 0);
    }
    const Device cpu_device;

    network->To(device);
    if (ddp_pg != nullptr) {
        // Independent replicas are constructed serially in main; synchronize their initial parameters before DDP hooks.
        ddp_pg->Broadcast(network->Parameters(), /*root_rank_in_group=*/0);
    }

    std::shared_ptr<nn::Module> model = network;
    if (ddp_pg != nullptr) {
        model = std::make_shared<DistributedDataParallel>(model, rank, DistributedDataParallelConfig{});
    }

    nn::CrossEntropyLoss loss_fn;
    loss_fn.To(device);
    auto optimizer = optimizers::SGD(model->Parameters(), FLAGS_lr);
    DistributedDataLoader train_dataloader(train_dataset, FLAGS_bs, ddp_rank, ddp_world_size);
    DistributedDataLoader test_dataloader(test_dataset, FLAGS_bs, ddp_rank, ddp_world_size);

    const size_t full_train_batches = train_dataset->Size() / FLAGS_bs;
    const size_t train_steps
        = ddp_world_size > 1 ? full_train_batches / ddp_world_size : (train_dataset->Size() + FLAGS_bs - 1) / FLAGS_bs;
    CHECK_GT(train_steps, 0) << "The training set is smaller than one global batch.";

    for (int epoch = 0; epoch < FLAGS_num_epoch; ++epoch) {
        float train_loss_sum = 0.0f;
        float train_samples = 0.0f;
        const auto epoch_start = std::chrono::high_resolution_clock::now();

        auto train_iterator = train_dataloader.begin();
        for (size_t train_idx = 0; train_idx < train_steps; ++train_idx, ++train_iterator) {
            const auto [image, label] = *train_iterator;
            const auto batch_size = static_cast<float>(image->Dims()[0]);
            auto new_image = std::make_shared<Tensor>(image->To(device));
            auto new_label = std::make_shared<Tensor>(label->To(device));

            optimizer.ZeroGrad();
            const auto outputs = (*model)({new_image});
            const auto loss = loss_fn.Forward({outputs[0], new_label});
            loss[0]->Backward();

            // Reading after backward keeps CUDA forward and backward asynchronous with respect to the host.
            const Tensor loss_cpu = loss[0]->To(cpu_device);
            const float current_loss = static_cast<const float *>(loss_cpu.DataPtr())[0];
            train_loss_sum += current_loss * batch_size;
            train_samples += batch_size;
            if (rank.IsMainRank() && train_idx % kNumItersOfOutputDuration == 0) {
                LOG(INFO) << "epoch: " << epoch << ", [" << train_idx * FLAGS_bs * ddp_world_size << "/"
                          << train_dataset->Size() << "] loss: " << current_loss;
            }
            optimizer.Step();
        }

        const auto epoch_end = std::chrono::high_resolution_clock::now();
        const double duration_us = std::chrono::duration<double, std::micro>(epoch_end - epoch_start).count();
        const auto global_train_metrics = ReduceMetrics({train_loss_sum, 0.0f, train_samples}, device, ddp_pg);
        if (rank.IsMainRank()) {
            LOG(INFO) << std::format("epoch {:2d}/{} | train loss {:.6f} | lr {:.2e} | ({:.2f} ms | {:.0f} samples/s)",
                                     epoch, FLAGS_num_epoch - 1, global_train_metrics[0] / global_train_metrics[2],
                                     FLAGS_lr, duration_us / 1e3f, global_train_metrics[2] / (duration_us / 1e6));
        }

        float test_loss_sum = 0.0f;
        float correct = 0.0f;
        float test_samples = 0.0f;
        {
            autograd::NoGradGuard no_grad;
            for (const auto &[image, label] : test_dataloader) {
                auto new_image = std::make_shared<Tensor>(image->To(device));
                auto new_label = std::make_shared<Tensor>(label->To(device));
                const auto outputs = (*model)({new_image});
                const auto loss = loss_fn.Forward({outputs[0], new_label});

                const Tensor output_cpu = outputs[0]->To(cpu_device);
                const Tensor label_cpu = label->To(cpu_device);
                const Tensor loss_cpu = loss[0]->To(cpu_device);
                const int64_t batch_size = output_cpu.Dims()[0];
                const auto *labels = static_cast<const uint8_t *>(label_cpu.DataPtr());
                const auto *output_values = static_cast<const float *>(output_cpu.DataPtr());
                for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                    const auto *scores = output_values + batch_idx * kNumClasses;
                    const int prediction = std::max_element(scores, scores + kNumClasses) - scores;
                    correct += prediction == labels[batch_idx];
                }
                test_loss_sum += static_cast<const float *>(loss_cpu.DataPtr())[0] * batch_size;
                test_samples += batch_size;
            }
        }

        const auto global_test_metrics = ReduceMetrics({test_loss_sum, correct, test_samples}, device, ddp_pg);
        if (rank.IsMainRank()) {
            LOG(INFO) << "epoch: " << epoch << " | test loss: " << global_test_metrics[0] / global_test_metrics[2]
                      << " | test accuracy: " << global_test_metrics[1] / global_test_metrics[2] << " ("
                      << global_test_metrics[1] << "/" << global_test_metrics[2] << ")";
        }
    }
}
} // namespace

DEFINE_validator(device,
                 [](const char *, const std::string &value) { return value == kDeviceCPU || value == kDeviceCUDA; });
DEFINE_validator(nthread_per_process, [](const char *, int32_t value) { return value > 0; });

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    CHECK(!FLAGS_dataset.empty()) << "--dataset must point to the MNIST files.";
    CHECK_GT(FLAGS_bs, 0);
    if (FLAGS_nthread_per_process > 1) {
        CHECK_EQ(FLAGS_device, kDeviceCUDA) << "DDP requires --device=cuda.";
#if defined(USE_CUDA)
        const int visible_gpu_count = [] {
            int count = 0;
            const cudaError_t status = cudaGetDeviceCount(&count);
            CHECK_EQ(status, cudaSuccess) << "Unable to query CUDA-visible GPUs: " << cudaGetErrorString(status);
            return count;
        }();
        CHECK_LE(FLAGS_nthread_per_process, visible_gpu_count)
            << "--nthread_per_process exceeds the number of GPUs visible through CUDA_VISIBLE_DEVICES.";
#else
        LOG(FATAL) << "DDP requires a CUDA-enabled build.";
#endif
    }

    nn::parallel::global::InitAllEnv(FLAGS_nthread_per_process, /*tensor_parallel_size=*/1,
                                     /*sequence_parallel_enabled=*/false, /*pipeline_parallel_size=*/1,
                                     /*virtual_pipeline_parallel=*/1);
    LOG(INFO) << nn::parallel::global::ProcessGroupOverview();

    auto train_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, true);
    auto test_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, false);

    // Parameter initialization uses a shared RNG, so build each replica before launching training threads.
    std::vector<std::shared_ptr<MNIST>> networks;
    networks.reserve(FLAGS_nthread_per_process);
    for (int idx = 0; idx < FLAGS_nthread_per_process; ++idx) { networks.emplace_back(std::make_shared<MNIST>()); }

    if (FLAGS_nthread_per_process > 1) {
        std::vector<std::thread> threads;
        threads.reserve(FLAGS_nthread_per_process);
        for (int idx = 0; idx < FLAGS_nthread_per_process; ++idx) {
            nn::parallel::Rank rank(nn::parallel::global::GetGlobalProcRank(), idx,
                                    nn::parallel::global::GetNprocPerNode(), FLAGS_nthread_per_process);
            threads.emplace_back(Train, rank, networks[idx], train_dataset, test_dataset);
        }
        for (auto &thread : threads) { thread.join(); }
    } else {
        nn::parallel::Rank rank(nn::parallel::global::GetGlobalProcRank(), 0, nn::parallel::global::GetNprocPerNode(),
                                FLAGS_nthread_per_process);
        Train(rank, networks[0], train_dataset, test_dataset);
    }

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();
    return 0;
}
