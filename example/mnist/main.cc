#include <chrono>
#include <cstdlib>
#include <format>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel_config.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

#include "example/mnist/cnn_net.h"
#include "example/mnist/dataset.h"
#include "example/mnist/net.h"

DEFINE_string(dataset, "", "mnist dataset path");
DEFINE_string(model, "cnn", "model type (mlp/cnn)");
DEFINE_int32(bs, 64, "batch size");
// Defaults are tuned for the CNN demo (default --model=cnn) to reach ~97.8% test accuracy.
// The MLP reaches ~92% at these defaults; pass more epochs (e.g. --num_epoch=20) to reach ~95%.
DEFINE_int32(num_epoch, 3, "num epochs");
DEFINE_double(lr, 0.1, "learning rate");
DEFINE_string(device, "cpu", "device type (cpu/cuda)");

using namespace infini_train;

namespace {
constexpr int kNumItersOfOutputDuration = 10;
constexpr int kNumClasses = 10;

constexpr char kDeviceCPU[] = "cpu";
constexpr char kDeviceCUDA[] = "cuda";
constexpr char kModelMLP[] = "mlp";
constexpr char kModelCNN[] = "cnn";
}; // namespace

DEFINE_validator(device,
                 [](const char *, const std::string &value) { return value == kDeviceCPU || value == kDeviceCUDA; });
DEFINE_validator(model,
                 [](const char *, const std::string &value) { return value == kModelMLP || value == kModelCNN; });

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    CHECK_GT(FLAGS_bs, 0) << "--bs must be a positive batch size (got " << FLAGS_bs << ")";

    // Consume the WORLD_SIZE/RANK/LOCAL_RANK env injected by infini_run; without the launcher
    // every size defaults to 1 and the single-process path below is unchanged.
    nn::parallel::global::InitAllEnv(/*nthread_per_process=*/1, /*tensor_parallel_size=*/1,
                                     /*sequence_parallel_enabled=*/false, /*pipeline_parallel_size=*/1,
                                     /*virtual_pipeline_parallel_size=*/1);
    nn::parallel::Rank rank(nn::parallel::global::GetGlobalProcRank(), /*thread_rank=*/0,
                            nn::parallel::global::GetNprocPerNode(), nn::parallel::global::GetNthreadPerProc());
    const int ddp_world_size = nn::parallel::global::GetDataParallelSize();

    auto train_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, true);

    Device device;
    const nn::parallel::ProcessGroup *ddp_pg = nullptr;
    int ddp_rank = 0;
    if (rank.IsParallel()) {
        CHECK_EQ(FLAGS_device, kDeviceCUDA) << "Distributed training requires --device=cuda";
        // One GPU per process, taken from the LOCAL_RANK assigned by the launcher.
        device = Device(Device::DeviceType::kCUDA, nn::parallel::global::GetDeviceIndex(rank.thread_rank()));
        auto *pg_factory = nn::parallel::ProcessGroupFactory::Instance(device.type());
        ddp_pg = pg_factory->GetOrCreate(nn::parallel::GetDataParallelProcessGroupName(rank.GlobalRank()),
                                         nn::parallel::GetDataParallelGroupRanks(rank.GlobalRank()));
        ddp_rank = ddp_pg->GetGroupRank(rank.GlobalRank());
    } else {
        device = FLAGS_device == kDeviceCPU ? Device() : Device(Device::DeviceType::kCUDA, 0);
    }

    // Sharded batches per rank under DDP; the legacy loader keeps the single-process path.
    std::optional<DistributedDataLoader> ddp_train_dataloader;
    std::optional<DataLoader> plain_train_dataloader;
    if (ddp_pg != nullptr) {
        // The loss AllReduce runs step-for-step on every rank, so all ranks must see the same
        // batch count or a collective would hang out of step. DistributedDataLoader derives
        // that count from the global batch size, which keeps every rank aligned.
        ddp_train_dataloader.emplace(train_dataset, FLAGS_bs, ddp_rank, ddp_world_size);
    } else {
        plain_train_dataloader.emplace(train_dataset, FLAGS_bs);
    }
    const DataLoader &train_loader = ddp_pg != nullptr ? *ddp_train_dataloader : *plain_train_dataloader;

    // TODO(dcj): Add sampler & eval dataloader later.
    // The test loader stays unsharded so every rank reports metrics over the full test set.
    auto test_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, false);
    DataLoader test_dataloader(test_dataset, FLAGS_bs);

    std::shared_ptr<nn::Module> network;
    if (FLAGS_model == kModelCNN) {
        network = std::make_shared<MnistCnn>();
    } else {
        network = std::make_shared<MNIST>();
    }
    Device cpu_device = Device();
    network->To(device);

    auto loss_fn = std::make_shared<nn::CrossEntropyLoss>();
    loss_fn->To(device);

    // Wrap with DDP only after all device conversions: a later .To() recreates parameter
    // tensors and would leave the gradient hooks registered at wrap time dangling.
    if (ddp_pg != nullptr) {
        network = std::make_shared<nn::parallel::DistributedDataParallel>(
            network, rank, nn::parallel::DistributedDataParallelConfig{});
        // Keep every replica starting from the same state; parameter broadcast before training
        // is part of the DDP contract and left to the caller by the wrapper. Parameters only:
        // the demo network carries no buffers, which PyTorch would sync alongside them.
        ddp_pg->Broadcast(network->Parameters(), /*root_rank_in_group=*/0);
    }

    auto optimizer = optimizers::SGD(network->Parameters(), FLAGS_lr);

    for (int epoch = 0; epoch < FLAGS_num_epoch; ++epoch) {
        int train_idx = 0;
        float total_loss = 0.0;

        const auto epoch_start = std::chrono::high_resolution_clock::now();

        for (const auto &[image, label] : train_loader) {
            auto new_image = std::make_shared<Tensor>(image->To(device));
            auto new_label = std::make_shared<Tensor>(label->To(device));

            // Zero grads before forward: DDP rebinds param.grad to its bucket view during forward.
            optimizer.ZeroGrad();

            auto outputs = (*network)({new_image});

            auto loss = (*loss_fn)({outputs[0], new_label});
            loss[0]->Backward();

            // Defer the loss D2H copy until after backward; reading it earlier would synchronize CUDA
            // between forward and backward.
            auto loss_cpu = loss[0]->To(cpu_device);
            float current_loss = static_cast<float *>(loss_cpu.DataPtr())[0];
            if (ddp_pg != nullptr) {
                // Average the per-rank loss so the logged value matches the global batch. With an
                // uneven trailing batch the equal-weight average is a bounded per-step approximation
                // until a sampler lands.
                auto loss_stat
                    = std::make_shared<Tensor>(&current_loss, std::vector<int64_t>{}, DataType::kFLOAT32, device);
                nn::parallel::function::AllReduce(loss_stat, nn::parallel::function::ReduceOpType::kAvg, ddp_pg);
                auto loss_stat_cpu = loss_stat->To(cpu_device);
                current_loss = static_cast<const float *>(loss_stat_cpu.DataPtr())[0];
            }
            total_loss += current_loss;
            if (train_idx % kNumItersOfOutputDuration == 0) {
                LOG(ERROR) << "epoch: " << epoch << ", [" << train_idx * FLAGS_bs * ddp_world_size << "/"
                           << train_dataset->Size() << "] "
                           << " loss: " << current_loss;
            }

            optimizer.Step();
            train_idx += 1;
        }

        const auto epoch_end = std::chrono::high_resolution_clock::now();
        const double duration_us = std::chrono::duration<double, std::micro>(epoch_end - epoch_start).count();

        LOG(ERROR) << std::format("epoch {:2d}/{} | train loss {:.6f} | lr {:.2e} | ({:.2f} ms | {:.0f} samples/s)",
                                  epoch, FLAGS_num_epoch - 1, total_loss / train_idx, FLAGS_lr, duration_us / 1e3f,
                                  train_dataset->Size() / (duration_us / 1e6));
    }

    // TODO(dcj): Add no_grad() context manager later.
    std::vector<float> test_losses;
    int correct = 0;
    int total = 0;
    for (const auto &[image, label] : test_dataloader) {
        auto new_image = std::make_shared<Tensor>(image->To(device));
        auto new_label = std::make_shared<Tensor>(label->To(device));

        auto label_cpu = label->To(cpu_device);
        auto outputs = (*network)({new_image});
        auto output_cpu = outputs[0]->To(cpu_device);
        auto loss = (*loss_fn)({outputs[0], new_label});
        auto loss_cpu = loss[0]->To(cpu_device);

        const int batch_size = output_cpu.Dims()[0];
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            auto label_index = reinterpret_cast<uint8_t *>(label_cpu.DataPtr())[batch_idx];
            const auto *output_values = static_cast<float *>(output_cpu.DataPtr()) + batch_idx * kNumClasses;
            const int output_index = std::max_element(output_values, output_values + kNumClasses) - output_values;
            if (output_index == label_index) {
                ++correct;
            }
        }
        total += batch_size;
        test_losses.push_back(static_cast<float *>(loss_cpu.DataPtr())[0]);
    }
    const auto avg_loss = std::accumulate(test_losses.begin(), test_losses.end(), 0.0) / test_losses.size();
    LOG(ERROR) << "Total: " << total << ", Correct: " << correct
               << ", Accuracy: " << static_cast<float>(correct) / total << ", AverageLoss: " << avg_loss;

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}
