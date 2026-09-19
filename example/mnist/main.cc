#include <chrono>
#include <cstdlib>
#include <format>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/autograd/grad_mode.h"

#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/optimizer.h"

#include "example/mnist/dataset.h"
#include "example/mnist/net.h"

DEFINE_string(dataset, "", "mnist dataset path");
DEFINE_int32(bs, 64, "batch size");
DEFINE_int32(num_epoch, 1, "num epochs");
DEFINE_double(lr, 0.01, "learning rate");
DEFINE_string(device, "cpu", "device type (cpu/cuda)");
DEFINE_string(model, "mlp", "model type (mlp/cnn)");

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

    // Distributed init from env (WORLD_SIZE/RANK/LOCAL_RANK set by infini_run).
    // Same helper pattern as example/gpt2: pure data-parallel layout (TP=PP=1).
    nn::parallel::global::InitAllEnv(/*nthread_per_process=*/1, /*tensor_parallel_size=*/1,
                                     /*sequence_parallel_enabled=*/false, /*pipeline_parallel_size=*/1,
                                     /*virtual_pipeline_parallel=*/1);
    const int ddp_world_size = nn::parallel::global::GetDataParallelSize();
    const bool distributed = ddp_world_size > 1;
    nn::parallel::Rank rank(nn::parallel::global::GetGlobalProcRank(), 0,
                            nn::parallel::global::GetNprocPerNode(), /*threads_per_process=*/1);
    nn::parallel::global::thread_global_rank = rank.GlobalRank();
    const int ddp_rank = distributed ? rank.GlobalRank() : 0;
    const bool is_main_rank = rank.IsMainRank();

    const nn::parallel::ProcessGroup *ddp_pg = nullptr;
    if (distributed) {
        const auto device_type = Device::DeviceType::kCUDA;
        auto *pg_factory = nn::parallel::ProcessGroupFactory::Instance(device_type);
        ddp_pg = pg_factory->GetOrCreate(nn::parallel::GetDataParallelProcessGroupName(rank.GlobalRank()),
                                         nn::parallel::GetDataParallelGroupRanks(rank.GlobalRank()));
    }

    auto train_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, true);
    std::unique_ptr<DataLoader> train_dataloader
        = distributed ? std::unique_ptr<DataLoader>(
                            std::make_unique<DistributedDataLoader>(train_dataset, FLAGS_bs, ddp_rank, ddp_world_size))
                      : std::unique_ptr<DataLoader>(std::make_unique<DataLoader>(train_dataset, FLAGS_bs));

    // TODO(dcj): Add sampler & eval dataloader later.
    auto test_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, false);
    std::unique_ptr<DataLoader> test_dataloader
        = distributed ? std::unique_ptr<DataLoader>(
                            std::make_unique<DistributedDataLoader>(test_dataset, FLAGS_bs, ddp_rank, ddp_world_size))
                      : std::unique_ptr<DataLoader>(std::make_unique<DataLoader>(test_dataset, FLAGS_bs));

    std::shared_ptr<nn::Module> network;
    if (FLAGS_model == kModelCNN) {
        network = std::make_shared<MnistCnn>();
    } else {
        network = std::make_shared<MNIST>();
    }
    Device device = FLAGS_device == kDeviceCPU ? Device() : Device(Device::DeviceType::kCUDA, 0);
    if (distributed) {
        device = Device(Device::DeviceType::kCUDA, nn::parallel::global::GetLocalProcRank());
    }
    Device cpu_device = Device();
    network->To(device);

    if (distributed) {
        // Sync initial params from root so all ranks start from identical weights,
        // then wrap with DDP (grad-only sync during training, no per-step loss sync).
        // NOTE: complete all .To(device) conversions before wrapping (same rule as gpt2).
        ddp_pg->Broadcast(network->Parameters(), /*root_rank_in_group=*/0);
        network = std::make_shared<nn::parallel::DistributedDataParallel>(network, rank,
                                                                          nn::parallel::DistributedDataParallelConfig{});
    }

    auto loss_fn = std::make_shared<nn::CrossEntropyLoss>();
    loss_fn->To(device);
    auto optimizer = optimizers::SGD(network->Parameters(), FLAGS_lr);

    for (int epoch = 0; epoch < FLAGS_num_epoch; ++epoch) {
        int train_idx = 0;
        float total_loss = 0.0;

        const auto epoch_start = std::chrono::high_resolution_clock::now();

        for (const auto &[image, label] : *train_dataloader) {
            auto new_image = std::make_shared<Tensor>(image->To(device));
            auto new_label = std::make_shared<Tensor>(label->To(device));

            // NOTE: ZeroGrad must run BEFORE DDP Forward. Reducer::PrepareForBackward()
            // (inside DDP Forward, gradient_as_bucket_view=true) binds param.grad to the
            // bucket view; ZeroGrad(set_to_none=true) after Forward would reset that
            // binding, so backward would accumulate into a standalone grad that the
            // reducer never all-reduces (silent no-sync, cross-rank weight fork).
            optimizer.ZeroGrad();
            auto outputs = (*network)({new_image});

            auto loss = (*loss_fn)({outputs[0], new_label});
            loss[0]->Backward();

            // Defer the loss D2H copy until after backward; reading it earlier would synchronize CUDA
            // between forward and backward.
            auto loss_cpu = loss[0]->To(cpu_device);
            float current_loss = static_cast<float *>(loss_cpu.DataPtr())[0];
            total_loss += current_loss;
            // Distributed: gradients sync via DDP; loss is rank-local only (no per-step AllReduce).
            if (is_main_rank && train_idx % kNumItersOfOutputDuration == 0) {
                LOG(ERROR) << "epoch: " << epoch << ", [" << train_idx * FLAGS_bs << "/" << train_dataset->Size()
                           << "] "
                           << " loss: " << current_loss;
            }

            optimizer.Step();
            train_idx += 1;
        }

        const auto epoch_end = std::chrono::high_resolution_clock::now();
        const double duration_us = std::chrono::duration<double, std::micro>(epoch_end - epoch_start).count();

        if (is_main_rank) {
            LOG(ERROR) << std::format(
                "epoch {:2d}/{} | train loss {:.6f} | lr {:.2e} | ({:.2f} ms | {:.0f} samples/s)", epoch,
                FLAGS_num_epoch - 1, total_loss / train_idx, FLAGS_lr, duration_us / 1e3f,
                train_dataset->Size() / (duration_us / 1e6));
        }
    }

    // Evaluation builds forward-only graphs; keep it under NoGradGuard so it never
    // primes grad accumulators with a dependency count the next backward cannot satisfy
    // (which would silently stop gradient accumulation). Resolves TODO(dcj) no_grad().
    autograd::NoGradGuard no_grad;
    std::vector<float> test_losses;
    int correct = 0;
    int total = 0;
    // Weighted loss accumulator for the distributed path (sharded eval + epoch-level AllReduce).
    double local_loss_sum = 0.0;
    for (const auto &[image, label] : *test_dataloader) {
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
        const float batch_loss = static_cast<float *>(loss_cpu.DataPtr())[0];
        test_losses.push_back(batch_loss);
        local_loss_sum += static_cast<double>(batch_loss) * batch_size;
    }
    if (distributed) {
        // Each rank evaluated its own shard; reduce (loss_sum, correct, samples) once per epoch.
        const float stats[3]
            = {static_cast<float>(local_loss_sum), static_cast<float>(correct), static_cast<float>(total)};
        auto stats_tensor
            = std::make_shared<Tensor>(stats, std::vector<int64_t>{3}, DataType::kFLOAT32, device);
        ddp_pg->AllReduce(stats_tensor, nn::parallel::function::ReduceOpType::kSum);
        auto stats_cpu = stats_tensor->To(cpu_device);
        const auto *reduced = static_cast<const float *>(stats_cpu.DataPtr());
        if (is_main_rank) {
            LOG(ERROR) << "Total: " << static_cast<int>(reduced[2]) << ", Correct: " << static_cast<int>(reduced[1])
                       << ", Accuracy: " << reduced[1] / reduced[2] << ", AverageLoss: " << reduced[0] / reduced[2];
        }
    } else {
        const auto avg_loss = std::accumulate(test_losses.begin(), test_losses.end(), 0.0) / test_losses.size();
        LOG(ERROR) << "Total: " << total << ", Correct: " << correct
                   << ", Accuracy: " << static_cast<float>(correct) / total << ", AverageLoss: " << avg_loss;
    }

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}
