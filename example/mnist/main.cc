#include <chrono>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/checkpoint/checkpoint.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel_config.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/parallel_functional.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/reduce_op_type.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/optimizer.h"

#include "example/mnist/dataset.h"
#include "example/mnist/net.h"

DEFINE_string(dataset, "", "mnist dataset path");
DEFINE_string(model, "mlp", "model type (mlp/cnn)");
DEFINE_int32(bs, 64, "batch size per rank");
DEFINE_int32(num_epoch, 1, "num epochs");
DEFINE_double(lr, 0.01, "learning rate");
DEFINE_string(device, "cpu", "device type (cpu/cuda) for single-process runs");
DEFINE_string(init_weights, "", "checkpoint dir to load initial weights from (e.g. exported by PyTorch)");
DEFINE_string(metrics_file, "", "append train/test metrics as JSON lines to this file for visualization");

using namespace infini_train;

namespace {
constexpr int kNumItersOfOutputDuration = 10;
constexpr int kNumClasses = 10;

constexpr char kDeviceCPU[] = "cpu";
constexpr char kDeviceCUDA[] = "cuda";
constexpr char kModelMLP[] = "mlp";
constexpr char kModelCNN[] = "cnn";

// Appends one JSON line to the metrics file; a no-op when the flag is empty.
void AppendMetrics(const std::string &metrics_file, const std::string &json_line) {
    if (metrics_file.empty()) {
        return;
    }
    std::ofstream ofs(metrics_file, std::ios::app);
    ofs << json_line << "\n";
}

// Runs the test set under no_grad; returns {average test loss, accuracy}.
std::pair<float, float> Evaluate(nn::Module &network, nn::CrossEntropyLoss &loss_fn, DataLoader &test_dataloader,
                                 const Device &device) {
    Device cpu_device = Device();
    std::vector<float> test_losses;
    int correct = 0;
    int total = 0;
    autograd::NoGradGuard no_grad;
    for (const auto &[image, label] : test_dataloader) {
        auto new_image = std::make_shared<Tensor>(image->To(device));
        auto new_label = std::make_shared<Tensor>(label->To(device));

        auto label_cpu = label->To(cpu_device);
        auto outputs = network.Forward({new_image});
        auto output_cpu = outputs[0]->To(cpu_device);
        auto loss = loss_fn.Forward({outputs[0], new_label});
        auto loss_cpu = loss[0]->To(cpu_device);

        const int batch_size = output_cpu.Dims()[0];
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            auto label_index = reinterpret_cast<const uint8_t *>(label_cpu.DataPtr())[batch_idx];
            const auto *output_values = static_cast<const float *>(output_cpu.DataPtr()) + batch_idx * kNumClasses;
            const int output_index = std::max_element(output_values, output_values + kNumClasses) - output_values;
            if (output_index == label_index) {
                ++correct;
            }
        }
        total += batch_size;
        test_losses.push_back(static_cast<const float *>(loss_cpu.DataPtr())[0]);
    }
    const auto avg_loss = static_cast<float>(std::accumulate(test_losses.begin(), test_losses.end(), 0.0)
                                             / std::max<std::size_t>(test_losses.size(), 1));
    return {avg_loss, static_cast<float>(correct) / total};
}

int Train(const nn::parallel::Rank &rank) {
    const bool is_main_rank = rank.GlobalRank() == 0;
    const int ddp_world_size = nn::parallel::global::GetDataParallelSize();
    Device cpu_device = Device();

    auto train_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, true);
    // Strided batch sharding across DDP ranks (a no-op when ddp_world_size == 1).
    DistributedDataLoader train_dataloader(train_dataset, FLAGS_bs, rank.GlobalRank(), ddp_world_size);
    // The strided sharding can hand one rank one more batch than the other when the batch count
    // is not divisible by the world size; cap every rank at the same step count (dropping the
    // incomplete trailing global batch), otherwise the extra collective deadlocks the peer.
    const size_t train_samples_per_rank = train_dataset->Size() / ddp_world_size;
    const size_t steps_per_epoch = train_samples_per_rank / FLAGS_bs;

    // TODO(dcj): Add sampler & eval dataloader later. Each rank evaluates the full test set.
    auto test_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, false);
    DataLoader test_dataloader(test_dataset, FLAGS_bs);

    auto network = CreateMNISTNetwork(FLAGS_model);
    Device device = FLAGS_device == kDeviceCPU ? Device() : Device(Device::DeviceType::kCUDA, 0);
    const nn::parallel::ProcessGroup *ddp_pg = nullptr;
    if (rank.IsParallel()) {
        CHECK(device.IsCUDA()) << "DDP training requires --device cuda";
        device = Device(Device::DeviceType::kCUDA, nn::parallel::global::GetDeviceIndex(rank.thread_rank()));
    }
    network->To(device);

    if (!FLAGS_init_weights.empty()) {
        // Every rank loads the same file, so the initial weights are identical across ranks.
        TrainerState state;
        Checkpoint::Load(FLAGS_init_weights, *network, /*optimizer=*/nullptr, state, /*lr_scheduler=*/nullptr);
    }

    // The DP process group must exist before constructing DistributedDataParallel, and all
    // parameters must already live on the rank's device (see example/gpt2/main.cc).
    if (ddp_world_size > 1) {
        auto *pg_factory = nn::parallel::ProcessGroupFactory::Instance(device.type());
        ddp_pg = pg_factory->GetOrCreate(nn::parallel::GetDataParallelProcessGroupName(rank.GlobalRank()),
                                         nn::parallel::GetDataParallelGroupRanks(rank.GlobalRank()));
        // NOTE: Complete all device conversions before wrapping, otherwise gradient hooks are lost.
        auto ddp_config = nn::parallel::DistributedDataParallelConfig{.zero_stage = 0};
        network = std::make_shared<nn::parallel::DistributedDataParallel>(network, rank, ddp_config);
        // Match torch DDP semantics: broadcast the parameters of the global-rank-0 process so all
        // ranks start from identical weights even if the initialization were rank-dependent.
        ddp_pg->Broadcast(network->Parameters(), /*root_rank_in_group=*/0);
    }

    auto loss_fn = nn::CrossEntropyLoss();
    loss_fn.To(device);
    auto optimizer = optimizers::SGD(network->Parameters(), FLAGS_lr);

    int64_t global_step = 0;
    for (int epoch = 0; epoch < FLAGS_num_epoch; ++epoch) {
        int train_idx = 0;
        float total_loss = 0.0;

        const auto epoch_start = std::chrono::high_resolution_clock::now();

        for (const auto &[image, label] : train_dataloader) {
            if (static_cast<size_t>(train_idx) >= steps_per_epoch) {
                break;
            }
            auto new_image = std::make_shared<Tensor>(image->To(device));
            auto new_label = std::make_shared<Tensor>(label->To(device));

            // Zero before the forward pass: with DDP's gradient-as-bucket-view the forward's
            // PrepareForBackward binds parameter gradients to the bucket slices, and zeroing
            // afterwards would break that binding (matching example/gpt2/main.cc).
            optimizer.ZeroGrad();

            auto outputs = network->Forward({new_image});

            auto loss = loss_fn.Forward({outputs[0], new_label});
            loss[0]->Backward();

            // Defer the loss D2H copy until after backward; reading it earlier would synchronize CUDA
            // between forward and backward. With DDP the gradient all-reduce runs inside Backward().
            if (ddp_pg != nullptr) {
                nn::parallel::function::AllReduce(loss[0], nn::parallel::function::ReduceOpType::kAvg, ddp_pg);
            }
            auto loss_cpu = loss[0]->To(cpu_device);
            float current_loss = static_cast<float *>(loss_cpu.DataPtr())[0];
            total_loss += current_loss;
            if (is_main_rank && train_idx % kNumItersOfOutputDuration == 0) {
                LOG(ERROR) << "epoch: " << epoch << ", step: " << train_idx << ", [" << train_idx * FLAGS_bs << "/"
                           << train_dataset->Size() << "] "
                           << " loss: " << current_loss;
                AppendMetrics(FLAGS_metrics_file,
                              std::format("{{\"type\": \"train_step\", \"epoch\": {}, \"step\": {}, \"loss\": {:.6f}}}",
                                          epoch, global_step, current_loss));
            }

            optimizer.Step();
            train_idx += 1;
            global_step += 1;
        }

        const auto epoch_end = std::chrono::high_resolution_clock::now();
        const double duration_us = std::chrono::duration<double, std::micro>(epoch_end - epoch_start).count();

        const float train_loss = total_loss / train_idx;
        const auto [test_loss, test_accuracy] = Evaluate(*network, loss_fn, test_dataloader, device);
        if (is_main_rank) {
            LOG(ERROR) << std::format("epoch {:2d}/{} | train loss {:.6f} | lr {:.2e} | ({:.2f} ms | {:.0f} samples/s)",
                                      epoch, FLAGS_num_epoch - 1, train_loss, FLAGS_lr, duration_us / 1e3f,
                                      train_dataset->Size() * ddp_world_size / (duration_us / 1e6));
            LOG(ERROR) << std::format("epoch {:2d} | test loss {:.6f} | test accuracy {:.4f}", epoch, test_loss,
                                      test_accuracy);
            AppendMetrics(FLAGS_metrics_file,
                          std::format("{{\"type\": \"epoch_end\", \"epoch\": {}, \"train_loss\": {:.6f}, "
                                      "\"test_loss\": {:.6f}, \"test_accuracy\": {:.6f}}}",
                                      epoch, train_loss, test_loss, test_accuracy));
        }
    }

    return 0;
}
} // namespace

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    // Rank/world size come from the RANK / LOCAL_RANK / WORLD_SIZE / LOCAL_WORLD_SIZE environment
    // variables, which tools/infini_run sets for each spawned process. Without them this is a
    // plain single-process run.
    nn::parallel::global::InitAllEnv(/*nthread_per_process=*/1, /*tensor_parallel_size=*/1,
                                     /*sequence_parallel_enabled=*/false, /*pipeline_parallel_size=*/1,
                                     /*virtual_pipeline_parallel_size=*/1);
    nn::parallel::Rank rank(nn::parallel::global::GetGlobalProcRank(), /*thread_rank=*/0,
                            nn::parallel::global::GetNprocPerNode(), /*threads_per_process=*/1);

    Train(rank);

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}
