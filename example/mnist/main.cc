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
#include "infini_train/include/optimizer.h"

#include "example/mnist/dataset.h"
#include "example/mnist/net.h"

DEFINE_string(dataset, "", "mnist dataset path");
DEFINE_string(model, "mlp", "model type (mlp/cnn)");
DEFINE_int32(bs, 64, "batch size");
DEFINE_int32(num_epoch, 1, "num epochs");
DEFINE_double(lr, 0.01, "learning rate");
DEFINE_string(device, "cpu", "device type (cpu/cuda)");
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
} // namespace

DEFINE_validator(device,
                 [](const char *, const std::string &value) { return value == kDeviceCPU || value == kDeviceCUDA; });

DEFINE_validator(model,
                 [](const char *, const std::string &value) { return value == kModelMLP || value == kModelCNN; });

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

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    auto train_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, true);
    DataLoader train_dataloader(train_dataset, FLAGS_bs);

    // TODO(dcj): Add sampler & eval dataloader later.
    auto test_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, false);
    DataLoader test_dataloader(test_dataset, FLAGS_bs);

    auto network = CreateMNISTNetwork(FLAGS_model);
    Device device = FLAGS_device == kDeviceCPU ? Device() : Device(Device::DeviceType::kCUDA, 0);
    Device cpu_device = Device();
    network->To(device);

    if (!FLAGS_init_weights.empty()) {
        TrainerState state;
        Checkpoint::Load(FLAGS_init_weights, *network, /*optimizer=*/nullptr, state, /*lr_scheduler=*/nullptr);
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
            auto new_image = std::make_shared<Tensor>(image->To(device));
            auto new_label = std::make_shared<Tensor>(label->To(device));

            auto outputs = network->Forward({new_image});
            optimizer.ZeroGrad();

            auto loss = loss_fn.Forward({outputs[0], new_label});
            loss[0]->Backward();

            // Defer the loss D2H copy until after backward; reading it earlier would synchronize CUDA
            // between forward and backward.
            auto loss_cpu = loss[0]->To(cpu_device);
            float current_loss = static_cast<float *>(loss_cpu.DataPtr())[0];
            total_loss += current_loss;
            if (train_idx % kNumItersOfOutputDuration == 0) {
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
        LOG(ERROR) << std::format("epoch {:2d}/{} | train loss {:.6f} | lr {:.2e} | ({:.2f} ms | {:.0f} samples/s)",
                                  epoch, FLAGS_num_epoch - 1, train_loss, FLAGS_lr, duration_us / 1e3f,
                                  train_dataset->Size() / (duration_us / 1e6));

        const auto [test_loss, test_accuracy] = Evaluate(*network, loss_fn, test_dataloader, device);
        LOG(ERROR) << std::format("epoch {:2d} | test loss {:.6f} | test accuracy {:.4f}", epoch, test_loss,
                                  test_accuracy);
        AppendMetrics(FLAGS_metrics_file,
                      std::format("{{\"type\": \"epoch_end\", \"epoch\": {}, \"train_loss\": {:.6f}, "
                                  "\"test_loss\": {:.6f}, \"test_accuracy\": {:.6f}}}",
                                  epoch, train_loss, test_loss, test_accuracy));
    }

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}
