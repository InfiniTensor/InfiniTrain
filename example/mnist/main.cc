#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/loss.h"
#include "infini_train/include/optimizer.h"

#include "example/mnist/dataset.h"
#include "example/mnist/net.h"

DEFINE_string(dataset, "", "mnist dataset path");
DEFINE_int32(bs, 64, "batch size");
DEFINE_int32(num_epoch, 1, "num epochs");
DEFINE_double(lr, 0.01, "learning rate");
DEFINE_string(device, "cpu", "device type (cpu/cuda)");

using namespace infini_train;

namespace {
constexpr int kNumItersOfOutputDuration = 10;
constexpr int kNumClasses = 10;

constexpr char kDeviceCPU[] = "cpu";
constexpr char kDeviceCUDA[] = "cuda";
}; // namespace

DEFINE_validator(device,
                 [](const char *, const std::string &value) { return value == kDeviceCPU || value == kDeviceCUDA; });

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    auto train_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, true);
    DataLoader train_dataloader(train_dataset, FLAGS_bs);

    // TODO(dcj): Add sampler & eval dataloader later.
    auto test_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, false);
    DataLoader test_dataloader(test_dataset, FLAGS_bs);

    auto network = MNIST();
    Device device;
    if (FLAGS_device == kDeviceCPU) {
        device = Device(DeviceType::kCPU, 0);
    } else {
        device = Device(DeviceType::kCUDA, 0);
    }
    network.To(device);

    auto loss_fn = nn::CrossEntropyLoss();
    loss_fn.To(device);
    auto optimizer = optimizers::SGD(network.Parameters(), FLAGS_lr);

    for (int epoch = 0; epoch < FLAGS_num_epoch; ++epoch) {
        int train_idx = 0;
        for (const auto &[image, label] : train_dataloader) {
            auto new_image = std::make_shared<Tensor>(image->To(device));
            auto new_label = std::make_shared<Tensor>(label->To(device));

            auto outputs = network.Forward({new_image});
            optimizer.ZeroGrad();

            auto loss = loss_fn.Forward({outputs[0], new_label});
            auto loss_cpu = loss[0]->To(Device());
            if (train_idx % kNumItersOfOutputDuration == 0) {
                LOG(ERROR) << "epoch: " << epoch << ", [" << train_idx * FLAGS_bs << "/" << train_dataset->Size()
                           << "] "
                           << " loss: " << reinterpret_cast<float *>(loss_cpu.DataPtr())[0];
            }

            loss[0]->Backward();
            optimizer.Step();
            train_idx += 1;
        }
    }

    // TODO(dcj): Add no_grad() context manager later.
    std::vector<float> test_losses;
    int correct = 0;
    int total = 0;
    for (const auto &[image, label] : test_dataloader) {
        auto new_image = std::make_shared<Tensor>(image->To(device));
        auto new_label = std::make_shared<Tensor>(label->To(device));

        auto label_cpu = label->To(Device());
        auto outputs = network.Forward({new_image});
        auto output_cpu = outputs[0]->To(Device());
        auto loss = loss_fn.Forward({outputs[0], new_label});
        auto loss_cpu = loss[0]->To(Device());

        const int batch_size = output_cpu.Dims()[0];
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            auto label_index = reinterpret_cast<uint8_t *>(label_cpu.DataPtr())[batch_idx];
            const auto *output_values = reinterpret_cast<float *>(output_cpu.DataPtr()) + batch_idx * kNumClasses;
            const int output_index = std::max_element(output_values, output_values + kNumClasses) - output_values;
            if (output_index == label_index) {
                ++correct;
            }
        }
        total += batch_size;
        test_losses.push_back(reinterpret_cast<float *>(loss_cpu.DataPtr())[0]);
    }
    const auto avg_loss = std::accumulate(test_losses.begin(), test_losses.end(), 0.0) / test_losses.size();
    LOG(ERROR) << "Total: " << total << ", Correct: " << correct
               << ", Accuracy: " << static_cast<float>(correct) / total << ", AverageLoss: " << avg_loss;

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}
