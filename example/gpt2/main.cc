#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/dataloader.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/optimizer.h"

#include "example/gpt2/dataset.h"
#include "example/gpt2/net.h"

DEFINE_string(dataset, "", "tinyshakespeare dataset path");
DEFINE_int32(bs, 4, "batch size");
DEFINE_int32(num_epoch, 1, "num epochs");
DEFINE_double(lr, 0.01, "learning rate");
DEFINE_string(device, "cuda", "device type (cpu/cuda)");

using namespace infini_train;

namespace {
constexpr int kNumItersOfOutputDuration = 10;
constexpr int kSequenceLength = 64;

constexpr char kDeviceCPU[] = "cpu";
constexpr char kDeviceCUDA[] = "cuda";
}; // namespace

DEFINE_validator(device,
                 [](const char *, const std::string &value) { return value == kDeviceCPU || value == kDeviceCUDA; });

std::string ReadUint16DataToString(const uint8_t *ptr, size_t size) {
    // Output sequence in hex string
    std::ostringstream oss;

    if (!ptr || size == 0) {
        return "";
    }

    for (size_t i = 0; i < size; ++i) {
        uint16_t value = ptr[i * 2] | (ptr[i * 2 + 1] << 8);
        oss << std::uppercase << std::setw(4) << std::setfill('0') << std::hex << value;
    }

    return oss.str();
}

template <typename T> std::string ArrayToString(const T *array, size_t length) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < length; ++i) {
        oss << static_cast<float>(array[i]);
        if (i < length - 1) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    auto train_dataset = std::make_shared<TinyShakespeareDataset>(FLAGS_dataset, kSequenceLength, true);
    DataLoader train_dataloader(train_dataset, FLAGS_bs);

    // TODO(dcj): Add sampler & eval dataloader later.
    auto test_dataset = std::make_shared<TinyShakespeareDataset>(FLAGS_dataset, kSequenceLength, false);
    DataLoader test_dataloader(test_dataset, FLAGS_bs);

    auto network = GPT2();
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

    // TODO(zbl): Finish train/test logic
    for (int epoch = 0; epoch < FLAGS_num_epoch; ++epoch) {
        int train_idx = 0;
        for (const auto &[input, target] : train_dataloader) {
            auto new_input = std::make_shared<Tensor>(input->To(device));
            auto new_target = std::make_shared<Tensor>(target->To(device));
            auto outputs = network.Forward({new_input});
            auto output = outputs[0]->To(Device(DeviceType::kCPU, 0));
            std::cout << train_idx << ": "
                      << ArrayToString<float>(reinterpret_cast<float *>(output.DataPtr()), 4 * 64 * 768) << std::endl;
            // std::cout << train_idx << ": "
            //           << ArrayToString<uint16_t>(reinterpret_cast<uint16_t *>(output.DataPtr()), 4 * 64)
            //           << std::endl;
            train_idx += 1;
            break;
        }
    }

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}
