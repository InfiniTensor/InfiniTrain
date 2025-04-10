#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/dataloader.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/optimizer.h"

#include "example/gpt2/dataset.h"

DEFINE_string(dataset, "", "tinyshakespeare dataset path");
DEFINE_int32(bs, 4, "batch size");
DEFINE_int32(num_epoch, 1, "num epochs");
DEFINE_double(lr, 0.01, "learning rate");

using namespace infini_train;

namespace {
constexpr int kNumItersOfOutputDuration = 10;
constexpr int kSequenceLength = 64;
}; // namespace

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

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    auto train_dataset = std::make_shared<TinyShakespeareDataset>(FLAGS_dataset, kSequenceLength, true);
    DataLoader train_dataloader(train_dataset, FLAGS_bs);

    // TODO(dcj): Add sampler & eval dataloader later.
    auto test_dataset = std::make_shared<TinyShakespeareDataset>(FLAGS_dataset, kSequenceLength, false);
    DataLoader test_dataloader(test_dataset, FLAGS_bs);

    // TODO(zbl): Finish train/test logic
    for (int epoch = 0; epoch < FLAGS_num_epoch; ++epoch) {
        int train_idx = 0;
        for (const auto &[input, target] : train_dataloader) {
            std::cout << train_idx << ": "
                      << ReadUint16DataToString(static_cast<const uint8_t *>((*target).DataPtr()),
                                                FLAGS_bs * kSequenceLength)
                      << std::endl;
            // if (train_idx % kNumItersOfOutputDuration == 0) {
            //     LOG(ERROR) << "epoch: " << epoch << ", [" << train_idx * FLAGS_bs << "/" << train_dataset->Size()
            //                << "] " << " loss: XXX";
            // }
            train_idx += 1;
        }
    }

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}
