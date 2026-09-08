// Binary fixture bridge for independent CNN and DDP numerical validation.
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>
#include <vector>

#include "example/mnist/net.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/parallel/ddp/distributed_data_parallel.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/optimizer.h"

DEFINE_string(fixture_dir, "", "Directory of float32 parameter/input and uint8 label fixtures");
DEFINE_string(output_dir, "", "Directory for per-step binary snapshots");
DEFINE_string(device, "cpu", "cpu or cuda");
DEFINE_int32(world_size, 1, "Number of CUDA DDP replicas");
DEFINE_int32(batch_size, 8, "Global fixture batch size");
DEFINE_int32(steps, 3, "Consecutive SGD updates");
DEFINE_double(lr, 0.1, "SGD learning rate");

using namespace infini_train;
namespace fs = std::filesystem;

void Read(const fs::path &path, Tensor &tensor) {
    CHECK_EQ(fs::file_size(path), tensor.SizeInBytes()) << path;
    std::ifstream file(path, std::ios::binary);
    file.read(static_cast<char *>(tensor.DataPtr()), tensor.SizeInBytes());
    CHECK(file.good()) << path;
}
void Write(const fs::path &path, const std::shared_ptr<Tensor> &tensor) {
    CHECK(tensor != nullptr) << path;
    auto host = tensor->To(Device());
    std::ofstream file(path, std::ios::binary);
    file.write(static_cast<const char *>(host.DataPtr()), host.SizeInBytes());
    CHECK(file.good()) << path;
}
void Run(int index, std::shared_ptr<MNIST> network) {
    using namespace nn::parallel;
    global::thread_global_rank = index;
    const Device device = FLAGS_device == "cpu" ? Device() : Device(Device::DeviceType::kCUDA, index);
    network->To(device);
    std::shared_ptr<nn::Module> model = network;
    if (FLAGS_world_size > 1) {
        ProcessGroupFactory::Instance(device.type())
            ->GetOrCreate(GetDataParallelProcessGroupName(index), GetDataParallelGroupRanks(index));
        model = std::make_shared<DistributedDataParallel>(network, Rank(0, index, 1, FLAGS_world_size),
                                                          DistributedDataParallelConfig{});
    }
    nn::CrossEntropyLoss criterion;
    optimizers::SGD optimizer(model->Parameters(), FLAGS_lr);
    const fs::path output = fs::path(FLAGS_output_dir) / ("rank" + std::to_string(index));
    fs::create_directories(output);
    const int local_batch = FLAGS_batch_size / FLAGS_world_size;
    for (int step = 0; step < FLAGS_steps; ++step) {
        const std::string prefix = "step" + std::to_string(step) + ".";
        Tensor all_images({FLAGS_batch_size, 1, 28, 28}, DataType::kFLOAT32);
        Tensor all_labels({FLAGS_batch_size}, DataType::kUINT8);
        Read(fs::path(FLAGS_fixture_dir) / (prefix + "input.bin"), all_images);
        Read(fs::path(FLAGS_fixture_dir) / (prefix + "labels.bin"), all_labels);
        Tensor images(all_images, index * local_batch * 28 * 28 * sizeof(float), {local_batch, 1, 28, 28});
        Tensor labels(all_labels, index * local_batch * sizeof(uint8_t), {local_batch});
        optimizer.ZeroGrad();
        auto logits = (*model)({std::make_shared<Tensor>(images.To(device))})[0];
        auto loss = criterion.Forward({logits, std::make_shared<Tensor>(labels.To(device))})[0];
        loss->Backward();
        Write(output / (prefix + "logits.bin"), logits);
        Write(output / (prefix + "loss.bin"), loss);
        for (const auto &[name, parameter] : network->NamedParameters()) {
            Write(output / (prefix + name + ".grad.bin"), parameter->grad());
        }
        optimizer.Step();
        for (const auto &[name, parameter] : network->NamedParameters()) {
            Write(output / (prefix + name + ".parameter.bin"), parameter);
        }
    }
}
int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    CHECK(FLAGS_device == "cpu" || FLAGS_device == "cuda");
    CHECK_GT(FLAGS_world_size, 0);
    CHECK_GT(FLAGS_batch_size, 0);
    CHECK_GT(FLAGS_steps, 0);
    CHECK_EQ(FLAGS_batch_size % FLAGS_world_size, 0);
    CHECK(FLAGS_world_size == 1 || FLAGS_device == "cuda");
    CHECK(!FLAGS_fixture_dir.empty() && !FLAGS_output_dir.empty());
    nn::parallel::global::InitAllEnv(FLAGS_world_size, 1, false, 1, 1);
    if (FLAGS_world_size > 1) {
        nn::parallel::ProcessGroupFactory::Instance(Device::DeviceType::kCUDA);
    }
    std::vector<std::shared_ptr<MNIST>> networks;
    for (int index = 0; index < FLAGS_world_size; ++index) {
        auto network = std::make_shared<MNIST>();
        for (const auto &[name, parameter] : network->NamedParameters()) {
            Read(fs::path(FLAGS_fixture_dir) / (name + ".bin"), *parameter);
        }
        networks.push_back(network);
    }
    if (FLAGS_world_size == 1) {
        Run(0, networks[0]);
    } else {
        std::vector<std::thread> threads;
        for (int index = 0; index < FLAGS_world_size; ++index) { threads.emplace_back(Run, index, networks[index]); }
        for (auto &thread : threads) { thread.join(); }
    }
    return 0;
}
