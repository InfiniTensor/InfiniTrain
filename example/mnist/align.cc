// Deterministic numerical-alignment entry point. PyTorch fixture generation
// and comparison scripts are distributed separately from the framework PR.
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "example/mnist/distributed.h"
#include "example/mnist/net.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/optimizer.h"

DEFINE_string(fixture_dir, "", "Fixture directory exported by the PyTorch reference");
DEFINE_string(output_dir, "", "New directory for logits, loss, gradients and updated weights");
DEFINE_string(device, "cpu", "cpu or cuda");
DEFINE_bool(ddp, false, "Launch one process per GPU using infini_run");
DEFINE_bool(ddp_buckets, true, "Enable gradient bucketing");

using namespace infini_train;
namespace {
void Read(const std::filesystem::path &path, const std::shared_ptr<Tensor> &tensor) {
    CHECK(tensor->GetDevice().IsCPU());
    CHECK_EQ(std::filesystem::file_size(path), tensor->SizeInBytes()) << path;
    std::ifstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;
    file.read(static_cast<char *>(tensor->DataPtr()), tensor->SizeInBytes());
    CHECK(file.good()) << "Incomplete fixture: " << path;
}

void Write(const std::filesystem::path &path, const std::shared_ptr<Tensor> &tensor) {
    CHECK(tensor) << "Missing tensor: " << path;
    CHECK(tensor->Dtype() == DataType::kFLOAT32);
    auto cpu = tensor->To(Device());
    const auto device = tensor->GetDevice();
    core::GetDeviceGuardImpl(device.type())->SynchronizeDevice(device);
    const auto *data = static_cast<const float *>(cpu.DataPtr());
    for (size_t i = 0; i < cpu.NumElements(); ++i) { CHECK(std::isfinite(data[i])) << path << ": " << i; }
    std::ofstream file(path, std::ios::binary);
    CHECK(file.is_open()) << path;
    file.write(reinterpret_cast<const char *>(data), cpu.SizeInBytes());
    CHECK(file.good()) << "Failed to write " << path;
}
} // namespace

int main(int argc, char **argv) {
    static_assert(std::endian::native == std::endian::little, "Alignment fixtures use little-endian arrays");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    CHECK(!FLAGS_fixture_dir.empty() && !FLAGS_output_dir.empty());
    CHECK(FLAGS_device == "cpu" || FLAGS_device == "cuda");
#ifndef USE_CUDA
    CHECK_EQ(FLAGS_device, "cpu") << "Rebuild with USE_CUDA=ON";
#endif
    const mnist::DistributedContext distributed(FLAGS_ddp, FLAGS_device == "cuda");
    const std::filesystem::path fixture(FLAGS_fixture_dir);
    const auto output = FLAGS_ddp
                          ? std::filesystem::path(FLAGS_output_dir) / ("rank" + std::to_string(distributed.rank))
                          : std::filesystem::path(FLAGS_output_dir);
    CHECK(!std::filesystem::exists(output) || std::filesystem::is_empty(output))
        << "Use a new output directory; existing results are preserved";
    std::ifstream config(fixture / "case.txt");
    std::string version;
    int64_t batch = 0, steps = 0;
    float lr = 0;
    CHECK(static_cast<bool>(config >> version >> batch >> steps >> lr));
    CHECK_EQ(version, "mnist-alignment-v1");
    CHECK(batch > 0 && batch <= 1024);
    CHECK(steps > 0 && steps <= 100);
    CHECK(std::isfinite(lr) && lr > 0);
    auto network = std::make_shared<MNIST>();
    for (const auto &[name, param] : network->NamedParameters()) { Read(fixture / (name + ".bin"), param); }
    auto input = std::make_shared<Tensor>(std::vector<int64_t>{batch, 784}, DataType::kFLOAT32);
    auto labels = std::make_shared<Tensor>(std::vector<int64_t>{batch}, DataType::kINT64);
    Read(fixture / "input.bin", input);
    Read(fixture / "labels.bin", labels);
    const auto *target = static_cast<const int64_t *>(labels->DataPtr());
    for (int64_t i = 0; i < batch; ++i) { CHECK(target[i] >= 0 && target[i] < MNIST::kNumClasses); }
    CHECK_EQ(batch % distributed.world_size, 0) << "Alignment batch must divide evenly across ranks";
    const int64_t local_batch = batch / distributed.world_size;
    const int64_t offset = distributed.rank * local_batch;
    input = std::make_shared<Tensor>(*input, offset * 784 * sizeof(float), std::vector<int64_t>{local_batch, 784});
    labels = std::make_shared<Tensor>(*labels, offset * sizeof(int64_t), std::vector<int64_t>{local_batch});
    const auto device = distributed.device;
    network->To(device);
    auto model = distributed.Wrap(network, FLAGS_ddp_buckets);
    input = std::make_shared<Tensor>(input->To(device))->RequiresGrad();
    labels = std::make_shared<Tensor>(labels->To(device));
    optimizers::SGD optimizer(network->NamedParameters(), lr);
    nn::CrossEntropyLoss loss_fn;
    std::filesystem::create_directories(output);
    // Initial snapshots also prove the candidate loaded the fixture unchanged.
    for (const auto &[name, param] : network->NamedParameters()) {
        Write(output / ("initial." + name + ".bin"), param);
    }
    Write(output / "input.bin", input);
    for (int64_t step = 0; step < steps; ++step) {
        optimizer.ZeroGrad();
        input->ZeroGrad();
        auto logits = (*model)({input})[0];
        auto loss = loss_fn({logits, labels})[0];
        loss->Backward();
        const auto prefix = "step" + std::to_string(step) + ".";
        Write(output / (prefix + "logits.bin"), logits);
        Write(output / (prefix + "loss.bin"), loss);
        {
            autograd::NoGradGuard no_grad;
            // Input gradients are local (DDP only reduces parameter grads).
            // Export derivatives of the GLOBAL mean loss for concatenation.
            Write(output / (prefix + "input_grad.bin"), input->grad()->Mul(1.0f / distributed.world_size));
        }
        for (const auto &[name, param] : network->NamedParameters()) {
            Write(output / (prefix + "grad." + name + ".bin"), param->grad());
        }
        optimizer.Step();
        for (const auto &[name, param] : network->NamedParameters()) {
            Write(output / (prefix + "updated." + name + ".bin"), param);
        }
        std::cout << "Completed alignment step " << step << " on " << FLAGS_device << std::endl;
    }
    std::ofstream complete(output / "complete.txt");
    complete << version << '\n'
             << batch << ' ' << steps << ' ' << std::setprecision(9) << lr << '\n'
             << FLAGS_device << '\n';
    CHECK(complete.good());
    complete.close();
    std::ofstream rank_info(output / "rank.txt");
    rank_info << distributed.rank << ' ' << distributed.world_size << ' ' << local_batch << '\n';
    rank_info.close();
    distributed.Barrier();
    return 0;
}
