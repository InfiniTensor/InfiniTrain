// MnistCnn <-> torch parity fixture (OSpec P12).
//
// Dumps a fully deterministic 10-step SGD trajectory of MnistCnn
// (Conv2d(1,16,3)->ReLU->Conv2d(16,32,3)->ReLU->Flatten->Linear(18432,10))
// on a fixed synthetic batch so an out-of-repo torch script can compare
// logits / loss / grads / updated params at 1e-5.
//
// Layout (all little-endian, row-major / C-contiguous flat order):
//   meta.txt                      human-readable shapes, dtypes, hyperparams, file map
//   input.bin                     float32 [B,1,28,28]
//   labels.bin                    uint8   [B]
//   init__<name>.bin              float32, initial param `<name>` (e.g. conv1.weight)
//   step<NNN>__logits.bin        float32 [B,10], NNN = 000..009
//   step<NNN>__loss.bin          float32 scalar (1 element, mean CE loss)
//   step<NNN>__grad__<name>.bin  float32, grad of `<name>` BEFORE the SGD step
//   step<NNN>__param__<name>.bin float32, value of `<name>` AFTER the SGD step
//
// Determinism notes:
//   * Params are NOT the constructor RNG values: they are refilled here with a
//     single-threaded std::mt19937(42), U(-bound,bound), bound = 1/sqrt(fan_in),
//     consumed in sorted NamedParameters order. This matches the torch default
//     init math (kaiming_uniform_(a=sqrt(5)) == U(-1/sqrt(fan_in),1/sqrt(fan_in)))
//     while staying independent of the framework global RNG / OMP thread count.
//   * Input uses std::mt19937(1234) U[0,1); labels use std::mt19937(5678) % 10.
//   * Output dir: $MNIST_PARITY_OUT_DIR or ./mnist_parity_dump.
//
// Run:  MNIST_PARITY_OUT_DIR=/tmp/opencode/mnist_parity \
//         ./build/tests/example/test_mnist_parity

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "Eigen/Dense"

#ifdef USE_OMP
#include <omp.h>
#endif

#include "example/mnist/net.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace {

constexpr int kBatch = 4;
constexpr int kSteps = 10;
constexpr float kLr = 0.01f;
constexpr uint32_t kParamSeed = 42;
constexpr uint32_t kInputSeed = 1234;
constexpr uint32_t kLabelSeed = 5678;

std::string OutDir() {
    if (const char *env = std::getenv("MNIST_PARITY_OUT_DIR")) {
        if (env[0] != '\0') {
            return env;
        }
    }
    return "mnist_parity_dump";
}

std::string StepTag(int step) {
    std::ostringstream oss;
    oss << "step" << std::setw(3) << std::setfill('0') << step;
    return oss.str();
}

void WriteFloats(const std::filesystem::path &path, const float *data, size_t n) {
    std::ofstream ofs(path, std::ios::binary);
    ASSERT_TRUE(ofs.good()) << path;
    ofs.write(reinterpret_cast<const char *>(data), n * sizeof(float));
    ASSERT_TRUE(ofs.good()) << path;
}

void WriteBytes(const std::filesystem::path &path, const uint8_t *data, size_t n) {
    std::ofstream ofs(path, std::ios::binary);
    ASSERT_TRUE(ofs.good()) << path;
    ofs.write(reinterpret_cast<const char *>(data), n * sizeof(uint8_t));
    ASSERT_TRUE(ofs.good()) << path;
}

void DumpTensorFp32(const std::filesystem::path &path, const std::shared_ptr<infini_train::Tensor> &t) {
    ASSERT_NE(t, nullptr) << path;
    ASSERT_EQ(t->Dtype(), infini_train::DataType::kFLOAT32) << path;
    // Fixture runs on CPU; tensors are CPU already.
    WriteFloats(path, static_cast<const float *>(t->DataPtr()), t->NumElements());
}

std::string DimsStr(const std::vector<int64_t> &dims) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        oss << (i ? "," : "") << dims[i];
    }
    oss << "]";
    return oss.str();
}

// fan_in for bound = 1/sqrt(fan_in): weight [O,I,...] -> I*prod(rest);
// bias "m.bias" reuses sibling "m.weight" fan_in.
int64_t FanIn(const std::string &name, const std::vector<int64_t> &dims,
              const std::vector<std::pair<std::string, std::shared_ptr<infini_train::Tensor>>> &named) {
    if (dims.size() >= 2) {
        int64_t fan = dims[1];
        for (size_t i = 2; i < dims.size(); ++i) { fan *= dims[i]; }
        return fan;
    }
    const auto pos = name.rfind('.');
    if (pos != std::string::npos) {
        const std::string wname = name.substr(0, pos) + ".weight";
        for (const auto &[n, t] : named) {
            if (n == wname) { return FanIn(wname, t->Dims(), named); }
        }
    }
    return dims.empty() ? 1 : dims[0];
}

bool AllFinite(const float *data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(data[i])) {
            return false;
        }
    }
    return true;
}

TEST(MnistParity, DumpTrajectory) {
    namespace it = infini_train;
    // Pin single-threaded execution: Eigen/OpenMP parallel reductions change FP
    // summation order with the thread count (~2e-8 jitter), which would make the
    // dumped trajectory environment-dependent.
#ifdef USE_OMP
    omp_set_num_threads(1);
#endif
    Eigen::setNbThreads(1);

    const std::filesystem::path out = OutDir();
    std::filesystem::create_directories(out);

    auto net = std::make_shared<MnistCnn>();
    auto named = net->NamedParameters();
    std::sort(named.begin(), named.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    ASSERT_EQ(named.size(), 6);
    EXPECT_EQ(named[0].first, "conv1.bias");
    EXPECT_EQ(named[1].first, "conv1.weight");
    EXPECT_EQ(named[2].first, "conv2.bias");
    EXPECT_EQ(named[3].first, "conv2.weight");
    EXPECT_EQ(named[4].first, "fc.bias");
    EXPECT_EQ(named[5].first, "fc.weight");

    // Deterministic init refill (single-threaded mt19937, sorted order).
    {
        std::mt19937 gen(kParamSeed);
        for (const auto &[name, param] : named) {
            ASSERT_EQ(param->Dtype(), it::DataType::kFLOAT32) << name;
            const int64_t fan_in = FanIn(name, param->Dims(), named);
            ASSERT_GT(fan_in, 0) << name;
            const float bound = 1.0f / std::sqrt(static_cast<float>(fan_in));
            std::uniform_real_distribution<float> dis(-bound, bound);
            float *dst = static_cast<float *>(param->DataPtr());
            for (size_t i = 0; i < param->NumElements(); ++i) { dst[i] = dis(gen); }
        }
    }

    // Fixed synthetic input: U[0,1) like normalized MNIST pixels.
    std::vector<float> input_data(kBatch * 1 * 28 * 28);
    {
        std::mt19937 gen(kInputSeed);
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);
        for (auto &v : input_data) { v = dis(gen); }
    }
    auto input = std::make_shared<it::Tensor>(std::vector<int64_t>{kBatch, 1, 28, 28}, it::DataType::kFLOAT32,
                                              it::Device());
    std::copy(input_data.begin(), input_data.end(), static_cast<float *>(input->DataPtr()));

    // Fixed labels.
    std::vector<uint8_t> label_data(kBatch);
    {
        std::mt19937 gen(kLabelSeed);
        for (auto &v : label_data) { v = static_cast<uint8_t>(gen() % 10); }
    }
    auto labels = std::make_shared<it::Tensor>(std::vector<int64_t>{kBatch}, it::DataType::kUINT8, it::Device());
    std::copy(label_data.begin(), label_data.end(), static_cast<uint8_t *>(labels->DataPtr()));

    auto loss_fn = std::make_shared<it::nn::CrossEntropyLoss>();
    auto optimizer = it::optimizers::SGD(net->Parameters(), kLr);

    // meta.txt first (shapes/dtypes/hparams), then blobs.
    {
        std::ofstream meta(out / "meta.txt");
        ASSERT_TRUE(meta.good());
        meta << "# MnistCnn parity fixture (OSpec P12)\n";
        meta << "model: Conv2d(1,16,3,s1,p0)+ReLU+Conv2d(16,32,3,s1,p0)+ReLU+Flatten(1)+Linear(18432,10)\n";
        meta << "batch: " << kBatch << "\nsteps: " << kSteps << "\nloss: CrossEntropy(mean)\noptimizer: SGD lr="
             << kLr << "\n";
        meta << "seeds: param=" << kParamSeed << " input=" << kInputSeed << " label=" << kLabelSeed << "\n";
        meta << "input: float32 " << DimsStr(input->Dims()) << " input.bin\n";
        meta << "labels: uint8 " << DimsStr(labels->Dims()) << " labels.bin values=";
        for (int i = 0; i < kBatch; ++i) { meta << (i ? "," : "") << static_cast<int>(label_data[i]); }
        meta << "\n";
        for (const auto &[name, param] : named) {
            meta << "param " << name << ": float32 " << DimsStr(param->Dims()) << " init__" << name
                 << ".bin step<NNN>__grad__" << name << ".bin step<NNN>__param__" << name << ".bin\n";
        }
        meta << "logits: float32 [B,10] step<NNN>__logits.bin\nloss: float32 scalar step<NNN>__loss.bin\n";
        ASSERT_TRUE(meta.good());
    }

    WriteFloats(out / "input.bin", input_data.data(), input_data.size());
    WriteBytes(out / "labels.bin", label_data.data(), label_data.size());
    for (const auto &[name, param] : named) { DumpTensorFp32(out / ("init__" + name + ".bin"), param); }

    std::vector<float> losses;
    for (int step = 0; step < kSteps; ++step) {
        auto outputs = net->Forward({input});
        ASSERT_EQ(outputs.size(), 1);
        ASSERT_EQ(outputs[0]->Dims(), (std::vector<int64_t>{kBatch, 10})) << "step " << step;
        optimizer.ZeroGrad();
        auto loss = loss_fn->Forward({outputs[0], labels});
        ASSERT_EQ(loss.size(), 1);
        loss[0]->Backward();

        const float *logit_ptr = static_cast<const float *>(outputs[0]->DataPtr());
        const float loss_val = static_cast<const float *>(loss[0]->DataPtr())[0];
        ASSERT_TRUE(AllFinite(logit_ptr, outputs[0]->NumElements())) << "step " << step;
        ASSERT_TRUE(std::isfinite(loss_val)) << "step " << step;
        losses.push_back(loss_val);

        const std::string tag = StepTag(step);
        DumpTensorFp32(out / (tag + "__logits.bin"), outputs[0]);
        WriteFloats(out / (tag + "__loss.bin"), &loss_val, 1);
        for (const auto &[name, param] : named) {
            ASSERT_NE(param->grad(), nullptr) << "step " << step << " " << name;
            DumpTensorFp32(out / (tag + "__grad__" + name + ".bin"), param->grad());
        }

        optimizer.Step();
        for (const auto &[name, param] : named) { DumpTensorFp32(out / (tag + "__param__" + name + ".bin"), param); }

        std::cout << "[parity] " << tag << " loss=" << loss_val << "\n";
    }
    std::cout << "[parity] dumped " << (3 + named.size()) << " init files + " << kSteps * (2 + 2 * named.size())
              << " step files to " << out << "\n";
}

} // namespace
