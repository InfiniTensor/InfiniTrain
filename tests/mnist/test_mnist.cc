#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>

#include "example/mnist/dataset.h"
#include "example/mnist/net.h"
#include "example/mnist/training.h"
#include "infini_train/include/autograd/grad_mode.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dataloader.h"
#include "infini_train/include/nn/modules/loss.h"
#include "infini_train/include/optimizer.h"
#include "tests/common/test_utils.h"

using namespace infini_train;

namespace {
// Real IDX encoding, deliberately distinct images to catch byte-offset errors
// after the dataset converts its storage from UINT8 to FP32.
class IdxFixture {
public:
    IdxFixture() {
        path = std::filesystem::temp_directory_path()
             / ("infinitrain-mnist-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        CHECK(std::filesystem::create_directory(path));
        for (const auto *prefix : {"train", "t10k"}) {
            std::ofstream images(path / (std::string(prefix) + "-images-idx3-ubyte"), std::ios::binary);
            for (uint32_t v : {2051u, 3u, 28u, 28u}) { WriteInt(images, v); }
            for (unsigned char v : {0, 127, 255}) {
                const std::string image(784, static_cast<char>(v));
                images.write(image.data(), image.size());
            }
            std::ofstream labels(path / (std::string(prefix) + "-labels-idx1-ubyte"), std::ios::binary);
            for (uint32_t v : {2049u, 3u}) { WriteInt(labels, v); }
            const char values[] = {1, 5, 9};
            labels.write(values, sizeof(values));
            CHECK(images.good());
            CHECK(labels.good());
        }
    }
    ~IdxFixture() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    std::filesystem::path path;

private:
    static void WriteInt(std::ofstream &out, uint32_t v) {
        for (int shift : {24, 16, 8, 0}) { out.put(static_cast<char>((v >> shift) & 255)); }
    }
};
} // namespace

class MnistCnnTest : public test::InfiniTrainTest {
protected:
    void SetUp() override {
#ifdef USE_CUDA
        if (GetDevice().IsCUDA()) {
            int count = 0;
            const auto status = cudaGetDeviceCount(&count);
            if (status != cudaSuccess) {
                GTEST_SKIP() << cudaGetErrorString(status);
            }
            if (!count) {
                GTEST_SKIP() << "No CUDA device available";
            }
        }
#endif
    }

    std::shared_ptr<MNIST> Network() {
        auto net = std::make_shared<MNIST>();
        net->To(GetDevice());
        return net;
    }

    std::shared_ptr<Tensor> Input(int64_t batch) {
        std::vector<float> v(batch * 784);
        for (size_t i = 0; i < v.size(); ++i) { v[i] = static_cast<float>(i % 251) / 255.0f; }
        return std::make_shared<Tensor>(v.data(), std::vector<int64_t>{batch, 784}, DataType::kFLOAT32, GetDevice());
    }

    std::vector<float> Values(const std::shared_ptr<Tensor> &t) {
        auto cpu = t->To(Device());
        core::GetDeviceGuardImpl(GetDevice().type())->SynchronizeDevice(GetDevice());
        auto p = static_cast<const float *>(cpu.DataPtr());
        return {p, p + cpu.NumElements()};
    }
};

TEST_P(MnistCnnTest, LayerShapesAndNamedParameters) {
    auto net = Network();
    const std::map<std::string, std::vector<int64_t>> expected_params = {
        {"conv1.weight", {16, 1, 3, 3}}, {"conv1.bias", {16}}, {"conv2.weight", {32, 16, 3, 3}}, {"conv2.bias", {32}},
        {"fc.weight", {10, 18432}},      {"fc.bias", {10}}};
    size_t numel = 0;
    ASSERT_EQ(net->NamedParameters().size(), expected_params.size());
    for (const auto &[name, p] : net->NamedParameters()) {
        EXPECT_EQ(p->Dims(), expected_params.at(name));
        EXPECT_TRUE(p->requires_grad());
        EXPECT_EQ(p->GetDevice(), GetDevice());
        numel += p->NumElements();
    }
    EXPECT_EQ(numel, 189130);
    autograd::NoGradGuard guard;
    auto x = Input(2)->View({2, 1, 28, 28});
    for (const auto &[name, shape] :
         std::vector<std::pair<std::string, std::vector<int64_t>>>{{"conv1", {2, 16, 26, 26}},
                                                                   {"relu1", {2, 16, 26, 26}},
                                                                   {"conv2", {2, 32, 24, 24}},
                                                                   {"relu2", {2, 32, 24, 24}},
                                                                   {"flatten", {2, 18432}},
                                                                   {"fc", {2, 10}}}) {
        x = (*net->mutable_module(name))({x})[0];
        EXPECT_EQ(x->Dims(), shape) << name;
    }
}

TEST_P(MnistCnnTest, FlatAndNchwInputsAgreeForVariableBatches) {
    auto net = Network();
    // A constant classifier bias confirms the network returns raw logits,
    // without accidentally applying sigmoid or softmax before the loss.
    net->mutable_module("fc")->parameter("weight")->Fill(0.0f);
    net->mutable_module("fc")->parameter("bias")->Fill(2.0f);
    autograd::NoGradGuard guard;
    for (int64_t batch : {1, 3}) {
        auto flat = Input(batch);
        auto a = (*net)({flat})[0];
        auto b = (*net)({flat->View({batch, 1, 28, 28})})[0];
        EXPECT_EQ(a->Dims(), (std::vector<int64_t>{batch, 10}));
        EXPECT_EQ(Values(a), Values(b));
        for (float v : Values(a)) { EXPECT_EQ(v, 2.0f); }
        EXPECT_FALSE(a->requires_grad());
        EXPECT_EQ(a->grad_fn(), nullptr);
    }
    // Also compare nonconstant logits so layout errors cannot hide behind a zero head.
    net = Network();
    auto flat = Input(2);
    EXPECT_EQ(Values((*net)({flat})[0]), Values((*net)({flat->View({2, 1, 28, 28})})[0]));
}

TEST_P(MnistCnnTest, LossBackwardAndSgdReachEveryLayer) {
    auto net = Network();
    for (const auto *name : {"conv1", "conv2"}) {
        net->mutable_module(name)->parameter("weight")->Fill(0.01f);
        net->mutable_module(name)->parameter("bias")->Fill(0.02f);
    }
    std::vector<float> weights(10 * 18432);
    for (size_t i = 0; i < weights.size(); ++i) { weights[i] = (static_cast<int>(i / 18432) - 4) * 0.0001f; }
    auto w = std::make_shared<Tensor>(weights.data(), std::vector<int64_t>{10, 18432}, DataType::kFLOAT32, GetDevice());
    net->mutable_module("fc")->parameter("weight")->CopyFrom(w);
    net->mutable_module("fc")->parameter("bias")->Fill(0.0f);
    auto x = Input(2)->RequiresGrad();
    const float label_values[] = {3, 7};
    auto labels = std::make_shared<Tensor>(label_values, std::vector<int64_t>{2}, DataType::kFLOAT32, GetDevice());
    labels = std::make_shared<Tensor>(labels->To(DataType::kINT64));
    nn::CrossEntropyLoss loss_fn;
    optimizers::SGD optimizer(net->Parameters(), 0.001f);
    auto loss = loss_fn({(*net)({x})[0], labels})[0];
    const float before_loss = Values(loss)[0];
    ASSERT_TRUE(std::isfinite(before_loss));
    loss->Backward();
    ASSERT_NE(x->grad(), nullptr);
    EXPECT_EQ(x->grad()->Dims(), x->Dims());
    std::map<std::string, std::vector<float>> expected;
    for (const auto &[name, p] : net->NamedParameters()) {
        ASSERT_NE(p->grad(), nullptr) << name;
        EXPECT_EQ(p->grad()->Dims(), p->Dims());
        auto values = Values(p), grad = Values(p->grad());
        bool nonzero = false;
        for (size_t i = 0; i < values.size(); ++i) {
            ASSERT_TRUE(std::isfinite(grad[i])) << name;
            nonzero |= grad[i] != 0.0f;
            values[i] -= 0.001f * grad[i];
        }
        EXPECT_TRUE(nonzero) << name;
        expected[name] = std::move(values);
    }
    optimizer.Step();
    for (const auto &[name, p] : net->NamedParameters()) {
        auto values = Values(p);
        for (size_t i = 0; i < values.size(); ++i) { ASSERT_NEAR(values[i], expected.at(name)[i], 1e-6f) << name; }
    }
    optimizer.ZeroGrad();
    for (const auto &p : net->Parameters()) { EXPECT_EQ(p->grad(), nullptr); }
    autograd::NoGradGuard guard;
    EXPECT_LT(Values(loss_fn({(*net)({x})[0], labels})[0])[0], before_loss);
}

TEST_P(MnistCnnTest, IdxImagesAndPartialBatchesReachNetwork) {
    IdxFixture files;
    auto net = Network();
    autograd::NoGradGuard guard;
    for (bool train : {true, false}) {
        auto dataset = std::make_shared<MNISTDataset>(files.path.string(), train);
        ASSERT_EQ(dataset->Size(), 3);
        const std::vector<float> pixel = {0, 127.0f / 255.0f, 1};
        const std::vector<uint8_t> expected_labels = {1, 5, 9};
        for (int i = 0; i < 3; ++i) {
            auto [image, label] = (*dataset)[i];
            EXPECT_EQ(image->Dims(), (std::vector<int64_t>{28, 28}));
            EXPECT_EQ(image->Dtype(), DataType::kFLOAT32);
            for (size_t j = 0; j < 784; ++j) { ASSERT_FLOAT_EQ(static_cast<float *>(image->DataPtr())[j], pixel[i]); }
            EXPECT_EQ(*static_cast<uint8_t *>(label->DataPtr()), expected_labels[i]);
        }
        DataLoader loader(dataset, 2);
        int consumed = 0;
        for (const auto &[images, labels] : loader) {
            const int64_t n = consumed == 0 ? 2 : 1;
            EXPECT_EQ(images->Dims(), (std::vector<int64_t>{n, 784}));
            auto x = std::make_shared<Tensor>(images->To(GetDevice()));
            auto logits = (*net)({x})[0];
            EXPECT_EQ(logits->Dims(), (std::vector<int64_t>{n, 10}));
            for (float v : Values(logits)) { EXPECT_TRUE(std::isfinite(v)); }
            for (int j = 0; j < n; ++j) {
                EXPECT_EQ(static_cast<uint8_t *>(labels->DataPtr())[j], expected_labels[consumed + j]);
            }
            consumed += n;
        }
        EXPECT_EQ(consumed, 3);
    }
}

TEST_P(MnistCnnTest, RejectsInvalidImageShapes) {
    if (GetDevice().IsCUDA()) {
        GTEST_SKIP() << "Validation is shared with CPU";
    }
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    auto net = Network();
    auto input = Input(1);
    EXPECT_DEATH((*net)({}), "x.size");
    EXPECT_DEATH((*net)({input->View({1, 28, 28})}), "expects");
    EXPECT_DEATH((*net)({input->View({1, 1, 14, 56})}), "dims");
    EXPECT_DEATH((*net)({input->View({1, 2, 14, 28})}), "dims");
    EXPECT_DEATH((*net)({input->View({2, 392})}), "dims");
    auto integer = std::make_shared<Tensor>(input->To(DataType::kINT64));
    EXPECT_DEATH((*net)({integer}), "FP32");
}

TEST_P(MnistCnnTest, TrainingMetricsWeightPartialBatchesAndEvaluationDoesNotBuildGraph) {
    mnist::Metrics metrics;
    metrics.Add(2.0f, 1, 2);
    metrics.Add(5.0f, 1, 1);
    EXPECT_DOUBLE_EQ(metrics.Loss(), 3.0);
    EXPECT_DOUBLE_EQ(metrics.Accuracy(), 2.0 / 3.0);
    IdxFixture files;
    auto dataset = std::make_shared<MNISTDataset>(files.path.string(), false);
    auto net = Network();
    auto &fc = net->mutable_module("fc");
    fc->parameter("weight")->Fill(0.0f);
    std::vector<float> bias(10);
    std::iota(bias.begin(), bias.end(), 0.0f);
    fc->parameter("bias")->CopyFrom(
        std::make_shared<Tensor>(bias.data(), std::vector<int64_t>{10}, DataType::kFLOAT32, GetDevice()));
    auto a = mnist::Evaluate(*net, DataLoader(dataset, 2), GetDevice());
    auto b = mnist::Evaluate(*net, DataLoader(dataset, 3), GetDevice());
    EXPECT_EQ(a.samples, 3);
    EXPECT_EQ(a.correct, 1);
    EXPECT_NEAR(a.Loss(), b.Loss(), 1e-6);
    double exp_sum = 0;
    for (int i = 0; i < 10; ++i) { exp_sum += std::exp(static_cast<double>(i)); }
    EXPECT_NEAR(a.Loss(), std::log(exp_sum) - 5.0, 1e-6);
    EXPECT_TRUE(autograd::GradMode::IsEnabled());
    for (auto &p : net->Parameters()) { EXPECT_EQ(p->grad(), nullptr); }
}

TEST_P(MnistCnnTest, InitializationAndShufflingAreSeeded) {
    auto a = std::make_shared<MNIST>();
    auto b = std::make_shared<MNIST>();
    mnist::Initialize(*a, 42);
    mnist::Initialize(*b, 42);
    const auto pa = a->NamedParameters(), pb = b->NamedParameters();
    for (size_t i = 0; i < pa.size(); ++i) {
        EXPECT_EQ(pa[i].first, pb[i].first);
        const auto *x = static_cast<const float *>(pa[i].second->DataPtr());
        const auto *y = static_cast<const float *>(pb[i].second->DataPtr());
        EXPECT_TRUE(std::equal(x, x + pa[i].second->NumElements(), y));
    }
    mnist::Initialize(*b, 43);
    EXPECT_NE(*static_cast<float *>(pa[0].second->DataPtr()), *static_cast<float *>(pb[0].second->DataPtr()));
    IdxFixture files;
    auto dataset = std::make_shared<MNISTDataset>(files.path.string(), true);
    mnist::ShuffledDataset sampler(dataset);
    sampler.Reset(42);
    auto order = sampler.order();
    sampler.Reset(97);
    sampler.Reset(42);
    EXPECT_EQ(sampler.order(), order);
    std::sort(order.begin(), order.end());
    EXPECT_EQ(order, (std::vector<size_t>{0, 1, 2}));
    for (size_t i = 0; i < sampler.Size(); ++i) {
        EXPECT_EQ(*static_cast<uint8_t *>(sampler[i].second->DataPtr()),
                  *static_cast<uint8_t *>((*dataset)[sampler.order()[i]].second->DataPtr()));
    }
    sampler.Reset(42, false);
    EXPECT_EQ(sampler.order(), order);
}

INFINI_TRAIN_REGISTER_TEST(MnistCnnTest);

// One representative sampler test covers uneven sizes, deterministic reshuffle,
// disjoint shards, full evaluation coverage and an empty evaluation rank.
TEST(MnistDistributedSampler, DisjointDeterministicShardsAndTailPolicy) {
    class IndexDataset : public Dataset {
    public:
        size_t Size() const override { return 11; }
        std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> operator[](size_t) const override { return {}; }
    };
    auto source = std::make_shared<IndexDataset>();
    mnist::ShuffledDataset full(source), a(source, 0, 2, true), b(source, 1, 2, true);
    full.Reset(42);
    a.Reset(42);
    b.Reset(42);
    ASSERT_EQ(a.Size(), 5);
    ASSERT_EQ(b.Size(), 5);
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_EQ(a.order()[i], full.order()[2 * i]);
        EXPECT_EQ(b.order()[i], full.order()[2 * i + 1]);
    }
    auto previous = a.order();
    a.Reset(43);
    EXPECT_NE(previous, a.order());
    a.Reset(42);
    EXPECT_EQ(previous, a.order());
    // local bs=3 gives batches 3+2 on both ranks, so the tail is retained.
    EXPECT_EQ((a.Size() + 2) / 3, (b.Size() + 2) / 3);
    mnist::ShuffledDataset eval_a(source, 0, 2), eval_b(source, 1, 2), empty(source, 11, 12);
    auto combined = eval_a.order();
    combined.insert(combined.end(), eval_b.order().begin(), eval_b.order().end());
    std::sort(combined.begin(), combined.end());
    EXPECT_EQ(combined, (std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
    EXPECT_EQ(empty.Size(), 0);
}
