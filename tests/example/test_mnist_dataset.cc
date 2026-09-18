#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "example/mnist/dataset.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/tensor.h"

namespace {

std::string ResolveMnistDir() {
    if (const char *env = std::getenv("MNIST_DATA_DIR")) {
        if (std::filesystem::exists(env)) {
            return env;
        }
    }
    // Read-only shared copy (see task description); never copied into the worktree.
    constexpr char kSharedPath[] = "/home/Sota/infra/InfiniTrain/data/mnist";
    if (std::filesystem::exists(kSharedPath)) {
        return kSharedPath;
    }
    return "";
}

uint32_t ReadBigEndianU32(std::ifstream &ifs) {
    uint8_t bytes[4];
    ifs.read(reinterpret_cast<char *>(bytes), 4);
    return (static_cast<uint32_t>(bytes[0]) << 24) | (static_cast<uint32_t>(bytes[1]) << 16)
         | (static_cast<uint32_t>(bytes[2]) << 8) | static_cast<uint32_t>(bytes[3]);
}

// Raw IDX pixel bytes for sample `idx`, read straight from disk (no framework code involved).
std::vector<uint8_t> ReadRawImageBytes(const std::string &dir, size_t idx) {
    std::ifstream ifs(dir + "/train-images-idx3-ubyte", std::ios::binary);
    EXPECT_TRUE(ifs.good());
    ifs.seekg(0);
    EXPECT_EQ(ReadBigEndianU32(ifs), 0x00000803u);
    const uint32_t num_images = ReadBigEndianU32(ifs);
    const uint32_t rows = ReadBigEndianU32(ifs);
    const uint32_t cols = ReadBigEndianU32(ifs);
    EXPECT_LT(idx, num_images);
    EXPECT_EQ(rows, 28);
    EXPECT_EQ(cols, 28);
    std::vector<uint8_t> raw(rows * cols);
    ifs.seekg(16 + idx * rows * cols);
    ifs.read(reinterpret_cast<char *>(raw.data()), raw.size());
    EXPECT_TRUE(ifs.good());
    return raw;
}

uint8_t ReadRawLabelByte(const std::string &dir, size_t idx) {
    std::ifstream ifs(dir + "/train-labels-idx1-ubyte", std::ios::binary);
    EXPECT_TRUE(ifs.good());
    EXPECT_EQ(ReadBigEndianU32(ifs), 0x00000801u);
    const uint32_t num_labels = ReadBigEndianU32(ifs);
    EXPECT_LT(idx, num_labels);
    uint8_t label = 0;
    ifs.seekg(8 + idx);
    ifs.read(reinterpret_cast<char *>(&label), 1);
    EXPECT_TRUE(ifs.good());
    return label;
}

void ExpectSampleMatchesRaw(const MNISTDataset &dataset, const std::string &dir, size_t idx) {
    const auto [image, label] = dataset[idx];
    const std::vector<uint8_t> raw = ReadRawImageBytes(dir, idx);
    ASSERT_EQ(image->NumElements(), raw.size());
    const auto *data = static_cast<const float *>(image->DataPtr());
    for (size_t i = 0; i < raw.size(); ++i) {
        EXPECT_FLOAT_EQ(data[i], raw[i] / 255.0f) << "idx=" << idx << " pixel=" << i;
    }
    ASSERT_EQ(label->NumElements(), 1);
    EXPECT_EQ(*static_cast<const uint8_t *>(label->DataPtr()), ReadRawLabelByte(dir, idx)) << "idx=" << idx;
}

TEST(MNISTDatasetTest, SampleDimsAndDtypes) {
    const std::string dir = ResolveMnistDir();
    if (dir.empty()) {
        GTEST_SKIP() << "MNIST data not found (set MNIST_DATA_DIR)";
    }
    MNISTDataset dataset(dir, true);
    const auto [image, label] = dataset[0];
    EXPECT_EQ(image->Dims(), (std::vector<int64_t>{1, 28, 28}));
    EXPECT_EQ(image->Dtype(), infini_train::DataType::kFLOAT32);
    EXPECT_EQ(label->Dtype(), infini_train::DataType::kUINT8);
    EXPECT_EQ(label->NumElements(), 1);
}

TEST(MNISTDatasetTest, NormalizationMatchesRawBytes) {
    const std::string dir = ResolveMnistDir();
    if (dir.empty()) {
        GTEST_SKIP() << "MNIST data not found (set MNIST_DATA_DIR)";
    }
    MNISTDataset dataset(dir, true);
    ExpectSampleMatchesRaw(dataset, dir, 0);
}

TEST(MNISTDatasetTest, NonZeroIndicesAreCorrectlyAligned) {
    const std::string dir = ResolveMnistDir();
    if (dir.empty()) {
        GTEST_SKIP() << "MNIST data not found (set MNIST_DATA_DIR)";
    }
    MNISTDataset dataset(dir, true);
    // idx > 0 regresses the byte-stride bug (UINT8 stride used on a FLOAT32 buffer).
    ExpectSampleMatchesRaw(dataset, dir, 1);
    ExpectSampleMatchesRaw(dataset, dir, 100);
}
} // namespace
