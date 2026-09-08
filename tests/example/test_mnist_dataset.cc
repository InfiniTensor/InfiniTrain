#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "example/mnist/dataset.h"

namespace {

void WriteBigEndianU32(std::ofstream *stream, uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8) { stream->put(static_cast<char>((value >> shift) & 0xffU)); }
}

void WriteMnistFixture(const std::filesystem::path &directory) {
    std::filesystem::create_directories(directory);

    std::ofstream images(directory / "train-images-idx3-ubyte", std::ios::binary);
    ASSERT_TRUE(images.is_open());
    WriteBigEndianU32(&images, 0x00000803U);
    WriteBigEndianU32(&images, 2);
    WriteBigEndianU32(&images, 28);
    WriteBigEndianU32(&images, 28);
    for (int sample = 0; sample < 2; ++sample) {
        for (int pixel = 0; pixel < 28 * 28; ++pixel) { images.put(static_cast<char>((sample * 128 + pixel) % 256)); }
    }

    std::ofstream labels(directory / "train-labels-idx1-ubyte", std::ios::binary);
    ASSERT_TRUE(labels.is_open());
    WriteBigEndianU32(&labels, 0x00000801U);
    WriteBigEndianU32(&labels, 2);
    labels.put(static_cast<char>(3));
    labels.put(static_cast<char>(7));
}

class TemporaryDirectory {
public:
    TemporaryDirectory()
        : path_(std::filesystem::temp_directory_path()
                / ("infinitrain-mnist-dataset-"
                   + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {}

    ~TemporaryDirectory() { std::filesystem::remove_all(path_); }

    const std::filesystem::path &path() const { return path_; }

private:
    std::filesystem::path path_;
};

} // namespace

TEST(MNISTDatasetTest, UsesFloat32StrideAfterImageNormalization) {
    TemporaryDirectory fixture;
    WriteMnistFixture(fixture.path());

    MNISTDataset dataset(fixture.path().string(), true);
    ASSERT_EQ(dataset.Size(), 2U);

    const auto [first_image, first_label] = dataset[0];
    const auto [second_image, second_label] = dataset[1];
    EXPECT_EQ(first_image->Dims(), (std::vector<int64_t>{1, 28, 28}));
    EXPECT_EQ(second_image->Dims(), (std::vector<int64_t>{1, 28, 28}));
    EXPECT_EQ(first_label->Dims(), std::vector<int64_t>{});
    EXPECT_EQ(second_label->Dims(), std::vector<int64_t>{});

    const auto *first_pixels = static_cast<const float *>(first_image->DataPtr());
    const auto *second_pixels = static_cast<const float *>(second_image->DataPtr());
    EXPECT_FLOAT_EQ(first_pixels[0], 0.0f);
    EXPECT_FLOAT_EQ(first_pixels[1], 1.0f / 255.0f);
    EXPECT_FLOAT_EQ(second_pixels[0], 128.0f / 255.0f);
    EXPECT_FLOAT_EQ(second_pixels[1], 129.0f / 255.0f);
    EXPECT_EQ(*static_cast<const uint8_t *>(first_label->DataPtr()), 3);
    EXPECT_EQ(*static_cast<const uint8_t *>(second_label->DataPtr()), 7);
}
