#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/dataloader.h"
#include "infini_train/include/dataset.h"
#include "infini_train/include/tensor.h"

#include "tests/common/test_utils.h"

namespace infini_train {
namespace {

class ImageLikeDataset final : public Dataset {
public:
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> operator[](size_t index) const override {
        const float first = static_cast<float>(index * 4 + 1);
        const std::vector<float> image_values{first, first + 1.0f, first + 2.0f, first + 3.0f};
        const std::vector<float> label_values{static_cast<float>(index)};
        return {std::make_shared<Tensor>(image_values.data(), std::vector<int64_t>{1, 2, 2}, DataType::kFLOAT32),
                std::make_shared<Tensor>(label_values.data(), std::vector<int64_t>{}, DataType::kFLOAT32)};
    }

    size_t Size() const override { return 3; }
};

class DataLoaderShapeTest : public test::InfiniTrainTest {};

TEST_P(DataLoaderShapeTest, PreservesImageSampleDimensionsWhenStacking) {
    ONLY_CPU();
    auto dataset = std::make_shared<ImageLikeDataset>();
    DataLoader loader(dataset, 2);

    const auto [images, labels] = *loader.begin();
    EXPECT_EQ(images->Dims(), (std::vector<int64_t>{2, 1, 2, 2}));
    EXPECT_EQ(labels->Dims(), (std::vector<int64_t>{2}));
    test::ExpectTensorFloatEqual(images, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    test::ExpectTensorFloatEqual(labels, std::vector<float>{0.0f, 1.0f});
}

INFINI_TRAIN_REGISTER_TEST(DataLoaderShapeTest);

} // namespace
} // namespace infini_train
