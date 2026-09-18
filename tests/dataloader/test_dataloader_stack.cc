#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/dataloader.h"
#include "infini_train/include/dataset.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

namespace {
// Dataset yielding [1, 2, 2] FLOAT32 samples with values {idx*4, ..., idx*4+3}.
class ChannelDataset : public Dataset {
public:
    explicit ChannelDataset(size_t size) : size_(size) {}

    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> operator[](size_t idx) const override {
        auto data = std::make_shared<Tensor>(std::vector<int64_t>{1, 2, 2}, DataType::kFLOAT32);
        auto label = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kINT64);
        auto *data_ptr = static_cast<float *>(data->DataPtr());
        for (int i = 0; i < 4; ++i) { data_ptr[i] = static_cast<float>(idx * 4 + i); }
        *static_cast<int64_t *>(label->DataPtr()) = static_cast<int64_t>(idx);
        return {data, label};
    }

    size_t Size() const override { return size_; }

private:
    size_t size_ = 0;
};

// Dataset whose sample 1 has a different shape ([1, 2, 3]) than sample 0 ([1, 2, 2]).
class RaggedDataset : public Dataset {
public:
    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> operator[](size_t idx) const override {
        auto data = std::make_shared<Tensor>(idx == 0 ? std::vector<int64_t>{1, 2, 2} : std::vector<int64_t>{1, 2, 3},
                                             DataType::kFLOAT32);
        auto label = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kINT64);
        return {data, label};
    }

    size_t Size() const override { return 2; }
};
} // namespace

TEST(DataLoaderStackTest, PreservesSampleDims) {
    DataLoader loader(std::make_shared<ChannelDataset>(2), 2);
    auto it = loader.begin();
    const auto &[x, y] = *it;
    EXPECT_EQ(x->Dims(), (std::vector<int64_t>{2, 1, 2, 2}));
    EXPECT_EQ(x->Dtype(), DataType::kFLOAT32);
    const auto *data = static_cast<const float *>(x->DataPtr());
    for (int i = 0; i < 8; ++i) { EXPECT_FLOAT_EQ(data[i], static_cast<float>(i)); }
    EXPECT_EQ(y->Dims(), (std::vector<int64_t>{2, 1}));
}

TEST(DataLoaderStackTest, MismatchedShapesFail) {
    DataLoader loader(std::make_shared<RaggedDataset>(), 2);
    auto it = loader.begin();
    EXPECT_DEATH({ const auto batch = *it; }, "different shapes");
}
