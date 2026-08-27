#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "infini_train/include/dataloader.h"
#include "infini_train/include/dataset.h"
#include "infini_train/include/tensor.h"

using namespace infini_train;

static_assert(!std::is_constructible_v<DataLoaderIterator, const Dataset &, size_t, size_t, size_t, size_t, size_t>);

namespace {
class IndexDataset : public Dataset {
public:
    explicit IndexDataset(size_t size) : size_(size) {}

    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> operator[](size_t idx) const override {
        auto data = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kINT64);
        auto label = std::make_shared<Tensor>(std::vector<int64_t>{1}, DataType::kINT64);
        *static_cast<int64_t *>(data->DataPtr()) = static_cast<int64_t>(idx);
        *static_cast<int64_t *>(label->DataPtr()) = static_cast<int64_t>(idx + 1000);
        return {data, label};
    }

    size_t Size() const override { return size_; }

private:
    size_t size_ = 0;
};

std::vector<int64_t> TensorValues(const std::shared_ptr<Tensor> &tensor) {
    const auto *data = static_cast<const int64_t *>(tensor->DataPtr());
    return std::vector<int64_t>(data, data + tensor->NumElements());
}
} // namespace

TEST(DataLoaderTest, RejectsInvalidConstructorArguments) {
    EXPECT_DEATH({ DataLoader loader(nullptr, 2); }, "dataset must not be null");
    EXPECT_DEATH({ DataLoader loader(std::make_shared<IndexDataset>(5), 0); }, "batch_size must be greater than zero");
}

TEST(DataLoaderTest, RegularDataLoaderKeepsPartialLastBatch) {
    DataLoader loader(std::make_shared<IndexDataset>(5), 2);

    std::vector<std::vector<int64_t>> batches;
    for (const auto &[x, y] : loader) { batches.push_back(TensorValues(x)); }

    ASSERT_EQ(batches.size(), 3);
    EXPECT_EQ(batches[0], (std::vector<int64_t>{0, 1}));
    EXPECT_EQ(batches[1], (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(batches[2], (std::vector<int64_t>{4}));

    auto iter = loader.begin();
    iter.SeekDataLoaderStep(2);
    EXPECT_EQ(TensorValues((*iter).first), (std::vector<int64_t>{4}));
}

TEST(DataLoaderTest, DistributedDataLoaderPartitionsEachStepByRank) {
    const auto dataset = std::make_shared<IndexDataset>(13);
    const size_t batch_size = 2;
    const size_t world_size = 3;

    DistributedDataLoader rank0(dataset, batch_size, 0, world_size);
    DistributedDataLoader rank1(dataset, batch_size, 1, world_size);
    DistributedDataLoader rank2(dataset, batch_size, 2, world_size);

    auto r0 = rank0.begin();
    auto r1 = rank1.begin();
    auto r2 = rank2.begin();

    EXPECT_EQ(TensorValues((*r0).first), (std::vector<int64_t>{0, 1}));
    EXPECT_EQ(TensorValues((*r1).first), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(TensorValues((*r2).first), (std::vector<int64_t>{4, 5}));

    ++r0;
    ++r1;
    ++r2;

    EXPECT_EQ(TensorValues((*r0).first), (std::vector<int64_t>{6, 7}));
    EXPECT_EQ(TensorValues((*r1).first), (std::vector<int64_t>{8, 9}));
    EXPECT_EQ(TensorValues((*r2).first), (std::vector<int64_t>{10, 11}));

    ++r0;
    ++r1;
    ++r2;

    EXPECT_EQ(r0, rank0.end());
    EXPECT_EQ(r1, rank1.end());
    EXPECT_EQ(r2, rank2.end());
}

TEST(DataLoaderTest, SeekDataLoaderStepSupportsResumeAndEnd) {
    DistributedDataLoader loader(std::make_shared<IndexDataset>(13), 2, 2, 3);
    auto iter = loader.begin();

    EXPECT_EQ(loader.NumDataLoaderSteps(), 2);

    iter.SeekDataLoaderStep(1);
    EXPECT_EQ(iter.DataLoaderStep(), 1);
    EXPECT_EQ(TensorValues((*iter).first), (std::vector<int64_t>{10, 11}));

    iter.SeekDataLoaderStep(loader.NumDataLoaderSteps());
    EXPECT_EQ(iter, loader.end());
    EXPECT_DEATH(iter.SeekDataLoaderStep(loader.NumDataLoaderSteps() + 1), "Cannot seek past DataLoader end");

    size_t consumed_dataloader_steps = 5;
    iter = loader.begin();
    iter.SeekDataLoaderStep(consumed_dataloader_steps % loader.NumDataLoaderSteps());
    auto next_batch = [&]() {
        auto batch = *iter;
        ++iter;
        if (iter == loader.end()) {
            iter = loader.begin();
        }
        ++consumed_dataloader_steps;
        return batch;
    };

    EXPECT_EQ(TensorValues(next_batch().first), (std::vector<int64_t>{10, 11}));
    EXPECT_EQ(iter.DataLoaderStep(), 0);
    EXPECT_EQ(TensorValues(next_batch().first), (std::vector<int64_t>{4, 5}));
    EXPECT_EQ(consumed_dataloader_steps, 7);
}
