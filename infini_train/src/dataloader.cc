#include "infini_train/include/dataloader.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <numeric>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/dataset.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
namespace {
size_t CheckedCeilDiv(size_t numerator, size_t denominator) {
    CHECK_GT(denominator, 0);
    return (numerator + denominator - 1) / denominator;
}

// TODO(dcj): Use official stack implementation later.
std::shared_ptr<Tensor> Stack(const std::vector<std::shared_ptr<Tensor>> &tensors) {
    CHECK(!tensors.empty()) << "Cannot stack an empty batch. Check DataLoader iterator end handling.";
    const int batch_size = tensors.size();
    const auto &dims = tensors[0]->Dims();
    const int stacked_dim = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>());
    auto stacked_tensor = std::make_shared<Tensor>(std::vector<int64_t>{batch_size, stacked_dim}, tensors[0]->Dtype());
    for (const auto &tensor : tensors) {
        CHECK_EQ(static_cast<int>(tensors[0]->Dtype()), static_cast<int>(tensor->Dtype()));
        const auto &dims = tensor->Dims();
        CHECK_EQ(stacked_dim, std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int64_t>()));
    }

    size_t offset = 0;
    for (const auto &tensor : tensors) {
        memcpy(reinterpret_cast<uint8_t *>(stacked_tensor->DataPtr()) + offset, tensor->DataPtr(),
               tensor->SizeInBytes());
        offset += tensor->SizeInBytes();
    }
    return stacked_tensor;
}
} // namespace

DataLoaderIterator::DataLoaderIterator(const Dataset &dataset, size_t batch_size, size_t dataloader_step,
                                       size_t num_dataloader_steps, size_t ddp_rank, size_t ddp_world_size)
    : dataset_(&dataset), batch_size_(batch_size), dataloader_step_(dataloader_step),
      num_dataloader_steps_(num_dataloader_steps), ddp_rank_(ddp_rank), ddp_world_size_(ddp_world_size){};

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DataLoaderIterator::operator*() const {
    /*
      0,         1,            ..., x,                  ...
      [0, bs-1], [bs, 2*bs-1], ..., [x*bs, (x+1)*bs-1], ...
                                    ^
                                    dataloader_step
    */
    std::vector<std::shared_ptr<Tensor>> data_vec;
    std::vector<std::shared_ptr<Tensor>> label_vec;
    CHECK_LT(dataloader_step_, num_dataloader_steps_)
        << "Cannot dereference DataLoader end iterator. dataloader_step=" << dataloader_step_
        << ", num_dataloader_steps=" << num_dataloader_steps_ << ", ddp_rank=" << ddp_rank_
        << ", ddp_world_size=" << ddp_world_size_;
    const size_t start_idx = (dataloader_step_ * ddp_world_size_ + ddp_rank_) * batch_size_;
    CHECK_LT(start_idx, dataset_->Size())
        << "DataLoader batch starts past dataset end. dataloader_step=" << dataloader_step_
        << ", start_idx=" << start_idx << ", dataset_size=" << dataset_->Size() << ", batch_size=" << batch_size_
        << ", ddp_rank=" << ddp_rank_ << ", ddp_world_size=" << ddp_world_size_;
    const size_t end_idx = std::min(start_idx + batch_size_, dataset_->Size());
    for (size_t idx = start_idx; idx < end_idx; ++idx) {
        auto &&[data, label] = dataset_->operator[](idx);
        data_vec.push_back(std::move(data));
        label_vec.push_back(std::move(label));
    }
    return {Stack(std::move(data_vec)), Stack(std::move(label_vec))};
}

DataLoaderIterator &DataLoaderIterator::operator++() {
    dataloader_step_ = std::min(dataloader_step_ + 1, num_dataloader_steps_);
    return *this;
}

DataLoaderIterator DataLoaderIterator::operator++(int) {
    DataLoaderIterator tmp(*this);
    ++(*this);
    return tmp;
}

bool operator<(const DataLoaderIterator &lhs, const DataLoaderIterator &rhs) {
    return lhs.dataloader_step_ < rhs.dataloader_step_;
}

bool operator!=(const DataLoaderIterator &lhs, const DataLoaderIterator &rhs) {
    return lhs.dataloader_step_ != rhs.dataloader_step_;
}

bool operator==(const DataLoaderIterator &lhs, const DataLoaderIterator &rhs) {
    return lhs.dataloader_step_ == rhs.dataloader_step_;
}

size_t DataLoaderIterator::DataLoaderStep() const { return dataloader_step_; }

DataLoaderIterator &DataLoaderIterator::SeekDataLoaderStep(size_t dataloader_step) {
    CHECK_LE(dataloader_step, num_dataloader_steps_)
        << "Cannot seek past DataLoader end. dataloader_step=" << dataloader_step
        << ", num_dataloader_steps=" << num_dataloader_steps_;
    dataloader_step_ = dataloader_step;
    return *this;
}

DataLoader::DataLoader(const std::shared_ptr<Dataset> &dataset, size_t batch_size)
    : dataset_(dataset), batch_size_(batch_size) {
    CHECK(dataset_ != nullptr) << "DataLoader dataset must not be null";
    CHECK_GT(batch_size_, 0) << "DataLoader batch_size must be greater than zero";
    num_dataloader_steps_ = CheckedCeilDiv(dataset_->Size(), batch_size_);
}

DataLoaderIterator DataLoader::begin() const {
    return DataLoaderIterator(*dataset_, batch_size_, 0, num_dataloader_steps_, 0, 1);
}

DataLoaderIterator DataLoader::end() const {
    return DataLoaderIterator(*dataset_, batch_size_, num_dataloader_steps_, num_dataloader_steps_, 0, 1);
}

size_t DataLoader::NumDataLoaderSteps() const { return num_dataloader_steps_; }

DistributedDataLoader::DistributedDataLoader(const std::shared_ptr<Dataset> &dataset, size_t batch_size,
                                             size_t ddp_rank, size_t ddp_world_size)
    : DataLoader(dataset, batch_size), ddp_rank_(ddp_rank), ddp_world_size_(ddp_world_size) {
    CHECK_GT(ddp_world_size_, 0);
    CHECK_LT(ddp_rank_, ddp_world_size_);
    const size_t samples_per_dataloader_step = ddp_world_size_ * batch_size_;
    CHECK_GE(dataset_->Size(), samples_per_dataloader_step)
        << "DistributedDataLoader needs enough samples for one DataLoader step. dataset_size=" << dataset_->Size()
        << ", samples_per_dataloader_step=" << samples_per_dataloader_step << " (" << batch_size_ << " per rank * "
        << ddp_world_size_ << " ranks). Reduce batch size/world size or use a larger dataset.";
    num_dataloader_steps_ = dataset_->Size() / samples_per_dataloader_step;
}

DataLoaderIterator DistributedDataLoader::begin() const {
    return DataLoaderIterator(*dataset_, batch_size_, 0, num_dataloader_steps_, ddp_rank_, ddp_world_size_);
}

DataLoaderIterator DistributedDataLoader::end() const {
    return DataLoaderIterator(*dataset_, batch_size_, num_dataloader_steps_, num_dataloader_steps_, ddp_rank_,
                              ddp_world_size_);
}
} // namespace infini_train
