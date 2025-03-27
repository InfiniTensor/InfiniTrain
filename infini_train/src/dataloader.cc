#include "infini_train/include/dataloader.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/dataset.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
namespace {
// TODO(dcj): Use official stack implementation later.
std::shared_ptr<Tensor> Stack(const std::vector<std::shared_ptr<Tensor>> &tensors) {
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

DataLoaderIterator::DataLoaderIterator(const Dataset &dataset, size_t batch_size, size_t batch_idx,
                                       size_t max_batch_idx)
    : dataset_(&dataset), batch_size_(batch_size), batch_idx_(batch_idx), max_batch_idx_(max_batch_idx){};

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> DataLoaderIterator::operator*() const {
    /*
      0,         1,            ..., x,                  ...
      [0, bs-1], [bs, 2*bs-1], ..., [x*bs, (x+1)*bs-1], ...
                                    ^
                                    batch_idx
    */
    std::vector<std::shared_ptr<Tensor>> data_vec;
    std::vector<std::shared_ptr<Tensor>> label_vec;
    for (int idx = batch_idx_ * batch_size_; idx < (batch_idx_ + 1) * batch_size_ && idx < dataset_->Size(); ++idx) {
        auto &&[data, label] = dataset_->operator[](idx);
        data_vec.push_back(std::move(data));
        label_vec.push_back(std::move(label));
    }
    return {Stack(std::move(data_vec)), Stack(std::move(label_vec))};
}

DataLoaderIterator &DataLoaderIterator::operator++() {
    batch_idx_ = std::min(batch_idx_ + 1, max_batch_idx_);
    return *this;
}

DataLoaderIterator DataLoaderIterator::operator++(int) {
    DataLoaderIterator tmp(*this);
    ++(*this);
    return tmp;
}

bool operator<(const DataLoaderIterator &lhs, const DataLoaderIterator &rhs) { return lhs.batch_idx_ < rhs.batch_idx_; }

bool operator!=(const DataLoaderIterator &lhs, const DataLoaderIterator &rhs) {
    return lhs.batch_idx_ != rhs.batch_idx_;
}

bool operator==(const DataLoaderIterator &lhs, const DataLoaderIterator &rhs) {
    return lhs.batch_idx_ == rhs.batch_idx_;
}

DataLoader::DataLoader(const std::shared_ptr<Dataset> &dataset, size_t batch_size)
    : dataset_(dataset), batch_size_(batch_size), max_batch_idx_((dataset_->Size() + batch_size_ - 1) / batch_size_) {}

DataLoaderIterator DataLoader::begin() const { return DataLoaderIterator(*dataset_, batch_size_, 0, max_batch_idx_); }

DataLoaderIterator DataLoader::end() const {
    return DataLoaderIterator(*dataset_, batch_size_, max_batch_idx_, max_batch_idx_);
}
} // namespace infini_train
