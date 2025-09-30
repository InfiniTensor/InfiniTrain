#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "infini_train/include/dataset.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
class DataLoaderIterator {
public:
    DataLoaderIterator(const Dataset &dataset, size_t batch_size, size_t batch_idx, size_t max_batch_idx,
                       size_t ddp_rank = 0, size_t ddp_world_size = 1);

    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> operator*() const;

    DataLoaderIterator &operator++();
    DataLoaderIterator operator++(int);

    friend bool operator<(const DataLoaderIterator &lhs, const DataLoaderIterator &rhs);
    friend bool operator!=(const DataLoaderIterator &lhs, const DataLoaderIterator &rhs);
    friend bool operator==(const DataLoaderIterator &lhs, const DataLoaderIterator &rhs);

private:
    const Dataset *dataset_ = nullptr; // not owned
    size_t batch_size_ = 0;
    size_t batch_idx_ = 0;
    size_t max_batch_idx_ = 0;
    size_t ddp_rank_ = 0;
    size_t ddp_world_size_ = 1;
};

class DataLoader {
public:
    DataLoader(const std::shared_ptr<Dataset> &dataset, size_t batch_size);

    virtual DataLoaderIterator begin() const;
    virtual DataLoaderIterator end() const;

protected:
    std::shared_ptr<Dataset> dataset_;
    size_t batch_size_ = 0;
    size_t max_batch_idx_ = 0;
};

class DistributedDataLoader : public DataLoader {
public:
    DistributedDataLoader(const std::shared_ptr<Dataset> &dataset, size_t batch_size, size_t ddp_rank,
                          size_t ddp_world_size);

    DataLoaderIterator begin() const override;
    DataLoaderIterator end() const override;

private:
    size_t ddp_rank_ = 0;
    size_t ddp_world_size_ = 1;
};
} // namespace infini_train
