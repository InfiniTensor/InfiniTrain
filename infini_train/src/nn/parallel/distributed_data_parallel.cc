#include "infini_train/include/nn/parallel/distributed_data_parallel.h"

#include <format>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/tensor.h"

#include "infini_train/include/nn/parallel_functional.h"

namespace infini_train::nn::parallel {

DistributedDataParallel::Rank::Rank(int process_rank, int thread_rank, int process_size, int thread_size)
    : process_rank_(process_rank), thread_rank_(thread_rank), process_size_(process_size), thread_size_(thread_size) {}

int DistributedDataParallel::Rank::process_rank() const { return process_rank_; }
int DistributedDataParallel::Rank::thread_rank() const { return thread_rank_; }
int DistributedDataParallel::Rank::process_size() const { return process_size_; }
int DistributedDataParallel::Rank::thread_size() const { return thread_size_; }

int DistributedDataParallel::Rank::WorldSize() const { return process_size_ * thread_size_; }

bool DistributedDataParallel::Rank::IsDDP() const { return process_size_ * thread_size_ > 1; }

bool DistributedDataParallel::Rank::IsMainRank() const { return thread_rank_ == 0; }
} // namespace infini_train::nn::parallel
