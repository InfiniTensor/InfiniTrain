#include "infini_train/include/nn/parallel/rank.h"

namespace infini_train::nn::parallel {
Rank::Rank(int process_rank, int thread_rank, int process_size, int thread_size)
    : process_rank_(process_rank), thread_rank_(thread_rank), process_size_(process_size), thread_size_(thread_size) {}

int Rank::process_rank() const { return process_rank_; }
int Rank::thread_rank() const { return thread_rank_; }

int Rank::process_size() const { return process_size_; }
int Rank::thread_size() const { return thread_size_; }

int Rank::GlobalRank() const { return process_rank_ * thread_size_ + thread_rank_; }

bool Rank::IsParallel() const { return thread_size_ * process_size_ > 1; }
bool Rank::IsMainRank() const { return GlobalRank() == 0; }
} // namespace infini_train::nn::parallel
