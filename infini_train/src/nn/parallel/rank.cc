#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/global.h"

namespace infini_train::nn::parallel {
Rank::Rank(int global_process_rank, int thread_rank, int processes_per_node, int threads_per_process)
    : global_process_rank_(global_process_rank), thread_rank_(thread_rank), processes_per_node_(processes_per_node),
      threads_per_process_(threads_per_process) {}

int Rank::process_rank() const { return global_process_rank_; }
int Rank::thread_rank() const { return thread_rank_; }

int Rank::process_size() const { return processes_per_node_; }
int Rank::thread_size() const { return threads_per_process_; }

int Rank::GlobalRank() const { return global_process_rank_ * threads_per_process_ + thread_rank_; }

bool Rank::IsParallel() const { return global::GetWorldSize() > 1; }

bool Rank::IsMainRank() const { return GlobalRank() == 0; }

bool Rank::IsLastRank() const { return GlobalRank() == global::GetWorldSize() - 1; }
} // namespace infini_train::nn::parallel
