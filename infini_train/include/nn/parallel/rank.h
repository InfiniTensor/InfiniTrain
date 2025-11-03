#pragma once

namespace infini_train::nn::parallel {
class Rank {
public:
    Rank(int process_rank, int thread_rank, int process_size, int thread_size);

    int process_rank() const;
    int thread_rank() const;
    int process_size() const;
    int thread_size() const;

    int GlobalRank() const;

    bool IsParallel() const;

    bool IsMainRank() const;

private:
    const int process_rank_ = 0; // Rank of the current process within the node
    const int thread_rank_ = 0;  // Rank of the current thread within the process
    const int process_size_ = 1; // Total number of processes on this node
    const int thread_size_ = 1;  // Total number of threads in the current process
};
} // namespace infini_train::nn::parallel
