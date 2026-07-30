#pragma once

namespace infini_train::nn::parallel {
class Rank {
public:
    Rank(int global_process_rank, int thread_rank, int processes_per_node, int threads_per_process);

    int process_rank() const;
    int thread_rank() const;
    int process_size() const;
    int thread_size() const;

    int GlobalRank() const;

    bool IsParallel() const;

    bool IsMainRank() const;

    bool IsLastRank() const;

private:
    const int global_process_rank_ = 0; // Global rank of the current process
    const int thread_rank_ = 0;         // Rank of the current thread within the process
    const int processes_per_node_ = 1;  // Number of processes on each node
    const int threads_per_process_ = 1; // Number of threads in each process
};
} // namespace infini_train::nn::parallel
