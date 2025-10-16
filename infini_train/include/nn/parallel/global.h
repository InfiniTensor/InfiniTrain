#pragma once

#include <mutex>

namespace infini_train::nn::parallel::global {

class GlobalEnv {
public:
    static GlobalEnv &Instance();

    void Init(int threads_per_process, int tensor_parallel_size);

    int world_size() const;

    int global_proc_rank() const;

    int local_proc_rank() const;

    int nproc_per_node() const;

    int nthread_per_process() const;

    int tensor_parallel_size() const;

    int data_parallel_size() const;

private:
    GlobalEnv() = default;
    ~GlobalEnv() = default;

    GlobalEnv(const GlobalEnv &) = delete;
    GlobalEnv &operator=(const GlobalEnv &) = delete;

private:
    int world_size_ = 1;
    int nproc_per_node_ = 1;
    int nthread_per_process_ = 1;
    int global_proc_rank_ = 0;
    int local_proc_rank_ = 0;

    int tensor_parallel_size_ = 1;
    int data_parallel_size_ = 1;

    mutable std::mutex mutex_;
    bool initialized_ = false;
};

inline void InitAllEnv(int nthread_per_process, int tensor_parallel_size) {
    GlobalEnv::Instance().Init(nthread_per_process, tensor_parallel_size);
}

inline int GetWorldSize() { return GlobalEnv::Instance().world_size(); }
inline int GetNprocPerNode() { return GlobalEnv::Instance().nproc_per_node(); }
inline int GetNthreadPerProc() { return GlobalEnv::Instance().nthread_per_process(); }
inline int GetGlobalProcRank() { return GlobalEnv::Instance().global_proc_rank(); }
inline int GetLocalProcRank() { return GlobalEnv::Instance().local_proc_rank(); }

inline int GetTensorParallelSize() { return GlobalEnv::Instance().tensor_parallel_size(); }
inline int GetDataParallelSize() { return GlobalEnv::Instance().data_parallel_size(); }

} // namespace infini_train::nn::parallel::global
