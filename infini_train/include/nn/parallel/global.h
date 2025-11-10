#pragma once

#include <mutex>
#include <string>
#include <vector>

namespace infini_train::nn::parallel::global {

enum Axis : uint8_t { DP = 0, TP = 1, PP = 2, AXIS_COUNT = 3 };

struct Layout {
    int sizes[AXIS_COUNT]{1, 1, 1};
    // Default order according to Megatron-LM is TP-DP-PP. Ref:
    // https://github.com/NVIDIA/Megatron-LM/blob/e07c4a4450b6faa187a1ef4ec082a35ad7d2f085/megatron/core/parallel_state.py#L618
    Axis order[AXIS_COUNT]{TP, DP, PP};
    int strides[AXIS_COUNT]{1, 1, 1};

    void InitStrides();
    int RankOf(int dp, int tp, int pp) const;
    void CoordOf(int rank, int &dp, int &tp, int &pp) const;
    int GroupId(Axis target, int dp, int tp, int pp) const;
    std::vector<int> GroupRanks(Axis target, int fixed_dp, int fixed_tp, int fixed_pp) const;
};

class GlobalEnv {
public:
    static GlobalEnv &Instance();

    void Init(int threads_per_process, int tensor_parallel_size, bool sequence_parallel_enabled = false);

    int world_size() const;

    int global_proc_rank() const;

    int local_proc_rank() const;

    int nproc_per_node() const;

    int nthread_per_process() const;

    int tensor_parallel_size() const;

    bool sequence_parallel_enabled() const;

    int data_parallel_size() const;

    Layout layout() const;

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
    bool sequence_parallel_enabled_ = false;

    int data_parallel_size_ = 1;

    mutable std::mutex mutex_;
    bool initialized_ = false;

    Layout layout_;
};

inline void InitAllEnv(int nthread_per_process, int tensor_parallel_size, bool sequence_parallel_enabled = false) {
    GlobalEnv::Instance().Init(nthread_per_process, tensor_parallel_size, sequence_parallel_enabled);
}

inline int GetWorldSize() { return GlobalEnv::Instance().world_size(); }
inline int GetNprocPerNode() { return GlobalEnv::Instance().nproc_per_node(); }
inline int GetNthreadPerProc() { return GlobalEnv::Instance().nthread_per_process(); }
inline int GetGlobalProcRank() { return GlobalEnv::Instance().global_proc_rank(); }
inline int GetLocalProcRank() { return GlobalEnv::Instance().local_proc_rank(); }

inline int GetTensorParallelSize() { return GlobalEnv::Instance().tensor_parallel_size(); }
inline bool GetSequenceParallelEnabled() { return GlobalEnv::Instance().sequence_parallel_enabled(); }
inline int GetDataParallelSize() { return GlobalEnv::Instance().data_parallel_size(); }

// Layout Helper Functions
inline int GetRankOf(int dp, int tp, int pp) { return GlobalEnv::Instance().layout().RankOf(dp, tp, pp); }
inline void GetCoordOf(int rank, int &dp, int &tp, int &pp) {
    return GlobalEnv::Instance().layout().CoordOf(rank, dp, tp, pp);
}
inline int GetGroupId(Axis target, int dp, int tp, int pp) {
    return GlobalEnv::Instance().layout().GroupId(target, dp, tp, pp);
}
inline int GetGroupId(Axis target, int rank) {
    int dp, tp, pp;
    GetCoordOf(rank, dp, tp, pp);
    return GlobalEnv::Instance().layout().GroupId(target, dp, tp, pp);
}
inline std::vector<int> GetGroupRanks(Axis target, int dp, int tp, int pp) {
    return GlobalEnv::Instance().layout().GroupRanks(target, dp, tp, pp);
}
inline std::vector<int> GetGroupRanks(Axis target, int rank) {
    int dp, tp, pp;
    GetCoordOf(rank, dp, tp, pp);
    return GlobalEnv::Instance().layout().GroupRanks(target, dp, tp, pp);
}

std::string ProcessGroupOverview(const Layout &L = GlobalEnv::Instance().layout(), bool skip_trivial_axes = true);

} // namespace infini_train::nn::parallel::global
