#pragma once

#include <mutex>
#include <string>
#include <vector>

namespace infini_train::nn::parallel::global {

enum Axis : uint8_t { DP = 0, TP = 1, PP = 2, AXIS_COUNT = 3 };

struct Layout {
    int sizes[AXIS_COUNT]{1, 1, 1};
    Axis order[AXIS_COUNT]{DP, TP, PP};
    int strides[AXIS_COUNT]{1, 1, 1};

    inline void InitStrides() {
        // Calculate strides
        int stride = 1;
        for (int i = AXIS_COUNT - 1; i >= 0; --i) {
            const Axis ax = order[i];
            strides[ax] = stride;
            stride *= sizes[ax];
        }
    }

    inline int RankOf(int dp, int tp, int pp) const {
        // Return the thread rank given layout coords
        const int coord[AXIS_COUNT] = {dp, tp, pp};
        int r = 0;
        for (int i = 0; i < AXIS_COUNT; ++i) {
            const Axis ax = static_cast<Axis>(i);
            r += coord[ax] * strides[ax];
        }
        return r;
    }

    inline void CoordOf(int rank, int &dp, int &tp, int &pp) const {
        // Return the layout coords given thread rank
        dp = (rank / strides[DP]) % sizes[DP];
        tp = (rank / strides[TP]) % sizes[TP];
        pp = (rank / strides[PP]) % sizes[PP];
    }

    inline int GroupId(Axis target, int dp, int tp, int pp) const {
        // Return the parallel ProcessGroup ID where the rank is in
        int id = 0;
        int mult = 1;
        for (int i = AXIS_COUNT - 1; i >= 0; --i) {
            Axis ax = order[i];
            if (ax == target) {
                continue;
            }
            int c = (ax == DP ? dp : (ax == TP ? tp : pp));
            id += c * mult;
            mult *= sizes[ax];
        }
        return id;
    }

    inline std::vector<int> GroupRanks(Axis target, int fixed_dp, int fixed_tp, int fixed_pp) const {
        // Return all the ranks within the same parallel ProcessGroup
        std::vector<int> ranks;
        ranks.reserve(sizes[target]);
        int dp = fixed_dp, tp = fixed_tp, pp = fixed_pp;
        for (int v = 0; v < sizes[target]; ++v) {
            if (target == DP) {
                dp = v;
            } else if (target == TP) {
                tp = v;
            } else {
                pp = v;
            }
            ranks.push_back(RankOf(dp, tp, pp));
        }
        return ranks;
    }
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
inline int GetGroupId(Axis target, int thread_rank) {
    int dp, tp, pp;
    GetCoordOf(thread_rank, dp, tp, pp);
    return GlobalEnv::Instance().layout().GroupId(target, dp, tp, pp);
}
inline std::vector<int> GetGroupRanks(Axis target, int dp, int tp, int pp) {
    return GlobalEnv::Instance().layout().GroupRanks(target, dp, tp, pp);
}
inline std::vector<int> GetGroupRanks(Axis target, int thread_rank) {
    int dp, tp, pp;
    GetCoordOf(thread_rank, dp, tp, pp);
    return GlobalEnv::Instance().layout().GroupRanks(target, dp, tp, pp);
}

std::string ProcessGroupOverview(const Layout &L = GlobalEnv::Instance().layout(), bool skip_trivial_axes = true);

} // namespace infini_train::nn::parallel::global
