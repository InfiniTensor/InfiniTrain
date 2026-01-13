#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "infini_train/include/utils/precision_check_config.h"

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

    void Init(int threads_per_process, int tensor_parallel_size, bool sequence_parallel_enabled,
              int pipeline_parallel_size, int virtual_pipeline_parallel_size,
              const utils::PrecisionCheckConfig& precision_config = utils::PrecisionCheckConfig());

    enum class PrecisionCheckLevel { NONE, MODULE, FUNCTION };
    PrecisionCheckLevel GetPrecisionCheckLevel() const;
    const utils::PrecisionCheckConfig& GetPrecisionCheckConfig() const;

    int nnodes() const;

    int nproc_per_node() const;

    int world_size() const;

    int global_proc_rank() const;

    int local_proc_rank() const;

    int nthread_per_process() const;

    int tensor_parallel_size() const;

    int sequence_parallel_size() const;

    bool sequence_parallel_enabled() const;

    int data_parallel_size() const;

    int pipeline_parallel_size() const;

    int virtual_pipeline_parallel_size() const;

    Layout layout() const;

private:
    GlobalEnv() = default;
    ~GlobalEnv() = default;

    GlobalEnv(const GlobalEnv &) = delete;
    GlobalEnv &operator=(const GlobalEnv &) = delete;

private:
    int nnodes_ = 1;
    int nproc_per_node_ = 1;
    int nthread_per_process_ = 1;
    int world_size_ = 1;

    int global_proc_rank_ = 0;
    int local_proc_rank_ = 0;

    int tensor_parallel_size_ = 1;
    bool sequence_parallel_enabled_ = false;

    int data_parallel_size_ = 1;

    int pipeline_parallel_size_ = 1;
    int virtual_pipeline_parallel_size_ = 1;

    mutable std::mutex mutex_;
    bool initialized_ = false;

    Layout layout_;
    PrecisionCheckLevel precision_check_level_ = PrecisionCheckLevel::NONE;
    utils::PrecisionCheckConfig precision_check_config_;
};

inline void InitAllEnv(int nthread_per_process, int tensor_parallel_size, bool sequence_parallel_enabled,
                       int pipeline_parallel_size, int virtual_pipeline_parallel,
                       const utils::PrecisionCheckConfig& precision_config = utils::PrecisionCheckConfig()) {
    GlobalEnv::Instance().Init(nthread_per_process, tensor_parallel_size, sequence_parallel_enabled,
                               pipeline_parallel_size, virtual_pipeline_parallel, precision_config);
}
inline int GetNnodes() { return GlobalEnv::Instance().nnodes(); }
inline int GetWorldSize() { return GlobalEnv::Instance().world_size(); }
inline int GetNprocPerNode() { return GlobalEnv::Instance().nproc_per_node(); }
inline int GetNthreadPerProc() { return GlobalEnv::Instance().nthread_per_process(); }
inline int GetGlobalProcRank() { return GlobalEnv::Instance().global_proc_rank(); }
inline int GetLocalProcRank() { return GlobalEnv::Instance().local_proc_rank(); }

inline int GetTensorParallelSize() { return GlobalEnv::Instance().tensor_parallel_size(); }
inline int GetSequenceParallelSize() { return GlobalEnv::Instance().sequence_parallel_size(); }
inline bool GetSequenceParallelEnabled() { return GlobalEnv::Instance().sequence_parallel_enabled(); }
inline int GetDataParallelSize() { return GlobalEnv::Instance().data_parallel_size(); }
inline int GetPipelineParallelSize() { return GlobalEnv::Instance().pipeline_parallel_size(); }
inline int GetVirtualPipelineParallelSize() { return GlobalEnv::Instance().virtual_pipeline_parallel_size(); }

// =========================
// Layout Helper Functions
// =========================

/**
 * @brief Get the global rank corresponding to the given (dp, tp, pp) coordinate.
 */
inline int GetRankOf(int dp, int tp, int pp) { return GlobalEnv::Instance().layout().RankOf(dp, tp, pp); }
/**
 * @brief Get the (dp, tp, pp) coordinate corresponding to the given global rank.
 */
inline void GetCoordOf(int rank, int &dp, int &tp, int &pp) {
    return GlobalEnv::Instance().layout().CoordOf(rank, dp, tp, pp);
}

/**
 * @brief Get the group ID that the (dp, tp, pp) coordinate belongs to along a given parallel axis.
 */
inline int GetGroupId(Axis target, int dp, int tp, int pp) {
    return GlobalEnv::Instance().layout().GroupId(target, dp, tp, pp);
}
/**
 * @brief Get the group ID that a given rank belongs to along a specific parallel axis.
 */
inline int GetGroupId(Axis target, int rank) {
    int dp, tp, pp;
    GetCoordOf(rank, dp, tp, pp);
    return GlobalEnv::Instance().layout().GroupId(target, dp, tp, pp);
}

/**
 * @brief Get all ranks that belong to the same group as the given (dp, tp, pp) coordinate
 *        along a specified parallel axis (e.g., all ranks in the same TP group).
 */
inline std::vector<int> GetGroupRanks(Axis target, int dp, int tp, int pp) {
    return GlobalEnv::Instance().layout().GroupRanks(target, dp, tp, pp);
}

/**
 * @brief Get all ranks that belong to the same group as the given rank
 *        along a specified parallel axis (e.g., all ranks in the same DP group).
 */
inline std::vector<int> GetGroupRanks(Axis target, int rank) {
    int dp, tp, pp;
    GetCoordOf(rank, dp, tp, pp);
    return GlobalEnv::Instance().layout().GroupRanks(target, dp, tp, pp);
}

/**
 * @brief Generate a human-readable overview of all parallel communication groups.
 *
 * The output is intended for debugging, logging, and runtime verification of
 * distributed parallelism configuration.
 *
 * @param L  The Layout describing DP / TP / PP sizes and axis ordering.
 * @param skip_trivial_axes
 *        If true, axes whose size <= 1(i.e. parallel strategy that is not enabled)
 *        will be marked as "unenabled" and their detailed group listing will be skipped.
 *
 * @return A formatted string containing the full overview of process groups.
 *
 *         Example:
 *           === Parallel Communication Groups ===
 *           world_size = 8, config: {DP=2, TP=4, PP=1}, order: {TP -> DP -> PP}
 *           [DP] size=2, num_groups=4
 *           - DP 0 (dp=-, tp=0, pp=0): [0, 4]
 *           - DP 1 (dp=-, tp=1, pp=0): [1, 5]
 *           - DP 2 (dp=-, tp=2, pp=0): [2, 6]
 *           - DP 3 (dp=-, tp=3, pp=0): [3, 7]
 *
 *           [TP] size=4, num_groups=2
 *           - TP 0 (dp=0, tp=-, pp=0): [0, 1, 2, 3]
 *           - TP 1 (dp=1, tp=-, pp=0): [4, 5, 6, 7]
 *
 *           [PP] size=1, unenabled
 */
std::string ProcessGroupOverview(const Layout &L = GlobalEnv::Instance().layout(), bool skip_trivial_axes = true);

} // namespace infini_train::nn::parallel::global
