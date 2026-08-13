#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace infini_train::nn::parallel::global {

extern thread_local int thread_global_rank;

// TP and DP are view-local axes. In the expert rank generator they represent
// expert tensor parallelism (ETP) and expert data parallelism (EDP), matching
// Megatron-LM's RankGenerator semantics.
enum Axis : uint8_t { DP = 0, TP = 1, PP = 2, EP = 3, AXIS_COUNT = 4 };

class RankGenerator {
public:
    // Default order according to Megatron-LM is tp-cp-ep-dp-pp. Ref:
    // https://github.com/NVIDIA/Megatron-LM/blob/cf2f07d7b1315c96c05554c670c43207c6783e5e/megatron/core/parallel_state.py#L561
    explicit RankGenerator(int tensor_parallel_size, int expert_parallel_size, int data_parallel_size,
                           int pipeline_parallel_size, std::array<Axis, AXIS_COUNT> order = {TP, EP, DP, PP});

    int AxisSize(Axis axis) const;
    int WorldSize() const;
    const std::array<Axis, AXIS_COUNT> &order() const;

    // Generate all rank groups by varying the specified axes and fixing all
    // other axes, following Megatron-LM's RankGenerator::get_ranks() ordering.
    std::vector<std::vector<int>> GetRanks(std::initializer_list<Axis> varying_axes) const;
    std::vector<std::vector<int>> GetRanks(Axis varying_axis) const;

    int GroupId(std::initializer_list<Axis> varying_axes, int global_rank) const;
    int GroupId(Axis varying_axis, int global_rank) const;
    std::vector<int> GroupRanks(std::initializer_list<Axis> varying_axes, int global_rank) const;
    std::vector<int> GroupRanks(Axis varying_axis, int global_rank) const;

private:
    void InitStrides();

    int RankOf(int dp, int tp, int pp, int ep) const;
    void CoordOf(int rank, int &dp, int &tp, int &pp, int &ep) const;

    std::array<int, AXIS_COUNT> sizes_{1, 1, 1, 1};
    std::array<Axis, AXIS_COUNT> order_{TP, EP, DP, PP};
    std::array<int, AXIS_COUNT> strides_{1, 1, 1, 1};
};

class GlobalEnv {
public:
    static GlobalEnv &Instance();

    void Init(int threads_per_process, int tensor_parallel_size, bool sequence_parallel_enabled,
              int pipeline_parallel_size, int virtual_pipeline_parallel_size, int expert_parallel_size,
              std::optional<int> expert_tensor_parallel_size);

    int nnodes() const;

    int nproc_per_node() const;

    int world_size() const;

    int global_proc_rank() const;

    int local_proc_rank() const;

    int nthread_per_process() const;

    int tensor_parallel_size() const;

    int expert_tensor_parallel_size() const;

    int sequence_parallel_size() const;

    bool sequence_parallel_enabled() const;

    int data_parallel_size() const;

    int expert_data_parallel_size() const;

    int pipeline_parallel_size() const;

    int virtual_pipeline_parallel_size() const;

    int expert_parallel_size() const;

    // Two logical rank views over the same physical world. The dense view uses
    // TP/DP, while the expert view exposes the same axes as ETP/EDP.
    const RankGenerator &dense_rank_generator() const;
    const RankGenerator &expert_rank_generator() const;

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
    int expert_tensor_parallel_size_ = 1;
    bool sequence_parallel_enabled_ = false;

    int data_parallel_size_ = 1;
    int expert_data_parallel_size_ = 1;

    int pipeline_parallel_size_ = 1;
    int virtual_pipeline_parallel_size_ = 1;
    int expert_parallel_size_ = 1;

    mutable std::mutex mutex_;
    bool initialized_ = false;

    RankGenerator dense_rank_generator_{1, 1, 1, 1};
    RankGenerator expert_rank_generator_{1, 1, 1, 1};
};

inline void InitAllEnv(int nthread_per_process, int tensor_parallel_size, bool sequence_parallel_enabled,
                       int pipeline_parallel_size, int virtual_pipeline_parallel, int expert_parallel_size,
                       std::optional<int> expert_tensor_parallel_size) {
    GlobalEnv::Instance().Init(nthread_per_process, tensor_parallel_size, sequence_parallel_enabled,
                               pipeline_parallel_size, virtual_pipeline_parallel, expert_parallel_size,
                               expert_tensor_parallel_size);
}
inline int GetNnodes() { return GlobalEnv::Instance().nnodes(); }
inline int GetWorldSize() { return GlobalEnv::Instance().world_size(); }
inline int GetNprocPerNode() { return GlobalEnv::Instance().nproc_per_node(); }
inline int GetNthreadPerProc() { return GlobalEnv::Instance().nthread_per_process(); }
inline int GetGlobalProcRank() { return GlobalEnv::Instance().global_proc_rank(); }
inline int GetLocalProcRank() { return GlobalEnv::Instance().local_proc_rank(); }

inline int GetTensorParallelSize() { return GlobalEnv::Instance().tensor_parallel_size(); }
inline int GetExpertTensorParallelSize() { return GlobalEnv::Instance().expert_tensor_parallel_size(); }
inline int GetSequenceParallelSize() { return GlobalEnv::Instance().sequence_parallel_size(); }
inline bool GetSequenceParallelEnabled() { return GlobalEnv::Instance().sequence_parallel_enabled(); }
inline int GetDataParallelSize() { return GlobalEnv::Instance().data_parallel_size(); }
inline int GetExpertDataParallelSize() { return GlobalEnv::Instance().expert_data_parallel_size(); }
inline int GetPipelineParallelSize() { return GlobalEnv::Instance().pipeline_parallel_size(); }
inline int GetVirtualPipelineParallelSize() { return GlobalEnv::Instance().virtual_pipeline_parallel_size(); }
inline int GetExpertParallelSize() { return GlobalEnv::Instance().expert_parallel_size(); }

inline const RankGenerator &GetDenseRankGenerator() { return GlobalEnv::Instance().dense_rank_generator(); }
inline const RankGenerator &GetExpertRankGenerator() { return GlobalEnv::Instance().expert_rank_generator(); }

/**
 * @brief Generates a human-readable overview of dense parallel groups.
 *
 * The dense view reports only TP, DP, and PP. Its size-one EP axis is an
 * implementation detail of RankGenerator and is intentionally omitted. If
 * dense_rank_generator is omitted, the global dense rank generator is used.
 *
 * @param dense_rank_generator Rank generator for the dense TP/DP/PP view.
 * @param skip_trivial_axes If true, groups whose size is one are marked as
 *        "unenabled" and their rank lists are omitted.
 *
 * Example output for dense {TP=2, DP=2, PP=1}:
 * @code
 * === Parallel Communication Groups ===
 * world_size = 4, config: {DP=2, TP=2, PP=1}
 * [Dense Rank View] shape={TP=2, DP=2, PP=1}, order={ TP -> DP -> PP }
 * [DP] size=2, num_groups=2
 *   - DP 0: [0, 2]
 *   - DP 1: [1, 3]
 * [TP] size=2, num_groups=2
 *   - TP 0: [0, 1]
 *   - TP 1: [2, 3]
 * [PP] size=1, unenabled
 * @endcode
 *
 * @return A formatted overview suitable for logging and topology validation.
 */
std::string ProcessGroupOverview(const RankGenerator &dense_rank_generator = GetDenseRankGenerator(),
                                 bool skip_trivial_axes = true);

/**
 * @brief Generates a human-readable overview of dense and expert parallel groups.
 *
 * Both rank generators describe logical views over the same physical ranks.
 * In the expert view, the TP and DP axes represent ETP and EDP. This overload
 * additionally reports EP and combined ETP+EP groups.
 *
 * @param dense_rank_generator Rank generator for the dense TP/DP/PP view.
 * @param expert_rank_generator Rank generator for the ETP/EP/EDP/PP view.
 * @param skip_trivial_axes If true, groups whose size is one are marked as
 *        "unenabled" and their rank lists are omitted.
 *
 * Example output for dense {TP=2, DP=2, PP=1} and expert
 * {ETP=1, EP=2, EDP=2, PP=1} views:
 * @code
 * === Parallel Communication Groups ===
 * world_size = 4, config: {DP=2, EDP=2, TP=2, ETP=1, PP=1, EP=2}
 * [Dense Rank View] shape={TP=2, DP=2, PP=1}, order={ TP -> DP -> PP }
 * [DP] size=2, num_groups=2
 *   - DP 0: [0, 2]
 *   - DP 1: [1, 3]
 * [TP] size=2, num_groups=2
 *   - TP 0: [0, 1]
 *   - TP 1: [2, 3]
 * [PP] size=1, unenabled
 *
 * [Expert Rank View] shape={ETP=1, EP=2, EDP=2, PP=1}, order={ ETP -> EP -> EDP -> PP }
 * [EDP] size=2, num_groups=2
 *   - EDP 0: [0, 2]
 *   - EDP 1: [1, 3]
 * [ETP] size=1, unenabled
 * [EP] size=2, num_groups=2
 *   - EP 0: [0, 1]
 *   - EP 1: [2, 3]
 * [ETP_EP] size=2, num_groups=2
 *   - ETP_EP 0: [0, 1]
 *   - ETP_EP 1: [2, 3]
 * @endcode
 *
 * @return A formatted overview suitable for logging and topology validation.
 */
std::string ProcessGroupOverview(const RankGenerator &dense_rank_generator, const RankGenerator &expert_rank_generator,
                                 bool skip_trivial_axes = true);

} // namespace infini_train::nn::parallel::global
