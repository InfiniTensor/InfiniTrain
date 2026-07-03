#include "infini_train/include/nn/parallel/global.h"

#include <cstdlib>
#include <format>
#include <string>

#include "glog/logging.h"

namespace {

int GetEnvAsInt(const std::string &name, int default_value) {
    const char *value = std::getenv(name.c_str());
    return value ? std::atoi(value) : default_value;
}

bool HasEnv(const std::string &name) { return std::getenv(name.c_str()) != nullptr; }

} // namespace

namespace infini_train::nn::parallel::global {

thread_local int thread_global_rank = 0;

void Layout::InitStrides() {
    // Calculate strides
    int stride = 1;
    for (int i = 0; i < AXIS_COUNT; ++i) {
        const Axis ax = order[i];
        strides[ax] = stride;
        stride *= sizes[ax];
    }
}

int Layout::RankOf(int dp, int tp, int pp) const {
    // Return the thread rank given layout coords
    const int coord[AXIS_COUNT] = {dp, tp, pp};
    int r = 0;
    for (int i = 0; i < AXIS_COUNT; ++i) {
        const Axis ax = static_cast<Axis>(i);
        r += coord[ax] * strides[ax];
    }
    return r;
}

void Layout::CoordOf(int rank, int &dp, int &tp, int &pp) const {
    // Return the layout coords given thread rank
    dp = (rank / strides[DP]) % sizes[DP];
    tp = (rank / strides[TP]) % sizes[TP];
    pp = (rank / strides[PP]) % sizes[PP];
}

int Layout::GroupId(Axis target, int dp, int tp, int pp) const {
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

std::vector<int> Layout::GroupRanks(Axis target, int fixed_dp, int fixed_tp, int fixed_pp) const {
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

GlobalEnv &GlobalEnv::Instance() {
    static GlobalEnv instance;
    return instance;
}

void GlobalEnv::Init(int nthread_per_process, int tensor_parallel_size, bool sequence_parallel_enabled,
                     int pipeline_parallel_size, int virtual_pipeline_parallel_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    CHECK(!initialized_) << "Repeated initialization of GlobalEnv!";

    const int proc_world_size = GetEnvAsInt("PROC_WORLD_SIZE", GetEnvAsInt("WORLD_SIZE", 1));
    nproc_per_node_ = GetEnvAsInt("NPROC_PER_NODE", GetEnvAsInt("LOCAL_WORLD_SIZE", 1));
    CHECK_GT(nproc_per_node_, 0) << "NPROC_PER_NODE/LOCAL_WORLD_SIZE must be positive";
    CHECK_GT(proc_world_size, 0) << "PROC_WORLD_SIZE/WORLD_SIZE must be positive";
    CHECK_EQ(proc_world_size % nproc_per_node_, 0)
        << "PROC_WORLD_SIZE/WORLD_SIZE must be divisible by NPROC_PER_NODE/LOCAL_WORLD_SIZE";
    const bool nnodes_env_set = HasEnv("NNODES");
    nnodes_ = GetEnvAsInt("NNODES", proc_world_size / nproc_per_node_);
    CHECK_GT(nnodes_, 0) << "NNODES must be positive";
    if (nnodes_env_set) {
        CHECK_EQ(nnodes_ * nproc_per_node_, proc_world_size)
            << "NNODES * NPROC_PER_NODE/LOCAL_WORLD_SIZE must equal PROC_WORLD_SIZE/WORLD_SIZE";
    }
    global_proc_rank_ = GetEnvAsInt("GLOBAL_PROC_RANK", GetEnvAsInt("RANK", 0));
    local_proc_rank_ = GetEnvAsInt("LOCAL_PROC_RANK", GetEnvAsInt("LOCAL_RANK", 0));
    CHECK_GE(global_proc_rank_, 0) << "GLOBAL_PROC_RANK/RANK must be non-negative";
    CHECK_LT(global_proc_rank_, proc_world_size)
        << "GLOBAL_PROC_RANK/RANK must be less than PROC_WORLD_SIZE/WORLD_SIZE";
    CHECK_GE(local_proc_rank_, 0) << "LOCAL_PROC_RANK/LOCAL_RANK must be non-negative";
    CHECK_LT(local_proc_rank_, nproc_per_node_)
        << "LOCAL_PROC_RANK/LOCAL_RANK must be less than NPROC_PER_NODE/LOCAL_WORLD_SIZE";

    nthread_per_process_ = nthread_per_process;
    world_size_ = proc_world_size * nthread_per_process;
    CHECK_GE(tensor_parallel_size, 1) << "Tensor Parallel size must be >= 1";
    tensor_parallel_size_ = tensor_parallel_size;
    sequence_parallel_enabled_ = sequence_parallel_enabled;
    pipeline_parallel_size_ = pipeline_parallel_size;
    virtual_pipeline_parallel_size_ = virtual_pipeline_parallel_size;
    data_parallel_size_ = world_size_ / tensor_parallel_size_ / pipeline_parallel_size_;

    layout_.sizes[DP] = data_parallel_size_;
    layout_.sizes[TP] = tensor_parallel_size_;
    layout_.sizes[PP] = pipeline_parallel_size_;
    layout_.InitStrides();

    initialized_ = true;
}

int GlobalEnv::nnodes() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return nnodes_;
}

int GlobalEnv::nproc_per_node() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return nproc_per_node_;
}

int GlobalEnv::nthread_per_process() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return nthread_per_process_;
}

int GlobalEnv::world_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return world_size_;
}

int GlobalEnv::global_proc_rank() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return global_proc_rank_;
}

int GlobalEnv::local_proc_rank() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return local_proc_rank_;
}

int GlobalEnv::tensor_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return tensor_parallel_size_;
}

int GlobalEnv::sequence_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return sequence_parallel_enabled_ ? tensor_parallel_size_ : 1;
}

bool GlobalEnv::sequence_parallel_enabled() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return sequence_parallel_enabled_;
}

int GlobalEnv::data_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return data_parallel_size_;
}

int GlobalEnv::pipeline_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return pipeline_parallel_size_;
}

int GlobalEnv::virtual_pipeline_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return virtual_pipeline_parallel_size_;
}

Layout GlobalEnv::layout() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return layout_;
}

namespace {
inline const char *AxisName(Axis a) { return a == DP ? "DP" : (a == TP ? "TP" : "PP"); }

inline int NumGroups(const Layout &L, Axis target) {
    int n = 1;
    for (int i = 0; i < AXIS_COUNT; ++i) {
        if (i != target) {
            n *= L.sizes[i];
        }
    }
    return n;
}
} // namespace

std::string ProcessGroupOverview(const Layout &L, bool skip_trivial_axes) {
    std::ostringstream oss;
    oss << std::format("\n=== Parallel Communication Groups ===\n"
                       "world_size = {}, config: {{DP={}, TP={}, PP={}}}, order: {{",
                       GetWorldSize(), L.sizes[DP], L.sizes[TP], L.sizes[PP]);

    for (int i = 0; i < AXIS_COUNT; ++i) { oss << AxisName(L.order[i]) << (i + 1 == AXIS_COUNT ? "" : " -> "); }
    oss << "}\n";

    for (int a = 0; a < AXIS_COUNT; ++a) {
        Axis ax = static_cast<Axis>(a);
        if (skip_trivial_axes && L.sizes[ax] <= 1) {
            oss << std::format("[{}] size={}, unenabled\n", AxisName(ax), L.sizes[ax]);
            continue;
        }
        // Build <Group ID, <DP, TP, PP>> mapping
        std::vector<std::pair<int, std::tuple<int, int, int>>> groups;
        for (int dp = 0; dp < (ax == DP ? 1 : L.sizes[DP]); ++dp) {
            for (int tp = 0; tp < (ax == TP ? 1 : L.sizes[TP]); ++tp) {
                for (int pp = 0; pp < (ax == PP ? 1 : L.sizes[PP]); ++pp) {
                    int gid = L.GroupId(ax, dp, tp, pp);
                    groups.emplace_back(gid, std::make_tuple(dp, tp, pp));
                }
            }
        }
        // Sort by the order of Group ID
        std::sort(groups.begin(), groups.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

        const int num_groups = NumGroups(L, ax);
        const auto name = AxisName(ax);
        oss << std::format("[{}] size={}, num_groups={}\n", name, L.sizes[ax], num_groups);

        // Iterate and print in the order of Group ID
        for (const auto &pair : groups) {
            int gid = pair.first;
            int dp, tp, pp;
            std::tie(dp, tp, pp) = pair.second;
            auto ranks = L.GroupRanks(ax, dp, tp, pp);
            std::sort(ranks.begin(), ranks.end());

            auto dp_size_str = (ax == DP) ? "-" : std::to_string(dp);
            auto tp_size_str = (ax == TP) ? "-" : std::to_string(tp);
            auto pp_size_str = (ax == PP) ? "-" : std::to_string(pp);

            std::string ranks_str;
            ranks_str.reserve(ranks.size() * 4);
            for (size_t i = 0; i < ranks.size(); ++i) {
                if (i > 0) {
                    ranks_str += ", ";
                }
                ranks_str += std::to_string(ranks[i]);
            }
            oss << std::format("  - {} {} (dp={}, tp={}, pp={}): [{}]\n", name, gid, dp_size_str, tp_size_str,
                               pp_size_str, ranks_str);
        }
        if (a + 1 < AXIS_COUNT) {
            oss << "\n";
        }
    }
    oss << "\n";
    return oss.str();
}

} // namespace infini_train::nn::parallel::global
