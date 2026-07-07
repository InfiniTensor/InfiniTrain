#include "infini_train/include/nn/parallel/global.h"

#include <cstdlib>
#include <format>
#include <sstream>
#include <string>
#include <tuple>

#include "glog/logging.h"

namespace {

int GetEnvAsInt(const std::string &name, int default_value) {
    const char *value = std::getenv(name.c_str());
    return value ? std::atoi(value) : default_value;
}

} // namespace

namespace infini_train::nn::parallel::global {

thread_local int thread_global_rank = 0;

void Layout::InitStrides() {
    int stride = 1;
    for (int i = 0; i < AXIS_COUNT; ++i) {
        const Axis ax = order[i];
        strides[ax] = stride;
        stride *= sizes[ax];
    }
}

int Layout::RankOf(int dp, int tp, int pp) const { return RankOf(dp, tp, /*cp=*/0, pp); }

int Layout::RankOf(int dp, int tp, int cp, int pp) const {
    const int coord[AXIS_COUNT] = {dp, tp, cp, pp};
    int r = 0;
    for (int i = 0; i < AXIS_COUNT; ++i) {
        const Axis ax = static_cast<Axis>(i);
        r += coord[ax] * strides[ax];
    }
    return r;
}

void Layout::CoordOf(int rank, int &dp, int &tp, int &pp) const {
    int cp = 0;
    CoordOf(rank, dp, tp, cp, pp);
}

void Layout::CoordOf(int rank, int &dp, int &tp, int &cp, int &pp) const {
    dp = (rank / strides[DP]) % sizes[DP];
    tp = (rank / strides[TP]) % sizes[TP];
    cp = (rank / strides[CP]) % sizes[CP];
    pp = (rank / strides[PP]) % sizes[PP];
}

int Layout::GroupId(Axis target, int dp, int tp, int pp) const { return GroupId(target, dp, tp, /*cp=*/0, pp); }

int Layout::GroupId(Axis target, int dp, int tp, int cp, int pp) const {
    int id = 0;
    int mult = 1;
    for (int i = AXIS_COUNT - 1; i >= 0; --i) {
        Axis ax = order[i];
        if (ax == target) {
            continue;
        }
        int c = (ax == DP ? dp : (ax == TP ? tp : (ax == CP ? cp : pp)));
        id += c * mult;
        mult *= sizes[ax];
    }
    return id;
}

std::vector<int> Layout::GroupRanks(Axis target, int fixed_dp, int fixed_tp, int fixed_pp) const {
    return GroupRanks(target, fixed_dp, fixed_tp, /*fixed_cp=*/0, fixed_pp);
}

std::vector<int> Layout::GroupRanks(Axis target, int fixed_dp, int fixed_tp, int fixed_cp, int fixed_pp) const {
    std::vector<int> ranks;
    ranks.reserve(sizes[target]);
    int dp = fixed_dp, tp = fixed_tp, cp = fixed_cp, pp = fixed_pp;
    for (int v = 0; v < sizes[target]; ++v) {
        if (target == DP) {
            dp = v;
        } else if (target == TP) {
            tp = v;
        } else if (target == CP) {
            cp = v;
        } else {
            pp = v;
        }
        ranks.push_back(RankOf(dp, tp, cp, pp));
    }
    return ranks;
}

GlobalEnv &GlobalEnv::Instance() {
    static GlobalEnv instance;
    return instance;
}

void GlobalEnv::Init(int nthread_per_process, int tensor_parallel_size, bool sequence_parallel_enabled,
                     int context_parallel_size, const std::string &context_parallel_comm_type,
                     int pipeline_parallel_size, int virtual_pipeline_parallel_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    CHECK(!initialized_) << "Repeated initialization of GlobalEnv!";

    nnodes_ = GetEnvAsInt("NNODES", 1);
    nproc_per_node_ = GetEnvAsInt("NPROC_PER_NODE", 1);
    world_size_ = GetEnvAsInt("PROC_WORLD_SIZE", 1) * nthread_per_process;
    global_proc_rank_ = GetEnvAsInt("GLOBAL_PROC_RANK", 0);
    local_proc_rank_ = GetEnvAsInt("LOCAL_PROC_RANK", 0);

    nthread_per_process_ = nthread_per_process;
    CHECK_GE(tensor_parallel_size, 1) << "Tensor Parallel size must be >= 1";
    tensor_parallel_size_ = tensor_parallel_size;
    sequence_parallel_enabled_ = sequence_parallel_enabled;
    CHECK_GE(context_parallel_size, 1) << "Context Parallel size must be >= 1";
    context_parallel_size_ = context_parallel_size;
    context_parallel_comm_type_ = context_parallel_comm_type;
    CHECK_GE(pipeline_parallel_size, 1) << "Pipeline Parallel size must be >= 1";
    pipeline_parallel_size_ = pipeline_parallel_size;
    virtual_pipeline_parallel_size_ = virtual_pipeline_parallel_size;

    CHECK_EQ(world_size_ % (tensor_parallel_size_ * context_parallel_size_ * pipeline_parallel_size_), 0)
        << "world_size must be divisible by TP * CP * PP";
    data_parallel_size_ = world_size_ / tensor_parallel_size_ / context_parallel_size_ / pipeline_parallel_size_;

    layout_.sizes[DP] = data_parallel_size_;
    layout_.sizes[TP] = tensor_parallel_size_;
    layout_.sizes[CP] = context_parallel_size_;
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

int GlobalEnv::context_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return context_parallel_size_;
}

const std::string &GlobalEnv::context_parallel_comm_type() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return context_parallel_comm_type_;
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
inline const char *AxisName(Axis a) { return a == DP ? "DP" : (a == TP ? "TP" : (a == CP ? "CP" : "PP")); }

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
                       "world_size = {}, config: {{DP={}, TP={}, CP={}, PP={}}}, order: {{",
                       GetWorldSize(), L.sizes[DP], L.sizes[TP], L.sizes[CP], L.sizes[PP]);

    for (int i = 0; i < AXIS_COUNT; ++i) { oss << AxisName(L.order[i]) << (i + 1 == AXIS_COUNT ? "" : " -> "); }
    oss << "}\n";

    for (int a = 0; a < AXIS_COUNT; ++a) {
        Axis ax = static_cast<Axis>(a);
        if (skip_trivial_axes && L.sizes[ax] <= 1) {
            oss << std::format("[{}] size={}, unenabled\n", AxisName(ax), L.sizes[ax]);
            continue;
        }

        std::vector<std::pair<int, std::tuple<int, int, int, int>>> groups;
        for (int dp = 0; dp < (ax == DP ? 1 : L.sizes[DP]); ++dp) {
            for (int tp = 0; tp < (ax == TP ? 1 : L.sizes[TP]); ++tp) {
                for (int cp = 0; cp < (ax == CP ? 1 : L.sizes[CP]); ++cp) {
                    for (int pp = 0; pp < (ax == PP ? 1 : L.sizes[PP]); ++pp) {
                        int gid = L.GroupId(ax, dp, tp, cp, pp);
                        groups.emplace_back(gid, std::make_tuple(dp, tp, cp, pp));
                    }
                }
            }
        }
        std::sort(groups.begin(), groups.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

        const int num_groups = NumGroups(L, ax);
        const auto name = AxisName(ax);
        oss << std::format("[{}] size={}, num_groups={}\n", name, L.sizes[ax], num_groups);

        for (const auto &pair : groups) {
            int gid = pair.first;
            auto [dp, tp, cp, pp] = pair.second;
            auto coord = std::format("dp={}, tp={}, cp={}, pp={}", ax == DP ? "-" : std::to_string(dp),
                                     ax == TP ? "-" : std::to_string(tp), ax == CP ? "-" : std::to_string(cp),
                                     ax == PP ? "-" : std::to_string(pp));

            oss << std::format("- {} {} ({}): [", name, gid, coord);
            auto ranks = L.GroupRanks(ax, dp, tp, cp, pp);
            for (size_t i = 0; i < ranks.size(); ++i) { oss << ranks[i] << (i + 1 == ranks.size() ? "" : ", "); }
            oss << "]\n";
        }
    }
    oss << "=====================================\n";
    return oss.str();
}

} // namespace infini_train::nn::parallel::global
