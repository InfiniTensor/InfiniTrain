#include "infini_train/include/nn/parallel/global.h"

#include <cstdlib>
#include <string>

#include "glog/logging.h"

namespace {

int GetEnvAsInt(const std::string &name, int default_value) {
    const char *value = std::getenv(name.c_str());
    return value ? std::atoi(value) : default_value;
}

} // namespace

namespace infini_train::nn::parallel::global {

GlobalEnv &GlobalEnv::Instance() {
    static GlobalEnv instance;
    return instance;
}

void GlobalEnv::Init(int nthread_per_process, int tensor_parallel_size, bool sequence_parallel_enabled) {
    std::lock_guard<std::mutex> lock(mutex_);

    CHECK(!initialized_) << "Repeated initialization of GlobalEnv!";

    world_size_ = GetEnvAsInt("PROC_WORLD_SIZE", 1) * nthread_per_process;
    nproc_per_node_ = GetEnvAsInt("NPROC_PER_NODE", 1);
    global_proc_rank_ = GetEnvAsInt("GLOBAL_PROC_RANK", 0);
    local_proc_rank_ = GetEnvAsInt("LOCAL_PROC_RANK", 0);

    nthread_per_process_ = nthread_per_process;
    CHECK_GE(tensor_parallel_size, 1) << "Tensor Parallel size must be >= 1";
    tensor_parallel_size_ = tensor_parallel_size;
    sequence_parallel_enabled_ = sequence_parallel_enabled;
    data_parallel_size_ = world_size_ / tensor_parallel_size_;

    layout_.sizes[DP] = data_parallel_size_;
    layout_.sizes[TP] = tensor_parallel_size_;
    // FIXME(zbl): set PP size
    layout_.sizes[PP] = 1;
    layout_.InitStrides();

    initialized_ = true;
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

int GlobalEnv::nproc_per_node() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return nproc_per_node_;
}

int GlobalEnv::nthread_per_process() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return nthread_per_process_;
}

int GlobalEnv::tensor_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return tensor_parallel_size_;
}

bool GlobalEnv::sequence_parallel_enabled() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return sequence_parallel_enabled_;
}

int GlobalEnv::data_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return data_parallel_size_;
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

inline void AppendAxisGroups(std::ostringstream &oss, const Layout &L, Axis target) {
    const int ng = NumGroups(L, target);
    const auto name = AxisName(target);
    oss << "[" << name << "] size=" << L.sizes[target] << ", num_groups=" << ng << "\n";

    for (int dp = 0; dp < (target == DP ? 1 : L.sizes[DP]); ++dp) {
        for (int tp = 0; tp < (target == TP ? 1 : L.sizes[TP]); ++tp) {
            for (int pp = 0; pp < (target == PP ? 1 : L.sizes[PP]); ++pp) {
                const int gid = L.GroupId(target, dp, tp, pp);
                auto ranks = L.GroupRanks(target, dp, tp, pp);
                std::sort(ranks.begin(), ranks.end());

                oss << "  - " << name << " " << gid << " (dp=" << (target == DP ? "-" : std::to_string(dp))
                    << ", tp=" << (target == TP ? "-" : std::to_string(tp))
                    << ", pp=" << (target == PP ? "-" : std::to_string(pp)) << "): [";

                for (size_t i = 0; i < ranks.size(); ++i) {
                    if (i) {
                        oss << ", ";
                    }
                    oss << ranks[i];
                }
                oss << "]\n";
            }
        }
    }
}

/* Example:
    === Parallel Communication Groups ===
    world_size = 8, config: {DP=2, TP=4, PP=1}, order: {DP -> TP -> PP}
    [DP] size=2, num_groups=4
    - DP 0 (dp=-, tp=0, pp=0): [0, 4]
    - DP 1 (dp=-, tp=1, pp=0): [1, 5]
    - DP 2 (dp=-, tp=2, pp=0): [2, 6]
    - DP 3 (dp=-, tp=3, pp=0): [3, 7]

    [TP] size=4, num_groups=2
    - TP 0 (dp=0, tp=-, pp=0): [0, 1, 2, 3]
    - TP 1 (dp=1, tp=-, pp=0): [4, 5, 6, 7]

    [PP] size=1, unenabled
*/
std::string ProcessGroupOverview(const Layout &L, bool skip_trivial_axes) {
    std::ostringstream oss;
    oss << "\n=== Parallel Communication Groups ===\n"
        << "world_size = " << GetWorldSize() << ", config: {DP=" << L.sizes[DP] << ", TP=" << L.sizes[TP]
        << ", PP=" << L.sizes[PP] << "}, order: {";
    for (int i = 0; i < AXIS_COUNT; ++i) { oss << AxisName(L.order[i]) << (i + 1 == AXIS_COUNT ? "" : " -> "); }
    oss << "}\n";

    for (int a = 0; a < AXIS_COUNT; ++a) {
        Axis ax = static_cast<Axis>(a);
        if (skip_trivial_axes && L.sizes[ax] <= 1) {
            oss << "[" << AxisName(ax) << "] size=" << L.sizes[ax] << ", unenabled\n";
            continue;
        }
        AppendAxisGroups(oss, L, ax);
        if (a + 1 < AXIS_COUNT) {
            oss << "\n";
        }
    }
    oss << "\n";
    return oss.str();
}

} // namespace infini_train::nn::parallel::global
