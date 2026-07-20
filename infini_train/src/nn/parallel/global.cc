#include "infini_train/include/nn/parallel/global.h"

#include <array>
#include <cstdlib>
#include <format>
#include <sstream>
#include <string>

#include "glog/logging.h"

namespace {

int GetEnvAsInt(const std::string &name, int default_value) {
    const char *value = std::getenv(name.c_str());
    return value ? std::atoi(value) : default_value;
}

} // namespace

namespace infini_train::nn::parallel::global {

thread_local int thread_global_rank = 0;

namespace {

std::array<bool, AXIS_COUNT> MakeAxisMask(std::initializer_list<Axis> axes) {
    CHECK_GT(axes.size(), 0);
    std::array<bool, AXIS_COUNT> mask{};
    for (const Axis axis : axes) {
        CHECK_GE(static_cast<int>(axis), 0);
        CHECK_LT(static_cast<int>(axis), AXIS_COUNT);
        mask[axis] = true;
    }
    return mask;
}

const char *AxisName(Axis axis) {
    if (axis == DP) {
        return "DP";
    } else if (axis == TP) {
        return "TP";
    } else if (axis == PP) {
        return "PP";
    } else if (axis == EP) {
        return "EP";
    }
    CHECK(false) << "Invalid Axis value: " << static_cast<int>(axis);
}

std::string OrderString(const RankGenerator &rank_generator, bool expert_view) {
    std::string result;
    bool is_first_axis = true;
    for (int index = 0; index < AXIS_COUNT; ++index) {
        const Axis axis = rank_generator.order()[index];
        if (!expert_view && axis == EP) {
            continue;
        }
        if (!is_first_axis) {
            result += " -> ";
        }
        is_first_axis = false;
        if (expert_view && axis == TP) {
            result += "ETP";
        } else if (expert_view && axis == DP) {
            result += "EDP";
        } else {
            result += AxisName(axis);
        }
    }
    return result;
}

void AppendGroups(std::ostringstream &oss, const char *name, const RankGenerator &rank_generator,
                  std::initializer_list<Axis> varying_axes, bool skip_trivial_axes) {
    const auto varying_axis_mask = MakeAxisMask(varying_axes);
    int group_size = 1;
    for (int axis = 0; axis < AXIS_COUNT; ++axis) {
        if (varying_axis_mask[axis]) {
            group_size *= rank_generator.AxisSize(static_cast<Axis>(axis));
        }
    }

    if (skip_trivial_axes && group_size <= 1) {
        oss << std::format("[{}] size={}, unenabled\n", name, group_size);
        return;
    }

    const auto groups = rank_generator.GetRanks(varying_axes);
    oss << std::format("[{}] size={}, num_groups={}\n", name, group_size, groups.size());
    for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
        std::string ranks_string;
        for (size_t rank_index = 0; rank_index < groups[group_id].size(); ++rank_index) {
            if (rank_index > 0) {
                ranks_string += ", ";
            }
            ranks_string += std::to_string(groups[group_id][rank_index]);
        }
        oss << std::format("  - {} {}: [{}]\n", name, group_id, ranks_string);
    }
}

void AppendDenseRankView(std::ostringstream &oss, const RankGenerator &rank_generator, bool skip_trivial_axes) {
    CHECK_EQ(rank_generator.AxisSize(EP), 1) << "The dense rank view must not contain an expert-parallel axis";
    oss << std::format("[Dense Rank View] shape={{TP={}, DP={}, PP={}}}, order={{ {} }}\n", rank_generator.AxisSize(TP),
                       rank_generator.AxisSize(DP), rank_generator.AxisSize(PP),
                       OrderString(rank_generator, /*expert_view=*/false));
    AppendGroups(oss, "DP", rank_generator, {DP}, skip_trivial_axes);
    AppendGroups(oss, "TP", rank_generator, {TP}, skip_trivial_axes);
    AppendGroups(oss, "PP", rank_generator, {PP}, skip_trivial_axes);
}

void AppendExpertRankView(std::ostringstream &oss, const RankGenerator &rank_generator, bool skip_trivial_axes) {
    oss << std::format("[Expert Rank View] shape={{ETP={}, EP={}, EDP={}, PP={}}}, order={{ {} }}\n",
                       rank_generator.AxisSize(TP), rank_generator.AxisSize(EP), rank_generator.AxisSize(DP),
                       rank_generator.AxisSize(PP), OrderString(rank_generator, /*expert_view=*/true));
    AppendGroups(oss, "EDP", rank_generator, {DP}, skip_trivial_axes);
    AppendGroups(oss, "ETP", rank_generator, {TP}, skip_trivial_axes);
    AppendGroups(oss, "EP", rank_generator, {EP}, skip_trivial_axes);
    AppendGroups(oss, "ETP_EP", rank_generator, {TP, EP}, skip_trivial_axes);
}

} // namespace

RankGenerator::RankGenerator(int tensor_parallel_size, int expert_parallel_size, int data_parallel_size,
                             int pipeline_parallel_size, std::array<Axis, AXIS_COUNT> order)
    : order_(order) {
    sizes_[DP] = data_parallel_size;
    sizes_[TP] = tensor_parallel_size;
    sizes_[PP] = pipeline_parallel_size;
    sizes_[EP] = expert_parallel_size;

    for (const int size : sizes_) { CHECK_GE(size, 1) << "Parallel axis size must be >= 1"; }
    InitStrides();
}

void RankGenerator::InitStrides() {
    int stride = 1;
    for (int i = 0; i < AXIS_COUNT; ++i) {
        const Axis ax = order_[i];
        strides_[ax] = stride;
        stride *= sizes_[ax];
    }
}

int RankGenerator::AxisSize(Axis axis) const {
    CHECK_GE(static_cast<int>(axis), 0);
    CHECK_LT(static_cast<int>(axis), AXIS_COUNT);
    return sizes_[axis];
}

int RankGenerator::WorldSize() const {
    int world_size = 1;
    for (const int size : sizes_) { world_size *= size; }
    return world_size;
}

const std::array<Axis, AXIS_COUNT> &RankGenerator::order() const { return order_; }

int RankGenerator::RankOf(int dp, int tp, int pp, int ep) const {
    const int coord[AXIS_COUNT] = {dp, tp, pp, ep};
    int r = 0;
    for (int i = 0; i < AXIS_COUNT; ++i) {
        const Axis ax = static_cast<Axis>(i);
        r += coord[ax] * strides_[ax];
    }
    return r;
}

void RankGenerator::CoordOf(int rank, int &dp, int &tp, int &pp, int &ep) const {
    dp = (rank / strides_[DP]) % sizes_[DP];
    tp = (rank / strides_[TP]) % sizes_[TP];
    pp = (rank / strides_[PP]) % sizes_[PP];
    ep = (rank / strides_[EP]) % sizes_[EP];
}

int RankGenerator::GroupId(std::initializer_list<Axis> varying_axes, int global_rank) const {
    const auto varying_axis_mask = MakeAxisMask(varying_axes);
    int dp, tp, pp, ep;
    CoordOf(global_rank, dp, tp, pp, ep);
    const std::array<int, AXIS_COUNT> coordinates{dp, tp, pp, ep};

    // The first unmasked axis in rank order varies fastest in the group
    // ordinal, matching generate_masked_orthogonal_rank_groups in Megatron-LM.
    int group_id = 0;
    int group_stride = 1;
    for (const Axis axis : order_) {
        if (varying_axis_mask[axis]) {
            continue;
        }
        group_id += coordinates[axis] * group_stride;
        group_stride *= sizes_[axis];
    }
    return group_id;
}

int RankGenerator::GroupId(Axis varying_axis, int global_rank) const { return GroupId({varying_axis}, global_rank); }

std::vector<std::vector<int>> RankGenerator::GetRanks(std::initializer_list<Axis> varying_axes) const {
    const auto varying_axis_mask = MakeAxisMask(varying_axes);
    int group_size = 1;
    for (int axis = 0; axis < AXIS_COUNT; ++axis) {
        if (varying_axis_mask[axis]) {
            group_size *= sizes_[axis];
        }
    }

    const int num_groups = WorldSize() / group_size;
    std::vector<std::vector<int>> groups(num_groups);
    for (int rank = 0; rank < WorldSize(); ++rank) {
        const int group_id = GroupId(varying_axes, rank);
        CHECK_GE(group_id, 0);
        CHECK_LT(group_id, num_groups);
        groups[group_id].push_back(rank);
    }
    for (const auto &group : groups) { CHECK_EQ(group.size(), group_size); }
    return groups;
}

std::vector<std::vector<int>> RankGenerator::GetRanks(Axis varying_axis) const { return GetRanks({varying_axis}); }

std::vector<int> RankGenerator::GroupRanks(std::initializer_list<Axis> varying_axes, int global_rank) const {
    const auto varying_axis_mask = MakeAxisMask(varying_axes);
    int dp, tp, pp, ep;
    CoordOf(global_rank, dp, tp, pp, ep);
    std::array<int, AXIS_COUNT> coordinates{dp, tp, pp, ep};

    int group_size = 1;
    for (int axis = 0; axis < AXIS_COUNT; ++axis) {
        if (varying_axis_mask[axis]) {
            group_size *= sizes_[axis];
        }
    }

    std::vector<int> ranks;
    ranks.reserve(group_size);
    for (int rank_in_group = 0; rank_in_group < group_size; ++rank_in_group) {
        int remaining_index = rank_in_group;
        for (const Axis axis : order_) {
            if (!varying_axis_mask[axis]) {
                continue;
            }
            coordinates[axis] = remaining_index % sizes_[axis];
            remaining_index /= sizes_[axis];
        }
        CHECK_EQ(remaining_index, 0);
        ranks.push_back(RankOf(coordinates[DP], coordinates[TP], coordinates[PP], coordinates[EP]));
    }
    return ranks;
}

std::vector<int> RankGenerator::GroupRanks(Axis varying_axis, int global_rank) const {
    return GroupRanks({varying_axis}, global_rank);
}

GlobalEnv &GlobalEnv::Instance() {
    static GlobalEnv instance;
    return instance;
}

void GlobalEnv::Init(int nthread_per_process, int tensor_parallel_size, bool sequence_parallel_enabled,
                     int pipeline_parallel_size, int virtual_pipeline_parallel_size, int expert_parallel_size,
                     std::optional<int> expert_tensor_parallel_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    CHECK(!initialized_) << "Repeated initialization of GlobalEnv!";

    nnodes_ = GetEnvAsInt("NNODES", 1);
    nproc_per_node_ = GetEnvAsInt("NPROC_PER_NODE", 1);
    world_size_ = GetEnvAsInt("PROC_WORLD_SIZE", 1) * nthread_per_process;
    global_proc_rank_ = GetEnvAsInt("GLOBAL_PROC_RANK", 0);
    local_proc_rank_ = GetEnvAsInt("LOCAL_PROC_RANK", 0);

    nthread_per_process_ = nthread_per_process;
    CHECK_GE(tensor_parallel_size, 1) << "Tensor Parallel size must be >= 1";
    CHECK_GE(pipeline_parallel_size, 1) << "Pipeline Parallel size must be >= 1";
    CHECK_GE(virtual_pipeline_parallel_size, 1) << "Virtual Pipeline Parallel size must be >= 1";
    CHECK_GE(expert_parallel_size, 1) << "Expert Parallel size must be >= 1";
    const int resolved_expert_tensor_parallel_size = expert_tensor_parallel_size.value_or(tensor_parallel_size);
    CHECK_GE(resolved_expert_tensor_parallel_size, 1) << "Expert Tensor Parallel size must be >= 1";

    tensor_parallel_size_ = tensor_parallel_size;
    expert_tensor_parallel_size_ = resolved_expert_tensor_parallel_size;
    sequence_parallel_enabled_ = sequence_parallel_enabled;
    pipeline_parallel_size_ = pipeline_parallel_size;
    virtual_pipeline_parallel_size_ = virtual_pipeline_parallel_size;
    expert_parallel_size_ = expert_parallel_size;

    const int dense_model_parallel_size = tensor_parallel_size_ * pipeline_parallel_size_;
    CHECK_EQ(world_size_ % dense_model_parallel_size, 0)
        << "World size must be divisible by tensor_parallel_size * pipeline_parallel_size";
    data_parallel_size_ = world_size_ / dense_model_parallel_size;

    const int expert_model_parallel_size
        = expert_tensor_parallel_size_ * expert_parallel_size_ * pipeline_parallel_size_;
    CHECK_EQ(world_size_ % expert_model_parallel_size, 0)
        << "World size must be divisible by expert_tensor_parallel_size * expert_parallel_size"
           " * pipeline_parallel_size";
    expert_data_parallel_size_ = world_size_ / expert_model_parallel_size;

    // These are two logical views over the same physical ranks. TP and DP in
    // the expert generator are exposed as ETP and EDP, respectively.
    dense_rank_generator_ = RankGenerator(tensor_parallel_size_, /*expert_parallel_size=*/1, data_parallel_size_,
                                          pipeline_parallel_size_);
    expert_rank_generator_ = RankGenerator(expert_tensor_parallel_size_, expert_parallel_size_,
                                           expert_data_parallel_size_, pipeline_parallel_size_);

    CHECK(dense_rank_generator_.GetRanks(PP) == expert_rank_generator_.GetRanks(PP))
        << "Dense and expert rank views must generate identical pipeline-parallel groups";

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

int GlobalEnv::expert_tensor_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return expert_tensor_parallel_size_;
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

int GlobalEnv::expert_data_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return expert_data_parallel_size_;
}

int GlobalEnv::pipeline_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return pipeline_parallel_size_;
}

int GlobalEnv::virtual_pipeline_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return virtual_pipeline_parallel_size_;
}

int GlobalEnv::expert_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return expert_parallel_size_;
}

const RankGenerator &GlobalEnv::dense_rank_generator() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return dense_rank_generator_;
}

const RankGenerator &GlobalEnv::expert_rank_generator() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return expert_rank_generator_;
}

std::string ProcessGroupOverview(const RankGenerator &dense_rank_generator, bool skip_trivial_axes) {
    std::ostringstream oss;
    oss << std::format("\n=== Parallel Communication Groups ===\n"
                       "world_size = {}, config: {{DP={}, TP={}, PP={}}}\n",
                       dense_rank_generator.WorldSize(), dense_rank_generator.AxisSize(DP),
                       dense_rank_generator.AxisSize(TP), dense_rank_generator.AxisSize(PP));
    AppendDenseRankView(oss, dense_rank_generator, skip_trivial_axes);
    oss << "\n";
    return oss.str();
}

std::string ProcessGroupOverview(const RankGenerator &dense_rank_generator, const RankGenerator &expert_rank_generator,
                                 bool skip_trivial_axes) {
    CHECK_EQ(dense_rank_generator.WorldSize(), expert_rank_generator.WorldSize())
        << "Dense and expert rank views must cover the same physical world";

    std::ostringstream oss;
    oss << std::format("\n=== Parallel Communication Groups ===\n"
                       "world_size = {}, config: {{DP={}, EDP={}, TP={}, ETP={}, PP={}, EP={}}}\n",
                       dense_rank_generator.WorldSize(), dense_rank_generator.AxisSize(DP),
                       expert_rank_generator.AxisSize(DP), dense_rank_generator.AxisSize(TP),
                       expert_rank_generator.AxisSize(TP), dense_rank_generator.AxisSize(PP),
                       expert_rank_generator.AxisSize(EP));

    AppendDenseRankView(oss, dense_rank_generator, skip_trivial_axes);
    oss << "\n";
    AppendExpertRankView(oss, expert_rank_generator, skip_trivial_axes);
    oss << "\n";
    return oss.str();
}

} // namespace infini_train::nn::parallel::global
