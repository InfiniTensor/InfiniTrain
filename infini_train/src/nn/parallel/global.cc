#include "infini_train/include/nn/parallel/global.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <system_error>
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

}  // namespace

namespace infini_train::nn::parallel::global {

thread_local int thread_global_rank = 0;

namespace {

std::array<bool, kAxisCount> MakeAxisMask(std::initializer_list<Axis> axes) {
  CHECK_GT(axes.size(), 0);
  std::array<bool, kAxisCount> mask{};
  for (const Axis axis : axes) {
    CHECK_GE(static_cast<int>(axis), 0);
    CHECK_LT(static_cast<int>(axis), kAxisCount);
    mask[axis] = true;
  }
  return mask;
}

const char *AxisName(Axis axis) {
  if (axis == kDp) {
    return "DP";
  } else if (axis == kTp) {
    return "TP";
  } else if (axis == kPp) {
    return "PP";
  } else if (axis == kCp) {
    return "CP";
  } else if (axis == kEp) {
    return "EP";
  }
  CHECK(false) << "Invalid Axis value: " << static_cast<int>(axis);
}

std::string OrderString(const RankGenerator &rank_generator, bool expert_view) {
  std::string result;
  bool is_first_axis = true;
  for (int index = 0; index < kAxisCount; ++index) {
    const Axis axis = rank_generator.order()[index];
    if ((!expert_view && axis == kEp) || (axis == kCp && rank_generator.AxisSize(kCp) == 1)) {
      continue;
    }
    if (!is_first_axis) {
      result += " -> ";
    }
    is_first_axis = false;
    if (expert_view && axis == kTp) {
      result += "ETP";
    } else if (expert_view && axis == kDp) {
      result += "EDP";
    } else {
      result += AxisName(axis);
    }
  }
  return result;
}

void AppendGroups(std::ostringstream &oss, const char *name,
                  const RankGenerator &rank_generator,
                  std::initializer_list<Axis> varying_axes,
                  bool skip_trivial_axes) {
  const auto varying_axis_mask = MakeAxisMask(varying_axes);
  int group_size = 1;
  for (int axis = 0; axis < kAxisCount; ++axis) {
    if (varying_axis_mask[axis]) {
      group_size *= rank_generator.AxisSize(static_cast<Axis>(axis));
    }
  }

  if (skip_trivial_axes && group_size <= 1) {
    oss << std::format("[{}] size={}, unenabled\n", name, group_size);
    return;
  }

  const auto groups = rank_generator.GetRanks(varying_axes);
  oss << std::format("[{}] size={}, num_groups={}\n", name, group_size,
                     groups.size());
  for (size_t group_id = 0; group_id < groups.size(); ++group_id) {
    std::string ranks_string;
    for (size_t rank_index = 0; rank_index < groups[group_id].size();
         ++rank_index) {
      if (rank_index > 0) {
        ranks_string += ", ";
      }
      ranks_string += std::to_string(groups[group_id][rank_index]);
    }
    oss << std::format("  - {} {}: [{}]\n", name, group_id, ranks_string);
  }
}

void AppendDenseRankView(std::ostringstream &oss,
                         const RankGenerator &rank_generator,
                         bool skip_trivial_axes) {
  CHECK_EQ(rank_generator.AxisSize(kEp), 1)
      << "The dense rank view must not contain an expert-parallel axis";
  oss << std::format(
      "[Dense Rank View] shape={{TP={}, DP={}, PP={}}}, order={{ {} }}\n",
      rank_generator.AxisSize(kTp), rank_generator.AxisSize(kDp),
      rank_generator.AxisSize(kPp),
      OrderString(rank_generator, /*expert_view=*/false));
  AppendGroups(oss, "DP", rank_generator, {kDp}, skip_trivial_axes);
  AppendGroups(oss, "TP", rank_generator, {kTp}, skip_trivial_axes);
  AppendGroups(oss, "CP", rank_generator, {kCp}, skip_trivial_axes);
  AppendGroups(oss, "PP", rank_generator, {kPp}, skip_trivial_axes);
}

void AppendExpertRankView(std::ostringstream &oss,
                          const RankGenerator &rank_generator,
                          bool skip_trivial_axes) {
  oss << std::format(
      "[Expert Rank View] shape={{ETP={}, EP={}, EDP={}, PP={}}}, order={{ {} "
      "}}\n",
      rank_generator.AxisSize(kTp), rank_generator.AxisSize(kEp),
      rank_generator.AxisSize(kDp), rank_generator.AxisSize(kPp),
      OrderString(rank_generator, /*expert_view=*/true));
  AppendGroups(oss, "EDP", rank_generator, {kDp}, skip_trivial_axes);
  AppendGroups(oss, "ETP", rank_generator, {kTp}, skip_trivial_axes);
  AppendGroups(oss, "EP", rank_generator, {kEp}, skip_trivial_axes);
  AppendGroups(oss, "ETP_EP", rank_generator, {kTp, kEp}, skip_trivial_axes);
}

}  // namespace

RankGenerator::RankGenerator(int tensor_parallel_size, int expert_parallel_size,
                             int data_parallel_size, int pipeline_parallel_size,
                             int context_parallel_size,
                             std::array<Axis, kAxisCount> order)
    : order_(order) {
  sizes_[kDp] = data_parallel_size;
  sizes_[kTp] = tensor_parallel_size;
  sizes_[kPp] = pipeline_parallel_size;
  sizes_[kEp] = expert_parallel_size;
  sizes_[kCp] = context_parallel_size;

  for (const int size : sizes_) {
    CHECK_GE(size, 1) << "Parallel axis size must be >= 1";
  }
  InitStrides();
}

void RankGenerator::InitStrides() {
  int stride = 1;
  for (int i = 0; i < kAxisCount; ++i) {
    const Axis ax = order_[i];
    strides_[ax] = stride;
    stride *= sizes_[ax];
  }
}

int RankGenerator::AxisSize(Axis axis) const {
  CHECK_GE(static_cast<int>(axis), 0);
  CHECK_LT(static_cast<int>(axis), kAxisCount);
  return sizes_[axis];
}

int RankGenerator::WorldSize() const {
  int world_size = 1;
  for (const int size : sizes_) {
    world_size *= size;
  }
  return world_size;
}

const std::array<Axis, kAxisCount> &RankGenerator::order() const {
  return order_;
}

int RankGenerator::RankOf(int dp, int tp, int pp, int ep, int cp) const {
  const int coord[kAxisCount] = {dp, tp, pp, ep, cp};
  int r = 0;
  for (int i = 0; i < kAxisCount; ++i) {
    const Axis ax = static_cast<Axis>(i);
    r += coord[ax] * strides_[ax];
  }
  return r;
}

void RankGenerator::CoordOf(int rank, int &dp, int &tp, int &pp,
                            int &ep, int &cp) const {
  dp = (rank / strides_[kDp]) % sizes_[kDp];
  tp = (rank / strides_[kTp]) % sizes_[kTp];
  pp = (rank / strides_[kPp]) % sizes_[kPp];
  ep = (rank / strides_[kEp]) % sizes_[kEp];
  cp = (rank / strides_[kCp]) % sizes_[kCp];
}

int RankGenerator::GroupId(std::initializer_list<Axis> varying_axes,
                           int global_rank) const {
  const auto varying_axis_mask = MakeAxisMask(varying_axes);
  int dp, tp, pp, ep, cp;
  CoordOf(global_rank, dp, tp, pp, ep, cp);
  const std::array<int, kAxisCount> coordinates{dp, tp, pp, ep, cp};

  // Preserve the legacy Layout::GroupId numbering used in process-group
  // names: fixed axes later in rank order vary faster in the group ordinal.
  int group_id = 0;
  int group_stride = 1;
  for (int index = kAxisCount - 1; index >= 0; --index) {
    const Axis axis = order_[index];
    if (varying_axis_mask[axis]) {
      continue;
    }
    group_id += coordinates[axis] * group_stride;
    group_stride *= sizes_[axis];
  }
  return group_id;
}

int RankGenerator::GroupId(Axis varying_axis, int global_rank) const {
  return GroupId({varying_axis}, global_rank);
}

std::vector<std::vector<int>> RankGenerator::GetRanks(
    std::initializer_list<Axis> varying_axes) const {
  const auto varying_axis_mask = MakeAxisMask(varying_axes);
  int group_size = 1;
  for (int axis = 0; axis < kAxisCount; ++axis) {
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
  for (const auto &group : groups) {
    CHECK_EQ(group.size(), group_size);
  }
  return groups;
}

std::vector<std::vector<int>> RankGenerator::GetRanks(Axis varying_axis) const {
  return GetRanks({varying_axis});
}

std::vector<int> RankGenerator::GroupRanks(
    std::initializer_list<Axis> varying_axes, int global_rank) const {
  const auto varying_axis_mask = MakeAxisMask(varying_axes);
  int dp, tp, pp, ep, cp;
  CoordOf(global_rank, dp, tp, pp, ep, cp);
  std::array<int, kAxisCount> coordinates{dp, tp, pp, ep, cp};

  int group_size = 1;
  for (int axis = 0; axis < kAxisCount; ++axis) {
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
    ranks.push_back(RankOf(coordinates[kDp], coordinates[kTp], coordinates[kPp],
                           coordinates[kEp], coordinates[kCp]));
  }
  return ranks;
}

std::vector<int> RankGenerator::GroupRanks(Axis varying_axis,
                                           int global_rank) const {
  return GroupRanks({varying_axis}, global_rank);
}

GlobalEnv &GlobalEnv::Instance() {
  static GlobalEnv instance;
  return instance;
}

void GlobalEnv::Init(int nthread_per_process, int tensor_parallel_size,
                     bool sequence_parallel_enabled, int pipeline_parallel_size,
                     int virtual_pipeline_parallel_size,
                     int expert_parallel_size,
                     std::optional<int> expert_tensor_parallel_size,
                     int context_parallel_size,
                     const std::string &context_parallel_comm_type,
                     const std::string &hierarchical_context_parallel_sizes) {
  std::lock_guard<std::mutex> lock(mutex_);

  CHECK(!initialized_) << "Repeated initialization of GlobalEnv!";

  const int proc_world_size = GetEnvAsInt("WORLD_SIZE", 1);
  nproc_per_node_ = GetEnvAsInt("LOCAL_WORLD_SIZE", 1);
  CHECK_GT(nproc_per_node_, 0) << "LOCAL_WORLD_SIZE must be positive";
  CHECK_GT(proc_world_size, 0) << "WORLD_SIZE must be positive";
  CHECK_EQ(proc_world_size % nproc_per_node_, 0)
      << "WORLD_SIZE must be divisible by LOCAL_WORLD_SIZE";
  nnodes_ = proc_world_size / nproc_per_node_;
  global_proc_rank_ = GetEnvAsInt("RANK", 0);
  local_proc_rank_ = GetEnvAsInt("LOCAL_RANK", 0);
  CHECK_GE(global_proc_rank_, 0) << "RANK must be non-negative";
  CHECK_LT(global_proc_rank_, proc_world_size)
      << "RANK must be less than WORLD_SIZE";
  CHECK_GE(local_proc_rank_, 0) << "LOCAL_RANK must be non-negative";
  CHECK_LT(local_proc_rank_, nproc_per_node_)
      << "LOCAL_RANK must be less than LOCAL_WORLD_SIZE";

  nthread_per_process_ = nthread_per_process;
  world_size_ = proc_world_size * nthread_per_process;
  CHECK_GE(tensor_parallel_size, 1) << "Tensor Parallel size must be >= 1";
  CHECK_GE(pipeline_parallel_size, 1) << "Pipeline Parallel size must be >= 1";
  CHECK_GE(virtual_pipeline_parallel_size, 1)
      << "Virtual Pipeline Parallel size must be >= 1";
  CHECK_GE(expert_parallel_size, 1) << "Expert Parallel size must be >= 1";
  const int resolved_expert_tensor_parallel_size =
      expert_tensor_parallel_size.value_or(tensor_parallel_size);
  CHECK_GE(resolved_expert_tensor_parallel_size, 1)
      << "Expert Tensor Parallel size must be >= 1";

  tensor_parallel_size_ = tensor_parallel_size;
  expert_tensor_parallel_size_ = resolved_expert_tensor_parallel_size;
  sequence_parallel_enabled_ = sequence_parallel_enabled;
    CHECK_GE(context_parallel_size, 1) << "Context Parallel size must be >= 1";
    context_parallel_size_ = context_parallel_size;
    context_parallel_comm_type_ = context_parallel_comm_type;
    CHECK(context_parallel_comm_type == "p2p" || context_parallel_comm_type == "all_gather"
          || context_parallel_comm_type == "a2a" || context_parallel_comm_type == "a2a+p2p")
        << "Unsupported CP communication type: " << context_parallel_comm_type;
    if (context_parallel_comm_type == "a2a+p2p") {
        const auto separator = hierarchical_context_parallel_sizes.find(',');
        CHECK(separator != std::string::npos) << "hierarchical_context_parallel_sizes must be A,P";
        const auto *begin = hierarchical_context_parallel_sizes.data();
        const auto *end = begin + hierarchical_context_parallel_sizes.size();
        int a2a_size = 0, p2p_size = 0;
        const auto a2a_result = std::from_chars(begin, begin + separator, a2a_size);
        const auto p2p_result = std::from_chars(begin + separator + 1, end, p2p_size);
        CHECK(a2a_result.ec == std::errc{} && a2a_result.ptr == begin + separator && p2p_result.ec == std::errc{}
              && p2p_result.ptr == end)
            << "hierarchical_context_parallel_sizes must contain exactly two integers: A,P";
        CHECK_GT(a2a_size, 0);
        CHECK_GT(p2p_size, 0);
        CHECK_EQ(static_cast<int64_t>(a2a_size) * p2p_size, context_parallel_size)
            << "A2A size * P2P size must equal context_parallel";
        hierarchical_context_parallel_sizes_ = {a2a_size, p2p_size};
    } else {
        CHECK(hierarchical_context_parallel_sizes.empty())
            << "hierarchical_context_parallel_sizes requires cp_comm_type=a2a+p2p";
    }

  pipeline_parallel_size_ = pipeline_parallel_size;
  virtual_pipeline_parallel_size_ = virtual_pipeline_parallel_size;
  expert_parallel_size_ = expert_parallel_size;

  const int dense_model_parallel_size =
      tensor_parallel_size_ * context_parallel_size_ * pipeline_parallel_size_;
  CHECK_EQ(world_size_ % dense_model_parallel_size, 0)
      << "World size must be divisible by tensor_parallel_size * context_parallel_size * "
         "pipeline_parallel_size";
  data_parallel_size_ = world_size_ / dense_model_parallel_size;

  const int expert_model_parallel_size = expert_tensor_parallel_size_ *
                                         expert_parallel_size_ *
                                         pipeline_parallel_size_;
  CHECK_EQ(world_size_ % expert_model_parallel_size, 0)
      << "World size must be divisible by expert_tensor_parallel_size * "
         "expert_parallel_size"
         " * pipeline_parallel_size";
  expert_data_parallel_size_ = world_size_ / expert_model_parallel_size;

  // These are two logical views over the same physical ranks. TP and DP in
  // the expert generator are exposed as ETP and EDP, respectively.
  dense_rank_generator_ =
      RankGenerator(tensor_parallel_size_, /*expert_parallel_size=*/1,
                    data_parallel_size_, pipeline_parallel_size_, context_parallel_size_);
  expert_rank_generator_ =
      RankGenerator(expert_tensor_parallel_size_, expert_parallel_size_,
                    expert_data_parallel_size_, pipeline_parallel_size_);

  const auto dense_pipeline_groups = dense_rank_generator_.GetRanks(kPp);
  const auto expert_pipeline_groups = expert_rank_generator_.GetRanks(kPp);
  CHECK(std::is_permutation(
      dense_pipeline_groups.begin(), dense_pipeline_groups.end(),
      expert_pipeline_groups.begin(), expert_pipeline_groups.end()))
      << "Dense and expert rank views must generate identical "
         "pipeline-parallel groups";

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

int GlobalEnv::context_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return context_parallel_size_;
}

const std::string &GlobalEnv::context_parallel_comm_type() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return context_parallel_comm_type_;
}

const std::vector<int> &GlobalEnv::hierarchical_context_parallel_sizes() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return hierarchical_context_parallel_sizes_;
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

std::string ProcessGroupOverview(const RankGenerator &dense_rank_generator,
                                 bool skip_trivial_axes) {
  std::ostringstream oss;
  oss << std::format(
      "\n=== Parallel Communication Groups ===\n"
      "world_size = {}, config: {{DP={}, TP={}, PP={}}}\n",
      dense_rank_generator.WorldSize(), dense_rank_generator.AxisSize(kDp),
      dense_rank_generator.AxisSize(kTp), dense_rank_generator.AxisSize(kPp));
  AppendDenseRankView(oss, dense_rank_generator, skip_trivial_axes);
  oss << "\n";
  return oss.str();
}

std::string ProcessGroupOverview(const RankGenerator &dense_rank_generator,
                                 const RankGenerator &expert_rank_generator,
                                 bool skip_trivial_axes) {
  CHECK_EQ(dense_rank_generator.WorldSize(), expert_rank_generator.WorldSize())
      << "Dense and expert rank views must cover the same physical world";

  std::ostringstream oss;
  oss << std::format(
      "\n=== Parallel Communication Groups ===\n"
      "world_size = {}, config: {{DP={}, EDP={}, TP={}, ETP={}, PP={}, "
      "EP={}}}\n",
      dense_rank_generator.WorldSize(), dense_rank_generator.AxisSize(kDp),
      expert_rank_generator.AxisSize(kDp), dense_rank_generator.AxisSize(kTp),
      expert_rank_generator.AxisSize(kTp), dense_rank_generator.AxisSize(kPp),
      expert_rank_generator.AxisSize(kEp));

  AppendDenseRankView(oss, dense_rank_generator, skip_trivial_axes);
  oss << "\n";
  AppendExpertRankView(oss, expert_rank_generator, skip_trivial_axes);
  oss << "\n";
  return oss.str();
}

}  // namespace infini_train::nn::parallel::global
