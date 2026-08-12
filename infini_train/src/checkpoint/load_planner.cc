#include "infini_train/include/checkpoint/load_planner.h"

#include <algorithm>
#include <set>
#include <tuple>
#include <unordered_map>

#include "glog/logging.h"

namespace infini_train::checkpoint {
namespace {

DataType StringToDataType(const std::string &value) {
    static const std::unordered_map<std::string, DataType> legacy_names = {
        {"bfloat16", DataType::kBFLOAT16},
        {"float16", DataType::kFLOAT16},
        {"float32", DataType::kFLOAT32},
        {"float64", DataType::kFLOAT64},
    };
    if (const auto it = legacy_names.find(value); it != legacy_names.end()) {
        return it->second;
    }
    for (const auto &[dtype, description] : kDataTypeToDesc) {
        if (description == value) {
            return dtype;
        }
    }
    LOG(FATAL) << "Unsupported checkpoint tensor dtype: " << value;
    return DataType::kFLOAT32;
}

int FragmentedAxis(const std::vector<int> &axis_fragmentations) {
    int fragmented_axis = -1;
    for (size_t dim = 0; dim < axis_fragmentations.size(); ++dim) {
        if (axis_fragmentations[dim] <= 1) {
            continue;
        }
        CHECK_EQ(fragmented_axis, -1) << "Multi-axis checkpoint sharding is not supported yet";
        fragmented_axis = static_cast<int>(dim);
    }
    return fragmented_axis;
}

bool IsVocabularyTensor(const std::string &key) {
    std::string parameter_key = key;
    if (parameter_key.starts_with("adam.m.")) {
        parameter_key = parameter_key.substr(7);
    } else if (parameter_key.starts_with("adam.v.")) {
        parameter_key = parameter_key.substr(7);
    }
    return parameter_key == "transformer.wte.weight" || parameter_key == "lm_head.weight";
}

bool IsPaddingCompatible(const std::string &key, const std::vector<int64_t> &source,
                         const std::vector<int64_t> &target) {
    if (!IsVocabularyTensor(key) || source.size() != target.size() || source.empty()) {
        return false;
    }
    for (size_t dim = 1; dim < source.size(); ++dim) {
        if (source[dim] != target[dim]) {
            return false;
        }
    }
    return true;
}

void ValidateCoordinates(const std::string &key, const std::vector<int64_t> &global_shape,
                         const std::vector<int64_t> &local_shape, const std::vector<int64_t> &global_offset,
                         const std::vector<int> &axis_fragmentations) {
    CHECK_EQ(global_shape.size(), local_shape.size()) << "Invalid local rank for tensor " << key;
    CHECK_EQ(global_shape.size(), global_offset.size()) << "Invalid offset rank for tensor " << key;
    CHECK_EQ(global_shape.size(), axis_fragmentations.size()) << "Invalid fragmentation rank for tensor " << key;
    for (size_t dim = 0; dim < global_shape.size(); ++dim) {
        CHECK_GT(global_shape[dim], 0) << "Invalid global shape for tensor " << key;
        CHECK_GT(local_shape[dim], 0) << "Invalid local shape for tensor " << key;
        CHECK_GE(global_offset[dim], 0) << "Invalid global offset for tensor " << key;
        CHECK_LE(global_offset[dim] + local_shape[dim], global_shape[dim])
            << "Shard exceeds global shape for tensor " << key;
        CHECK_GE(axis_fragmentations[dim], 1) << "Invalid axis fragmentation for tensor " << key;
    }
}

} // namespace

LoadPlan LoadPlanner::PlanReshard(const Checkpoint::CheckpointMetadata &metadata,
                                  const ShardedStateDict &target_state_dict) {
    LoadPlan plan;
    for (const auto &[key, target] : target_state_dict.tensors) {
        ValidateCoordinates(key, target.global_shape, target.local_shape, target.global_offset,
                            target.axis_fragmentations);
        TargetTensorPlan tensor_plan{.key = key,
                                     .dtype = target.dtype,
                                     .global_shape = target.global_shape,
                                     .target_shape = target.local_shape,
                                     .shard_dim = FragmentedAxis(target.axis_fragmentations)};

        std::vector<const Checkpoint::CheckpointMetadata::TensorEntry *> candidates;
        for (const auto &entry : metadata.tensors) {
            if (entry.key == key) {
                candidates.push_back(&entry);
            }
        }
        CHECK(!candidates.empty()) << "No saved shard found for target tensor: " << key;
        const int saved_axis = FragmentedAxis(candidates.front()->axis_fragmentations);
        for (const auto *source : candidates) {
            ValidateCoordinates(key, source->global_shape, source->local_shape, source->global_offset,
                                source->axis_fragmentations);
            CHECK(source->global_shape == target.global_shape
                  || IsPaddingCompatible(key, source->global_shape, target.global_shape))
                << "Global shape changed for tensor " << key;
            CHECK_EQ(FragmentedAxis(source->axis_fragmentations), saved_axis)
                << "Inconsistent saved shard dimensions for tensor " << key;
        }

        if (!target.segments.empty()) {
            if (tensor_plan.shard_dim < 0) {
                tensor_plan.shard_dim = 0;
            }
            int64_t target_covered = 0;
            for (const auto &target_segment : target.segments) {
                CHECK_EQ(target_segment.local_offset, target_covered)
                    << "Gap or overlap in target segments for " << key;
                CHECK_GT(target_segment.length, 0);
                CHECK_LE(target_segment.global_offset + target_segment.length,
                         target.global_shape[tensor_plan.shard_dim]);
                target_covered += target_segment.length;
                for (const auto *source : candidates) {
                    CHECK(!source->segments.empty()) << "Saved checkpoint lacks segmented layout metadata for " << key;
                    for (const auto &source_segment : source->segments) {
                        const int64_t overlap_start
                            = std::max(source_segment.global_offset, target_segment.global_offset);
                        const int64_t overlap_end = std::min(source_segment.global_offset + source_segment.length,
                                                             target_segment.global_offset + target_segment.length);
                        if (overlap_start >= overlap_end) {
                            continue;
                        }
                        tensor_plan.reads.push_back({
                            .key = key,
                            .filename = source->file,
                            .dtype = StringToDataType(source->dtype_str),
                            .global_shape = source->global_shape,
                            .byte_size = source->byte_size,
                            .data_offset = source->offset,
                            .shard_dim = tensor_plan.shard_dim,
                            .source_offset = source_segment.local_offset + overlap_start - source_segment.global_offset,
                            .target_offset = target_segment.local_offset + overlap_start - target_segment.global_offset,
                            .length = overlap_end - overlap_start,
                            .source_shape = source->local_shape,
                        });
                    }
                }
            }
            CHECK_EQ(target_covered, target.local_shape[tensor_plan.shard_dim])
                << "Segmented layout does not cover target local tensor " << key;
            std::sort(
                tensor_plan.reads.begin(), tensor_plan.reads.end(),
                [](const ReadItem &left, const ReadItem &right) { return left.target_offset < right.target_offset; });
            int64_t covered = 0;
            for (const auto &read : tensor_plan.reads) {
                CHECK_EQ(read.target_offset, covered) << "Gap or overlap in segmented target plan for " << key;
                covered += read.length;
            }
            CHECK_EQ(covered, target_covered) << "Incomplete segmented target plan for " << key;
            plan.tensors.emplace(key, std::move(tensor_plan));
            continue;
        }

        if (tensor_plan.shard_dim < 0) {
            tensor_plan.shard_dim = saved_axis;
        }
        if (tensor_plan.shard_dim < 0 && candidates.front()->global_shape != target.global_shape) {
            tensor_plan.shard_dim = 0;
        }
        if (saved_axis >= 0 && FragmentedAxis(target.axis_fragmentations) >= 0) {
            CHECK_EQ(saved_axis, tensor_plan.shard_dim) << "Shard dimension changed for tensor " << key;
        }

        if (tensor_plan.shard_dim < 0) {
            const auto *source = candidates.front();
            tensor_plan.reads.push_back({.key = key,
                                         .filename = source->file,
                                         .dtype = StringToDataType(source->dtype_str),
                                         .global_shape = source->global_shape,
                                         .byte_size = source->byte_size,
                                         .data_offset = source->offset,
                                         .shard_dim = -1,
                                         .source_shape = source->local_shape});
            plan.tensors.emplace(key, std::move(tensor_plan));
            continue;
        }

        const int dim = tensor_plan.shard_dim;
        const int64_t target_start = target.global_offset[dim];
        const int64_t target_length = target.local_shape[dim];
        const int64_t target_end = target_start + target_length;
        std::set<std::pair<int64_t, int64_t>> seen_source_ranges;

        for (const auto *source : candidates) {
            const int64_t saved_start = source->global_offset[dim];
            const int64_t saved_length = source->local_shape[dim];
            if (!seen_source_ranges.emplace(saved_start, saved_length).second) {
                continue;
            }
            const int64_t overlap_start = std::max(saved_start, target_start);
            const int64_t overlap_end = std::min(saved_start + saved_length, target_end);
            if (overlap_start >= overlap_end) {
                continue;
            }

            tensor_plan.reads.push_back({.key = key,
                                         .filename = source->file,
                                         .dtype = StringToDataType(source->dtype_str),
                                         .global_shape = source->global_shape,
                                         .byte_size = source->byte_size,
                                         .data_offset = source->offset,
                                         .shard_dim = dim,
                                         .source_offset = overlap_start - saved_start,
                                         .target_offset = overlap_start - target_start,
                                         .length = overlap_end - overlap_start,
                                         .source_shape = source->local_shape});
        }

        std::sort(tensor_plan.reads.begin(), tensor_plan.reads.end(),
                  [](const ReadItem &left, const ReadItem &right) { return left.target_offset < right.target_offset; });
        int64_t covered = 0;
        for (const auto &read : tensor_plan.reads) {
            CHECK_EQ(read.target_offset, covered) << "Gap or overlap in target shard plan for " << key;
            covered += read.length;
        }
        if (covered < target_length) {
            CHECK(IsVocabularyTensor(key)) << "Incomplete target shard plan for " << key;
            CHECK_EQ(dim, 0) << "Vocabulary padding is only supported along dim 0";
            CHECK_EQ(target_start + covered, candidates.front()->global_shape[0])
                << "Only trailing vocabulary padding is supported for " << key;
            tensor_plan.trailing_zero_fill = target_length - covered;
            covered = target_length;
        }
        CHECK_EQ(covered, target_length) << "Incomplete target shard plan for " << key;
        plan.tensors.emplace(key, std::move(tensor_plan));
    }
    return plan;
}

} // namespace infini_train::checkpoint
