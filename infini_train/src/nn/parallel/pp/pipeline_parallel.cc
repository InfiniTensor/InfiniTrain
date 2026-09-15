// pipeline_parallel.cc
#include "infini_train/include/nn/parallel/pp/pipeline_parallel.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "infini_train/include/nn/modules/container.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/pp/pipeline_schedule.h"
#include "infini_train/include/nn/parallel/pp/pipeline_stage.h"

namespace infini_train::nn::parallel {
namespace {
constexpr char kModuleName[] = "module";
thread_local std::optional<PipelineLayout> pipeline_layout;

void CheckStage(int stage, int num_stages) {
    if (stage < 0 || stage >= num_stages) {
        throw std::out_of_range("pipeline stage " + std::to_string(stage) + " is outside [0, "
                                + std::to_string(num_stages) + ")");
    }
}

std::string ExpandLayoutExpression(const std::string &expression) {
    std::string compact;
    for (char ch : expression) {
        if (ch != ',' && !std::isspace(static_cast<unsigned char>(ch))) { compact.push_back(ch); }
    }
    size_t pos = 0;
    std::function<std::string(bool)> parse_sequence = [&](bool in_group) {
        std::string result;
        while (pos < compact.size() && compact[pos] != ')') {
            std::string atom;
            if (compact[pos] == '(') {
                ++pos;
                atom = parse_sequence(true);
                if (pos >= compact.size() || compact[pos] != ')') {
                    throw std::invalid_argument("pipeline model layout has an unmatched '('");
                }
                ++pos;
            } else {
                atom.push_back(compact[pos++]);
            }
            int repetitions = 1;
            if (pos < compact.size() && compact[pos] == '*') {
                const size_t number_begin = ++pos;
                while (pos < compact.size() && std::isdigit(static_cast<unsigned char>(compact[pos]))) { ++pos; }
                if (number_begin == pos) {
                    throw std::invalid_argument("pipeline model layout repetition requires a positive integer");
                }
                const auto [ptr, ec]
                    = std::from_chars(compact.data() + number_begin, compact.data() + pos, repetitions);
                if (ec != std::errc() || ptr != compact.data() + pos || repetitions <= 0) {
                    throw std::invalid_argument("pipeline model layout repetition must be a positive integer");
                }
            }
            for (int i = 0; i < repetitions; ++i) { result += atom; }
        }
        if (!in_group && pos < compact.size()) {
            throw std::invalid_argument("pipeline model layout has an unmatched ')'");
        }
        return result;
    };
    const std::string expanded = parse_sequence(false);
    if (pos != compact.size()) { throw std::invalid_argument("invalid pipeline model layout expression"); }
    return expanded;
}
} // namespace

thread_local int pp_rank = 0;

PipelineLayout PipelineLayout::Uniform(int total_layers, int pp_size, int chunks_per_stage) {
    if (total_layers <= 0) { throw std::invalid_argument("pipeline layout requires total_layers > 0"); }
    if (pp_size <= 0) { throw std::invalid_argument("pipeline layout requires pp_size > 0"); }
    if (chunks_per_stage <= 0) { throw std::invalid_argument("pipeline layout requires chunks_per_stage > 0"); }

    PipelineLayout layout;
    layout.total_layers_ = total_layers;
    layout.num_stages_ = pp_size;
    layout.chunks_per_stage_ = chunks_per_stage;
    layout.ranges_.resize(pp_size);
    const int chunks = pp_size * chunks_per_stage;
    const int base = total_layers / chunks;
    const int remainder = total_layers % chunks;
    int start = 0;
    for (int global_chunk = 0; global_chunk < chunks; ++global_chunk) {
        const int count = base + (global_chunk < remainder ? 1 : 0);
        const int stage = global_chunk % pp_size;
        layout.chunk_stages_.push_back(stage);
        layout.chunk_local_indices_.push_back(layout.ranges_[stage].size());
        layout.chunk_ranges_.push_back({start, start + count});
        layout.ranges_[stage].push_back({start, start + count});
        start += count;
    }
    layout.embedding_stage_ = layout.chunk_stages_.front();
    layout.final_norm_stage_ = layout.chunk_stages_.back();
    layout.lm_head_stage_ = layout.chunk_stages_.back();
    return layout;
}

PipelineLayout PipelineLayout::Parse(int total_layers, int pp_size, const std::string &partition,
                                     int chunks_per_stage) {
    if (partition.empty()) { return Uniform(total_layers, pp_size, chunks_per_stage); }
    if (chunks_per_stage != 1) {
        throw std::invalid_argument("--pipeline_layer_partition is incompatible with "
                                    "--virtual_pipeline_parallel != 1");
    }
    if (total_layers <= 0 || pp_size <= 0) {
        throw std::invalid_argument("pipeline layout requires positive total_layers and pp_size");
    }

    std::vector<int> counts;
    size_t begin = 0;
    while (begin <= partition.size()) {
        const size_t comma = partition.find(',', begin);
        std::string_view token(partition.data() + begin,
                               (comma == std::string::npos ? partition.size() : comma) - begin);
        const size_t first = token.find_first_not_of(" \t");
        const size_t last = token.find_last_not_of(" \t");
        if (first == std::string_view::npos) {
            throw std::invalid_argument("pipeline layer partition contains an empty stage entry: '" + partition
                                        + "'");
        }
        token = token.substr(first, last - first + 1);
        int count = 0;
        const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), count);
        if (ec != std::errc() || ptr != token.data() + token.size() || count <= 0) {
            throw std::invalid_argument("pipeline layer partition entries must be positive integers; got '"
                                        + std::string(token) + "'");
        }
        counts.push_back(count);
        if (comma == std::string::npos) { break; }
        begin = comma + 1;
    }
    if (static_cast<int>(counts.size()) != pp_size) {
        throw std::invalid_argument("pipeline layer partition has " + std::to_string(counts.size())
                                    + " entries, but --pipeline_parallel is " + std::to_string(pp_size));
    }
    const int sum = std::accumulate(counts.begin(), counts.end(), 0);
    if (sum != total_layers) {
        throw std::invalid_argument("pipeline layer partition sums to " + std::to_string(sum)
                                    + " layers, but the model has " + std::to_string(total_layers));
    }

    PipelineLayout layout;
    layout.total_layers_ = total_layers;
    layout.num_stages_ = pp_size;
    layout.chunks_per_stage_ = 1;
    layout.ranges_.resize(pp_size);
    int start = 0;
    for (int stage = 0; stage < pp_size; ++stage) {
        layout.chunk_stages_.push_back(stage);
        layout.chunk_local_indices_.push_back(0);
        layout.chunk_ranges_.push_back({start, start + counts[stage]});
        layout.ranges_[stage].push_back({start, start + counts[stage]});
        start += counts[stage];
    }
    layout.embedding_stage_ = 0;
    layout.final_norm_stage_ = pp_size - 1;
    layout.lm_head_stage_ = pp_size - 1;
    return layout;
}

PipelineLayout PipelineLayout::FromLayerCosts(int total_layers, int pp_size, const std::string &layer_costs,
                                              int chunks_per_stage) {
    if (layer_costs.empty()) {
        throw std::invalid_argument("--pipeline_layer_costs must not be empty");
    }
    if (chunks_per_stage != 1) {
        throw std::invalid_argument("--pipeline_layer_costs is incompatible with "
                                    "--virtual_pipeline_parallel != 1");
    }
    if (total_layers <= 0 || pp_size <= 0 || pp_size > total_layers) {
        throw std::invalid_argument("automatic pipeline layout requires 0 < pp_size <= total_layers");
    }

    std::vector<double> costs;
    size_t begin = 0;
    while (begin <= layer_costs.size()) {
        const size_t comma = layer_costs.find(',', begin);
        std::string_view token(layer_costs.data() + begin,
                               (comma == std::string::npos ? layer_costs.size() : comma) - begin);
        const size_t first = token.find_first_not_of(" \t");
        const size_t last = token.find_last_not_of(" \t");
        if (first == std::string_view::npos) {
            throw std::invalid_argument("pipeline layer costs contain an empty entry: '" + layer_costs + "'");
        }
        token = token.substr(first, last - first + 1);
        double cost = 0.0;
        const auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), cost);
        if (ec != std::errc() || ptr != token.data() + token.size() || !std::isfinite(cost) || cost <= 0.0) {
            throw std::invalid_argument("pipeline layer costs must be finite positive numbers; got '"
                                        + std::string(token) + "'");
        }
        costs.push_back(cost);
        if (comma == std::string::npos) { break; }
        begin = comma + 1;
    }
    if (static_cast<int>(costs.size()) != total_layers) {
        throw std::invalid_argument("pipeline layer costs have " + std::to_string(costs.size())
                                    + " entries, but the model has " + std::to_string(total_layers) + " layers");
    }

    std::vector<double> prefix(total_layers + 1, 0.0);
    for (int layer = 0; layer < total_layers; ++layer) {
        prefix[layer + 1] = prefix[layer] + costs[layer];
        if (!std::isfinite(prefix[layer + 1])) {
            throw std::invalid_argument("pipeline layer costs have a non-finite total");
        }
    }
    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> best(pp_size + 1, std::vector<double>(total_layers + 1, infinity));
    std::vector<std::vector<int>> split(pp_size + 1, std::vector<int>(total_layers + 1, -1));
    best[0][0] = 0.0;
    for (int stages = 1; stages <= pp_size; ++stages) {
        for (int end = stages; end <= total_layers; ++end) {
            for (int start = stages - 1; start < end; ++start) {
                const double candidate = std::max(best[stages - 1][start], prefix[end] - prefix[start]);
                if (candidate < best[stages][end]) {
                    best[stages][end] = candidate;
                    split[stages][end] = start;
                }
            }
        }
    }

    std::vector<int> counts(pp_size);
    int end = total_layers;
    for (int stage = pp_size - 1; stage >= 0; --stage) {
        const int start = split[stage + 1][end];
        if (start < 0) { throw std::logic_error("failed to construct automatic pipeline layout"); }
        counts[stage] = end - start;
        end = start;
    }
    std::ostringstream partition;
    for (int stage = 0; stage < pp_size; ++stage) {
        if (stage > 0) { partition << ','; }
        partition << counts[stage];
    }
    return Parse(total_layers, pp_size, partition.str(), chunks_per_stage);
}

PipelineLayout PipelineLayout::FromChunkLayout(int total_layers, int pp_size, const std::string &chunk_layout) {
    if (total_layers <= 0 || pp_size <= 0 || chunk_layout.empty()) {
        throw std::invalid_argument("chunk pipeline layout requires positive layers/stages and a non-empty layout");
    }
    PipelineLayout layout;
    layout.total_layers_ = total_layers;
    layout.num_stages_ = pp_size;
    layout.ranges_.resize(pp_size);
    std::vector<int> chunks_per_stage(pp_size, 0);
    int layer = 0;
    size_t begin = 0;
    while (begin <= chunk_layout.size()) {
        const size_t comma = chunk_layout.find(',', begin);
        std::string_view token(chunk_layout.data() + begin,
                               (comma == std::string::npos ? chunk_layout.size() : comma) - begin);
        const size_t colon = token.find(':');
        int stage = -1;
        int count = -1;
        const auto stage_result
            = colon == std::string_view::npos
                ? std::from_chars(token.data(), token.data(), stage)
                : std::from_chars(token.data(), token.data() + colon, stage);
        const auto count_result
            = colon == std::string_view::npos
                ? std::from_chars(token.data(), token.data(), count)
                : std::from_chars(token.data() + colon + 1, token.data() + token.size(), count);
        if (colon == std::string_view::npos
            || stage_result.ec != std::errc() || stage_result.ptr != token.data() + colon
            || count_result.ec != std::errc() || count_result.ptr != token.data() + token.size()
            || stage < 0 || stage >= pp_size || count < 0) {
            throw std::invalid_argument("pipeline chunk layout entries must be STAGE:NON_NEGATIVE_LAYERS; got '"
                                        + std::string(token) + "'");
        }
        layout.chunk_stages_.push_back(stage);
        layout.chunk_local_indices_.push_back(chunks_per_stage[stage]++);
        layout.chunk_ranges_.push_back({layer, layer + count});
        layout.ranges_[stage].push_back({layer, layer + count});
        layer += count;
        if (comma == std::string::npos) { break; }
        begin = comma + 1;
    }
    if (layer != total_layers) {
        throw std::invalid_argument("pipeline chunk layout assigns " + std::to_string(layer)
                                    + " layers, but the model has " + std::to_string(total_layers));
    }
    if (layout.chunk_stages_.empty()
        || !std::all_of(chunks_per_stage.begin(), chunks_per_stage.end(),
                        [&](int count) { return count == chunks_per_stage.front() && count > 0; })) {
        throw std::invalid_argument("pipeline chunk layout must assign the same positive number of chunks to every stage");
    }
    layout.chunks_per_stage_ = chunks_per_stage.front();
    layout.embedding_stage_ = layout.chunk_stages_.front();
    layout.final_norm_stage_ = layout.chunk_stages_.back();
    layout.lm_head_stage_ = layout.chunk_stages_.back();
    return layout;
}

PipelineLayout PipelineLayout::FromMegatronLayout(int total_layers, int pp_size, const std::string &model_layout) {
    const std::string expanded = ExpandLayoutExpression(model_layout);
    std::vector<std::string> chunks(1);
    for (char symbol : expanded) {
        if (symbol == '|') {
            chunks.emplace_back();
        } else if (symbol == 'E' || symbol == 't' || symbol == 'N' || symbol == 'L') {
            chunks.back().push_back(symbol);
        } else {
            throw std::invalid_argument(std::string("invalid pipeline model layout symbol '") + symbol + "'");
        }
    }
    if (chunks.empty() || static_cast<int>(chunks.size()) % pp_size != 0) {
        throw std::invalid_argument("pipeline model layout chunk count must be divisible by --pipeline_parallel");
    }
    std::string flattened;
    for (const auto &chunk : chunks) { flattened += chunk; }
    if (flattened.empty() || std::count(flattened.begin(), flattened.end(), 'E') != 1 || flattened.front() != 'E') {
        throw std::invalid_argument("pipeline model layout must start with exactly one embedding symbol E");
    }
    if (std::count(flattened.begin(), flattened.end(), 'L') != 1 || flattened.back() != 'L') {
        throw std::invalid_argument("pipeline model layout must end with exactly one LM head symbol L");
    }
    const int norm_count = std::count(flattened.begin(), flattened.end(), 'N');
    if (norm_count > 1) { throw std::invalid_argument("pipeline model layout may contain at most one final norm N"); }
    if (norm_count == 1 && chunks.back().find('N') == std::string::npos) {
        throw std::invalid_argument("final norm N and LM head L must be in the same final logical chunk");
    }
    if (std::count(flattened.begin(), flattened.end(), 't') != total_layers) {
        throw std::invalid_argument("pipeline model layout Transformer count does not match the model layer count");
    }
    std::ostringstream chunk_layout;
    for (int global_chunk = 0; global_chunk < static_cast<int>(chunks.size()); ++global_chunk) {
        if (global_chunk > 0) { chunk_layout << ','; }
        chunk_layout << global_chunk % pp_size << ':' << std::count(chunks[global_chunk].begin(), chunks[global_chunk].end(), 't');
    }
    return FromChunkLayout(total_layers, pp_size, chunk_layout.str());
}

PipelineLayout ResolvePipelineLayout(int total_layers, int pp_size, int chunks_per_stage,
                                     const std::string &partition, const std::string &layer_costs,
                                     const std::string &chunk_layout, const std::string &model_layout) {
    const int configured = !partition.empty() + !layer_costs.empty() + !chunk_layout.empty() + !model_layout.empty();
    if (configured > 1) {
        throw std::invalid_argument("pipeline layout options are mutually exclusive");
    }
    PipelineLayout layout;
    if (!chunk_layout.empty()) { layout = PipelineLayout::FromChunkLayout(total_layers, pp_size, chunk_layout); }
    else if (!model_layout.empty()) { layout = PipelineLayout::FromMegatronLayout(total_layers, pp_size, model_layout); }
    else if (!layer_costs.empty()) {
        return PipelineLayout::FromLayerCosts(total_layers, pp_size, layer_costs, chunks_per_stage);
    } else {
        return PipelineLayout::Parse(total_layers, pp_size, partition, chunks_per_stage);
    }
    if (layout.chunks_per_stage() != chunks_per_stage) {
        throw std::invalid_argument("custom chunk layout requires --virtual_pipeline_parallel="
                                    + std::to_string(layout.chunks_per_stage()));
    }
    return layout;
}

bool PipelineLayout::is_first_stage(int stage) const {
    CheckStage(stage, num_stages_);
    return stage == 0;
}
bool PipelineLayout::is_last_stage(int stage) const {
    CheckStage(stage, num_stages_);
    return stage == num_stages_ - 1;
}
bool PipelineLayout::owns_embedding(int stage) const {
    CheckStage(stage, num_stages_);
    return stage == embedding_stage_;
}
bool PipelineLayout::owns_final_norm(int stage) const {
    CheckStage(stage, num_stages_);
    return stage == final_norm_stage_;
}
bool PipelineLayout::owns_lm_head(int stage) const {
    CheckStage(stage, num_stages_);
    return stage == lm_head_stage_;
}
int PipelineLayout::stage_for_chunk(int global_chunk) const {
    if (global_chunk < 0 || global_chunk >= num_global_chunks()) {
        throw std::out_of_range("pipeline global chunk is out of range");
    }
    return chunk_stages_[global_chunk];
}
int PipelineLayout::local_chunk_index(int global_chunk) const {
    if (global_chunk < 0 || global_chunk >= num_global_chunks()) {
        throw std::out_of_range("pipeline global chunk is out of range");
    }
    return chunk_local_indices_[global_chunk];
}
const std::pair<int, int> &PipelineLayout::chunk_range(int global_chunk) const {
    if (global_chunk < 0 || global_chunk >= num_global_chunks()) {
        throw std::out_of_range("pipeline global chunk is out of range");
    }
    return chunk_ranges_[global_chunk];
}
const std::vector<std::pair<int, int>> &PipelineLayout::layer_ranges(int stage) const {
    CheckStage(stage, num_stages_);
    return ranges_[stage];
}
int PipelineLayout::stage_for_layer(int layer) const {
    if (layer < 0 || layer >= total_layers_) {
        throw std::out_of_range("transformer layer " + std::to_string(layer) + " is outside [0, "
                                + std::to_string(total_layers_) + ")");
    }
    for (int stage = 0; stage < num_stages_; ++stage) {
        for (const auto &[start, end] : ranges_[stage]) {
            if (layer >= start && layer < end) { return stage; }
        }
    }
    throw std::logic_error("pipeline layout does not own transformer layer " + std::to_string(layer));
}
std::string PipelineLayout::ToString() const {
    std::ostringstream out;
    out << "Pipeline layout (" << total_layers_ << " layers, " << num_stages_ << " stages):";
    for (int stage = 0; stage < num_stages_; ++stage) {
        out << "\n  stage " << stage << ":";
        if (owns_embedding(stage)) { out << " embedding"; }
        for (const auto &[start, end] : ranges_[stage]) { out << " layers[" << start << "," << end << ")"; }
        if (owns_final_norm(stage)) { out << " final_norm"; }
        if (owns_lm_head(stage)) { out << " lm_head"; }
    }
    return out.str();
}

void SetPipelineLayout(std::optional<PipelineLayout> layout) { pipeline_layout = std::move(layout); }
bool HasPipelineLayout() { return pipeline_layout.has_value(); }
const PipelineLayout &GetPipelineLayout() {
    if (!pipeline_layout) { throw std::logic_error("pipeline layout has not been initialized"); }
    return *pipeline_layout;
}

void PipelineParallel::BuildPipelineStage(const std::vector<std::vector<int64_t>> &recv_shape, Device device,
                                          std::vector<std::shared_ptr<Module>> &&chunks) {
    pipeline_stage_ = std::make_shared<PipelineStage>(rank_, num_stages_, recv_shape, device, std::move(chunks));
}

void PipelineParallel::SetupSchedule(int num_micro_batches) {
    schedule_ = std::make_shared<PipelineSchedule>(pipeline_stage_, num_stages_, num_micro_batches);
}

float PipelineParallel::TrainStep(const std::vector<std::shared_ptr<Tensor>> &input,
                                  const std::vector<std::shared_ptr<Tensor>> &target,
                                  const std::shared_ptr<Optimizer> &optimizer, const std::shared_ptr<Module> &loss_fn,
                                  DataType dtype) {
    std::shared_ptr<Tensor> stage_input;
    std::shared_ptr<Tensor> stage_target = target[0];
    if (GetPipelineLayout().owns_embedding(rank_)) {
        stage_input = input[0];
    }

    return schedule_->Step(stage_input, stage_target, optimizer, loss_fn, dtype);
}

StageInfo PipelineParallel::GetStageInfo(int total_layers, int pp_size, int rank, int chunks_per_stage) {
    const PipelineLayout *layout = nullptr;
    PipelineLayout fallback;
    if (pipeline_layout && pipeline_layout->total_layers() == total_layers
        && pipeline_layout->num_stages() == pp_size
        && pipeline_layout->chunks_per_stage() == chunks_per_stage) {
        layout = &*pipeline_layout;
    } else {
        fallback = PipelineLayout::Uniform(total_layers, pp_size, chunks_per_stage);
        layout = &fallback;
    }
    return {layout->owns_embedding(rank), layout->owns_final_norm(rank) && layout->owns_lm_head(rank),
            layout->layer_ranges(rank)};
}

PipelineParallel::PipelineParallel(const std::shared_ptr<Module> module, int num_stages, int num_micro_batches,
                                   const std::vector<std::vector<int64_t>> &recv_shape, int pp_rank, Device device,
                                   int chunk_size)
    : num_stages_(num_stages), rank_(pp_rank) {
    modules_[kModuleName] = std::move(module);

    int stage_id = pp_rank;
    const auto &layout = GetPipelineLayout();

    std::vector<std::shared_ptr<Module>> chunks;
    for (int chunk_id = 0; chunk_id < chunk_size; ++chunk_id) {
        std::vector<std::shared_ptr<Module>> chunk_parts;
        if (chunk_id == 0 && layout.owns_embedding(stage_id)) {
            chunk_parts.push_back(module->mutable_module(kPPFirstStageName));
        }
        chunk_parts.push_back(module->mutable_module(kPPChunkNamePrefix + std::to_string(chunk_id)));
        if (chunk_id == chunk_size - 1 && layout.owns_final_norm(stage_id) && layout.owns_lm_head(stage_id)) {
            chunk_parts.push_back(module->mutable_module(kPPLastStageName));
        }
        chunks.push_back(std::make_shared<Sequential>(std::move(chunk_parts)));
    }

    BuildPipelineStage(recv_shape, device, std::move(chunks));

    SetupSchedule(num_micro_batches);
}

std::vector<std::shared_ptr<Module>> *PipelineParallel::mutable_chunks() { return pipeline_stage_->mutable_chunks(); }
} // namespace infini_train::nn::parallel
