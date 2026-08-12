#include "infini_train/include/checkpoint/checkpoint.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/checkpoint/save_planner.h"
#include "infini_train/include/lr_scheduler.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
namespace {
constexpr uint32_t kCkptMagic = 0x54504B43; // CKPT
constexpr uint32_t kCkptVersion = 1;
constexpr uint32_t kLRSchedulerMagic = 0x53524C53; // SLRS
constexpr uint32_t kLRSchedulerVersion = 1;

enum class LRSchedulerStateValueType : uint8_t {
    kInt64 = 1,
    kFloat = 2,
    kDouble = 3,
    kString = 4,
    kFloatVector = 5,
};

void WriteString(std::ofstream *ofs, const std::string &value) {
    uint32_t len = static_cast<uint32_t>(value.size());
    ofs->write(reinterpret_cast<const char *>(&len), sizeof(len));
    ofs->write(value.data(), len);
}

std::string ReadString(std::ifstream *ifs) {
    uint32_t len = 0;
    ifs->read(reinterpret_cast<char *>(&len), sizeof(len));
    std::string s(len, '\0');
    ifs->read(s.data(), len);
    return s;
}

void WriteLRSchedulerStateValue(std::ofstream *ofs, const StateValue &value) {
    if (std::holds_alternative<int64_t>(value)) {
        const auto type = LRSchedulerStateValueType::kInt64;
        const auto data = std::get<int64_t>(value);
        ofs->write(reinterpret_cast<const char *>(&type), sizeof(type));
        ofs->write(reinterpret_cast<const char *>(&data), sizeof(data));
    } else if (std::holds_alternative<float>(value)) {
        const auto type = LRSchedulerStateValueType::kFloat;
        const auto data = std::get<float>(value);
        ofs->write(reinterpret_cast<const char *>(&type), sizeof(type));
        ofs->write(reinterpret_cast<const char *>(&data), sizeof(data));
    } else if (std::holds_alternative<double>(value)) {
        const auto type = LRSchedulerStateValueType::kDouble;
        const auto data = std::get<double>(value);
        ofs->write(reinterpret_cast<const char *>(&type), sizeof(type));
        ofs->write(reinterpret_cast<const char *>(&data), sizeof(data));
    } else if (std::holds_alternative<std::string>(value)) {
        const auto type = LRSchedulerStateValueType::kString;
        ofs->write(reinterpret_cast<const char *>(&type), sizeof(type));
        WriteString(ofs, std::get<std::string>(value));
    } else if (std::holds_alternative<std::vector<float>>(value)) {
        const auto type = LRSchedulerStateValueType::kFloatVector;
        const auto &data = std::get<std::vector<float>>(value);
        const auto size = static_cast<uint64_t>(data.size());
        ofs->write(reinterpret_cast<const char *>(&type), sizeof(type));
        ofs->write(reinterpret_cast<const char *>(&size), sizeof(size));
        ofs->write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(size * sizeof(float)));
    } else {
        LOG(FATAL) << "Unsupported LR scheduler state value type.";
    }
}

StateValue ReadLRSchedulerStateValue(std::ifstream *ifs) {
    LRSchedulerStateValueType type{};
    ifs->read(reinterpret_cast<char *>(&type), sizeof(type));
    switch (type) {
    case LRSchedulerStateValueType::kInt64: {
        int64_t data = 0;
        ifs->read(reinterpret_cast<char *>(&data), sizeof(data));
        return data;
    }
    case LRSchedulerStateValueType::kFloat: {
        float data = 0.0f;
        ifs->read(reinterpret_cast<char *>(&data), sizeof(data));
        return data;
    }
    case LRSchedulerStateValueType::kDouble: {
        double data = 0.0;
        ifs->read(reinterpret_cast<char *>(&data), sizeof(data));
        return data;
    }
    case LRSchedulerStateValueType::kString:
        return ReadString(ifs);
    case LRSchedulerStateValueType::kFloatVector: {
        uint64_t size = 0;
        ifs->read(reinterpret_cast<char *>(&size), sizeof(size));
        std::vector<float> data(size);
        ifs->read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size * sizeof(float)));
        return data;
    }
    default:
        LOG(FATAL) << "Unsupported LR scheduler state value type: " << static_cast<int>(type);
    }
    return int64_t{0};
}

void SaveLRSchedulerState(const std::filesystem::path &path, const LRSchedulerStateDict &state_dict) {
    std::ofstream ofs(path, std::ios::binary);
    CHECK(ofs.is_open()) << "Failed to open LR scheduler checkpoint file: " << path;

    const uint32_t magic = kLRSchedulerMagic;
    const uint32_t version = kLRSchedulerVersion;
    const uint32_t count = static_cast<uint32_t>(state_dict.size());
    ofs.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    ofs.write(reinterpret_cast<const char *>(&version), sizeof(version));
    ofs.write(reinterpret_cast<const char *>(&count), sizeof(count));

    for (const auto &[name, value] : state_dict) {
        WriteString(&ofs, name);
        WriteLRSchedulerStateValue(&ofs, value);
    }
}

LRSchedulerStateDict LoadLRSchedulerState(const std::filesystem::path &path) {
    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open LR scheduler checkpoint file: " << path;

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t count = 0;
    ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char *>(&version), sizeof(version));
    ifs.read(reinterpret_cast<char *>(&count), sizeof(count));

    CHECK_EQ(magic, kLRSchedulerMagic) << "Invalid LR scheduler checkpoint magic: " << path;
    CHECK_EQ(version, kLRSchedulerVersion) << "Unsupported LR scheduler checkpoint version: " << path;

    LRSchedulerStateDict state;
    for (uint32_t i = 0; i < count; ++i) {
        auto name = ReadString(&ifs);
        state.emplace(std::move(name), ReadLRSchedulerStateValue(&ifs));
    }
    return state;
}

// TODO: This is a hand-rolled JSON field extractor. Replace with a proper JSON library (e.g., nlohmann/json) once
// available in the project dependencies.
template <typename T> T ExtractNumberField(const std::string &content, const std::string &key, T fallback) {
    const auto token = std::string("\"") + key + "\"";
    const auto key_pos = content.find(token);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    const auto colon_pos = content.find(':', key_pos);
    if (colon_pos == std::string::npos) {
        return fallback;
    }
    size_t value_start = colon_pos + 1;
    while (value_start < content.size() && (content[value_start] == ' ' || content[value_start] == '\n')) {
        ++value_start;
    }
    size_t value_end = value_start;
    while (value_end < content.size() && content[value_end] != ',' && content[value_end] != '\n'
           && content[value_end] != '}') {
        ++value_end;
    }
    std::stringstream ss(content.substr(value_start, value_end - value_start));
    T value = fallback;
    ss >> value;
    if (ss.fail()) {
        return fallback;
    }
    return value;
}
} // namespace

void Checkpoint::Save(const std::filesystem::path &checkpoint_dir, const nn::Module &model, const Optimizer *optimizer,
                      const TrainerState &state, const LRScheduler *lr_scheduler) {
    std::filesystem::create_directories(checkpoint_dir);
    LOG(INFO) << "[CKPT] Save begin: dir=" << checkpoint_dir << ", global_step=" << state.global_step;

    const auto model_path = checkpoint_dir / ("model.ckpt");

    SaveStateDict(model_path, model.StateDict());

    if (optimizer != nullptr) {
        auto opt_state = optimizer->StateDict();
        if (!opt_state.empty()) {
            const auto opt_path = checkpoint_dir / "optimizer.ckpt";
            SaveStateDict(opt_path, opt_state);
        }
    }

    if (lr_scheduler != nullptr) {
        SaveLRSchedulerState(checkpoint_dir / "lr_scheduler.ckpt", lr_scheduler->StateDict());
    }

    SaveTrainerState(checkpoint_dir / "trainer_state.json", state);
    LOG(ERROR) << "[CKPT] Save done: dir=" << checkpoint_dir;
}

void Checkpoint::Load(const std::filesystem::path &checkpoint_dir, nn::Module &model, Optimizer *optimizer,
                      TrainerState &state, LRScheduler *lr_scheduler) {
    const auto model_path = checkpoint_dir / "model.ckpt";
    LOG(INFO) << "[CKPT] Loading model: " << model_path;

    model.LoadStateDict(LoadStateDict(model_path));

    if (optimizer != nullptr) {
        const auto opt_path = checkpoint_dir / "optimizer.ckpt";
        if (std::filesystem::exists(opt_path)) {
            LOG(INFO) << "[CKPT] Loading optimizer: " << opt_path;
            optimizer->LoadStateDict(LoadStateDict(opt_path));
        } else {
            LOG(FATAL) << "Optimizer checkpoint not found at: " << opt_path;
        }
    }

    state = LoadTrainerState(checkpoint_dir / "trainer_state.json");

    if (lr_scheduler != nullptr) {
        const auto lr_scheduler_path = checkpoint_dir / "lr_scheduler.ckpt";
        if (std::filesystem::exists(lr_scheduler_path)) {
            LOG(INFO) << "[CKPT] Loading LR scheduler: " << lr_scheduler_path;
            lr_scheduler->LoadStateDict(LoadLRSchedulerState(lr_scheduler_path));
        } else {
            LOG(WARNING) << "[CKPT] LR scheduler checkpoint not found at: " << lr_scheduler_path
                         << ". Keeping the initialized scheduler state.";
        }
    }

    LOG(ERROR) << "[CKPT] Load done: global_step=" << state.global_step
               << ", consumed_train_samples=" << state.consumed_train_samples << ", topology(ddp,tp,sp,pp)=("
               << state.ddp_size << "," << state.tp_size << "," << state.sp_size << "," << state.pp_size << ")";
}

Checkpoint::SavedTensorLocations
Checkpoint::SaveStateDict(const std::filesystem::path &path,
                          const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) {
    std::ofstream ofs(path, std::ios::binary);
    CHECK(ofs.is_open()) << "Failed to open checkpoint file: " << path;

    uint32_t magic = kCkptMagic;
    uint32_t version = kCkptVersion;
    SavedTensorLocations locations;
    uint32_t count = static_cast<uint32_t>(state_dict.size());
    ofs.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    ofs.write(reinterpret_cast<const char *>(&version), sizeof(version));
    ofs.write(reinterpret_cast<const char *>(&count), sizeof(count));

    for (const auto &[name, tensor] : state_dict) {
        WriteString(&ofs, name);

        const int8_t dtype = static_cast<int8_t>(tensor->Dtype());
        ofs.write(reinterpret_cast<const char *>(&dtype), sizeof(dtype));

        const auto &dims = tensor->Dims();
        uint32_t ndim = static_cast<uint32_t>(dims.size());
        ofs.write(reinterpret_cast<const char *>(&ndim), sizeof(ndim));
        for (const auto dim : dims) { ofs.write(reinterpret_cast<const char *>(&dim), sizeof(dim)); }

        Tensor cpu_tensor = tensor->To(Device());
        uint64_t bytes = static_cast<uint64_t>(cpu_tensor.SizeInBytes());
        ofs.write(reinterpret_cast<const char *>(&bytes), sizeof(bytes));
        const auto data_offset = ofs.tellp();
        CHECK(data_offset != std::streampos(-1)) << "Failed to record tensor offset for " << name;
        locations.emplace(
            name, SavedTensorLocation{.data_offset = static_cast<uint64_t>(static_cast<std::streamoff>(data_offset)),
                                      .byte_size = bytes});
        ofs.write(reinterpret_cast<const char *>(cpu_tensor.DataPtr()), static_cast<std::streamsize>(bytes));
    }
    return locations;
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> Checkpoint::LoadStateDict(const std::filesystem::path &path) {
    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open checkpoint file: " << path;

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t count = 0;
    ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char *>(&version), sizeof(version));
    ifs.read(reinterpret_cast<char *>(&count), sizeof(count));

    CHECK_EQ(magic, kCkptMagic) << "Invalid checkpoint magic: " << path;
    CHECK_EQ(version, kCkptVersion) << "Unsupported checkpoint version: " << path;

    std::unordered_map<std::string, std::shared_ptr<Tensor>> state;
    for (uint32_t i = 0; i < count; ++i) {
        const std::string name = ReadString(&ifs);

        int8_t dtype_raw = 0;
        ifs.read(reinterpret_cast<char *>(&dtype_raw), sizeof(dtype_raw));
        DataType dtype = static_cast<DataType>(dtype_raw);

        uint32_t ndim = 0;
        ifs.read(reinterpret_cast<char *>(&ndim), sizeof(ndim));
        std::vector<int64_t> dims(ndim);
        for (uint32_t d = 0; d < ndim; ++d) { ifs.read(reinterpret_cast<char *>(&dims[d]), sizeof(dims[d])); }

        uint64_t bytes = 0;
        ifs.read(reinterpret_cast<char *>(&bytes), sizeof(bytes));

        auto tensor = std::make_shared<Tensor>(dims, dtype, Device());
        CHECK_EQ(bytes, tensor->SizeInBytes()) << "Tensor bytes mismatch for key: " << name;
        ifs.read(reinterpret_cast<char *>(tensor->DataPtr()), static_cast<std::streamsize>(bytes));
        state.emplace(name, tensor);
    }

    return state;
}

void Checkpoint::SaveTrainerState(const std::filesystem::path &path, const TrainerState &state) {
    std::ofstream ofs(path);
    CHECK(ofs.is_open()) << "Failed to open trainer state file: " << path;
    ofs << "{\n";
    ofs << "  \"n_layer\": " << state.n_layer << ",\n";
    ofs << "  \"n_head\": " << state.n_head << ",\n";
    ofs << "  \"n_kv_head\": " << state.n_kv_head << ",\n";
    ofs << "  \"n_embd\": " << state.n_embd << ",\n";
    ofs << "  \"vocab_size\": " << state.vocab_size << ",\n";
    ofs << "  \"global_step\": " << state.global_step << ",\n";
    ofs << "  \"consumed_train_samples\": " << state.consumed_train_samples << ",\n";
    ofs << "  \"ddp_size\": " << state.ddp_size << ",\n";
    ofs << "  \"tp_size\": " << state.tp_size << ",\n";
    ofs << "  \"sp_size\": " << state.sp_size << ",\n";
    ofs << "  \"pp_size\": " << state.pp_size << "\n";
    ofs << "}\n";
}

// TODO(jym): Add TrainerState JSON version compatibility, referencing PyTorch's checkpoint versioning.
TrainerState Checkpoint::LoadTrainerState(const std::filesystem::path &path) {
    std::ifstream ifs(path);
    CHECK(ifs.is_open()) << "Failed to open trainer state file: " << path;
    const std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    TrainerState state;
    state.n_layer = ExtractNumberField<int64_t>(content, "n_layer", 0);
    state.n_head = ExtractNumberField<int64_t>(content, "n_head", 0);
    state.n_kv_head = ExtractNumberField<int64_t>(content, "n_kv_head", 0);
    state.n_embd = ExtractNumberField<int64_t>(content, "n_embd", 0);
    state.vocab_size = ExtractNumberField<int64_t>(content, "vocab_size", 0);
    state.global_step = ExtractNumberField<int64_t>(content, "global_step", 0);
    state.consumed_train_samples = ExtractNumberField<int64_t>(content, "consumed_train_samples", 0);
    state.ddp_size = ExtractNumberField<int>(content, "ddp_size", 1);
    state.tp_size = ExtractNumberField<int>(content, "tp_size", 1);
    state.sp_size = ExtractNumberField<int>(content, "sp_size", 1);
    state.pp_size = ExtractNumberField<int>(content, "pp_size", 1);
    return state;
}

void Checkpoint::SaveTrainerStateFile(const std::filesystem::path &path, const TrainerState &state) {
    SaveTrainerState(path, state);
}

TrainerState Checkpoint::LoadTrainerStateFile(const std::filesystem::path &path) { return LoadTrainerState(path); }

void Checkpoint::SaveLRSchedulerStateFile(const std::filesystem::path &path, const LRSchedulerStateDict &state_dict) {
    SaveLRSchedulerState(path, state_dict);
}

LRSchedulerStateDict Checkpoint::LoadLRSchedulerStateFile(const std::filesystem::path &path) {
    return LoadLRSchedulerState(path);
}

void Checkpoint::SaveStateDictFile(const std::filesystem::path &path,
                                   const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) {
    SaveStateDict(path, state_dict);
}

std::unordered_map<std::string, std::shared_ptr<Tensor>>
Checkpoint::LoadStateDictFile(const std::filesystem::path &path) {
    return LoadStateDict(path);
}

// -----------------------------------------------------------------------------
// Save local shards and a temporary rank manifest from a ShardedStateDict.
// -----------------------------------------------------------------------------

static std::string DataTypeToString(DataType dt) {
    auto it = kDataTypeToDesc.find(dt);
    if (it != kDataTypeToDesc.end()) {
        return it->second;
    }
    return "fp32";
}

void Checkpoint::SaveSharded(const std::filesystem::path &checkpoint_dir,
                             const checkpoint::ShardedStateDict &sharded_sd,
                             const std::vector<checkpoint::WriteItem> &write_items,
                             const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict,
                             const std::unordered_map<std::string, std::shared_ptr<Tensor>> &optimizer_state,
                             const TrainerState &state, int global_rank) {
    std::filesystem::create_directories(checkpoint_dir);
    LOG(INFO) << "[CKPT] SaveSharded begin: dir=" << checkpoint_dir << ", global_step=" << state.global_step
              << ", rank=" << global_rank;

    SavedTensorLocations model_file_index;
    SavedTensorLocations optimizer_file_index;

    // Save model tensors separately from optimizer tensors.
    {
        std::unordered_map<std::string, std::shared_ptr<Tensor>> filtered_sd;
        for (const auto &[key, info] : sharded_sd.tensors) {
            // Optimizer tensors are serialized separately.
            if (key.starts_with("adam.")) {
                continue;
            }
            // Match metadata keys to the local tensor payloads.
            const auto &local_key = info.local_key.empty() ? key : info.local_key;
            auto it = state_dict.find(local_key);
            if (it != state_dict.end()) {
                filtered_sd.emplace(key, it->second);
            }
        }
        if (!filtered_sd.empty()) {
            model_file_index = SaveStateDict(checkpoint_dir / "model.ckpt", filtered_sd);
        }
    }

    // Save the rank-local optimizer state.
    if (!optimizer_state.empty()) {
        optimizer_file_index = SaveStateDict(checkpoint_dir / "optimizer.ckpt", optimizer_state);
    }

    // Write the temporary rank manifest.
    {
        std::ofstream ofs(checkpoint_dir / "metadata.json");
        CHECK(ofs.is_open()) << "Failed to open metadata.json: " << checkpoint_dir / "metadata.json";

        ofs << "{\n";
        ofs << "  \"version\": 3,\n";
        ofs << "  \"format\": \"infinitrain_sharded\",\n";
        ofs << "  \"iteration\": " << state.global_step << ",\n";
        ofs << "  \"parallel_config\": {\n";
        ofs << "    \"tp_size\": " << state.tp_size << ",\n";
        ofs << "    \"pp_size\": " << state.pp_size << ",\n";
        ofs << "    \"dp_size\": " << state.ddp_size << ",\n";
        ofs << "    \"sp_size\": " << state.sp_size << "\n";
        ofs << "  },\n";
        ofs << "  \"model_config\": {\n";
        ofs << "    \"n_layer\": " << state.n_layer << ",\n";
        ofs << "    \"n_head\": " << state.n_head << ",\n";
        ofs << "    \"n_kv_head\": " << state.n_kv_head << ",\n";
        ofs << "    \"n_embd\": " << state.n_embd << ",\n";
        ofs << "    \"vocab_size\": " << state.vocab_size << "\n";
        ofs << "  },\n";
        ofs << "  \"tensors\": [\n";

        std::vector<const checkpoint::WriteItem *> emitted_items;
        for (const auto &item : write_items) {
            if (sharded_sd.tensors.contains(item.key)) {
                emitted_items.push_back(&item);
            }
        }
        int dp_rank = 0, tp_rank = 0, pp_rank = 0;
        nn::parallel::global::GetCoordOf(global_rank, dp_rank, tp_rank, pp_rank);
        for (size_t i = 0; i < emitted_items.size(); ++i) {
            const auto &item = *emitted_items[i];
            const auto it = sharded_sd.tensors.find(item.key);
            const auto &file_index = item.filename == "optimizer.ckpt" ? optimizer_file_index : model_file_index;
            const auto storage_it = file_index.find(item.key);
            CHECK(storage_it != file_index.end()) << "Missing stored tensor metadata for " << item.key;
            const auto &storage = storage_it->second;
            CHECK_EQ(storage.byte_size, item.byte_size);

            ofs << "    {\n";
            ofs << "      \"key\": \"" << item.key << "\",\n";
            ofs << "      \"dtype\": \"" << DataTypeToString(item.dtype) << "\",\n";

            // global_shape
            ofs << "      \"global_shape\": [";
            const auto &gs = it->second.global_shape;
            for (size_t j = 0; j < gs.size(); ++j) { ofs << gs[j] << (j + 1 < gs.size() ? ", " : ""); }
            ofs << "],\n";

            ofs << "      \"local_shape\": [";
            const auto &ls = it->second.local_shape;
            for (size_t j = 0; j < ls.size(); ++j) { ofs << ls[j] << (j + 1 < ls.size() ? ", " : ""); }
            ofs << "],\n";

            ofs << "      \"global_offset\": [";
            for (size_t j = 0; j < it->second.global_offset.size(); ++j) {
                ofs << it->second.global_offset[j] << (j + 1 < it->second.global_offset.size() ? ", " : "");
            }
            ofs << "],\n";
            ofs << "      \"axis_fragmentations\": [";
            for (size_t j = 0; j < it->second.axis_fragmentations.size(); ++j) {
                ofs << it->second.axis_fragmentations[j] << (j + 1 < it->second.axis_fragmentations.size() ? ", " : "");
            }
            ofs << "],\n";
            auto write_segments = [&](const char *name, auto member) {
                ofs << "      \"" << name << "\": [";
                for (size_t j = 0; j < it->second.segments.size(); ++j) {
                    ofs << it->second.segments[j].*member << (j + 1 < it->second.segments.size() ? ", " : "");
                }
                ofs << "],\n";
            };
            write_segments("segment_global_offsets", &checkpoint::ShardSegment::global_offset);
            write_segments("segment_local_offsets", &checkpoint::ShardSegment::local_offset);
            write_segments("segment_lengths", &checkpoint::ShardSegment::length);

            ofs << "      \"file\": \"" << item.filename << "\",\n";
            ofs << "      \"offset\": " << storage.data_offset << ",\n";
            ofs << "      \"byte_size\": " << item.byte_size << ",\n";
            ofs << "      \"pp_rank\": " << pp_rank << ",\n";
            ofs << "      \"stored_on_ranks\": [" << global_rank << "]\n";
            ofs << "    }";
            if (i + 1 < emitted_items.size()) {
                ofs << ",";
            }
            ofs << "\n";
        }

        ofs << "  ]\n";
        ofs << "}\n";

        LOG(INFO) << "[CKPT] metadata.json written";
    }

    LOG(ERROR) << "[CKPT] SaveSharded done: dir=" << checkpoint_dir;
}

// Load one manifest or aggregate writer manifests while finalizing a checkpoint.
static std::string ExtractJsonString(const std::string &obj, const std::string &key) {
    auto token = std::string("\"") + key + "\"";
    auto pos = obj.find(token);
    if (pos == std::string::npos) {
        return "";
    }
    auto q1 = obj.find('"', pos + token.size());
    if (q1 == std::string::npos) {
        return "";
    }
    auto q2 = obj.find('"', q1 + 1);
    if (q2 == std::string::npos) {
        return "";
    }
    return obj.substr(q1 + 1, q2 - q1 - 1);
}

static Checkpoint::CheckpointMetadata LoadSingleMetadata(const std::filesystem::path &checkpoint_dir) {
    Checkpoint::CheckpointMetadata meta;
    auto metadata_path = checkpoint_dir / "metadata.json";
    if (!std::filesystem::exists(metadata_path)) {
        meta.has_metadata = false;
        return meta;
    }

    std::ifstream ifs(metadata_path);
    CHECK(ifs.is_open()) << "Failed to open metadata.json: " << metadata_path;
    const std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    meta.has_metadata = true;
    meta.version = ExtractNumberField<int>(content, "version", 0);
    meta.iteration = ExtractNumberField<int64_t>(content, "iteration", 0);
    meta.parallel_config.tp_size = ExtractNumberField<int>(content, "tp_size", 1);
    meta.parallel_config.pp_size = ExtractNumberField<int>(content, "pp_size", 1);
    meta.parallel_config.dp_size = ExtractNumberField<int>(content, "dp_size", 1);
    meta.parallel_config.sp_size = ExtractNumberField<int>(content, "sp_size", 1);

    // Locate the tensors array.
    auto tensors_key = content.find("\"tensors\"");
    if (tensors_key == std::string::npos) {
        return meta;
    }

    auto array_start = content.find('[', tensors_key);
    if (array_start == std::string::npos) {
        return meta;
    }

    int depth = 1;
    size_t pos = array_start + 1;
    while (pos < content.size() && depth > 0) {
        if (content[pos] == '[') {
            ++depth;
        } else if (content[pos] == ']') {
            --depth;
        }
        ++pos;
    }
    std::string tensor_block = content.substr(array_start + 1, pos - array_start - 2);

    // Parse each tensor object.
    size_t obj_pos = 0;
    while ((obj_pos = tensor_block.find('{', obj_pos)) != std::string::npos) {
        int object_depth = 1;
        size_t obj_end = obj_pos + 1;
        while (obj_end < tensor_block.size() && object_depth > 0) {
            if (tensor_block[obj_end] == '{') {
                ++object_depth;
            }
            if (tensor_block[obj_end] == '}') {
                --object_depth;
            }
            ++obj_end;
        }
        if (obj_end > 0) {
            --obj_end;
        }
        if (obj_end == std::string::npos) {
            break;
        }

        std::string obj = tensor_block.substr(obj_pos, obj_end - obj_pos + 1);

        Checkpoint::CheckpointMetadata::TensorEntry entry;
        entry.key = ExtractJsonString(obj, "key");
        entry.file = ExtractJsonString(obj, "file");
        entry.dtype_str = ExtractJsonString(obj, "dtype");
        entry.offset = ExtractNumberField<uint64_t>(obj, "offset", 0);
        entry.byte_size = ExtractNumberField<uint64_t>(obj, "byte_size", 0);
        entry.pp_rank = ExtractNumberField<int>(obj, "pp_rank", 0);

        // global_shape: [x, y, z]
        auto gs_pos = obj.find("\"global_shape\"");
        if (gs_pos != std::string::npos) {
            auto b1 = obj.find('[', gs_pos);
            auto b2 = obj.find(']', b1);
            if (b1 != std::string::npos && b2 != std::string::npos) {
                std::string gs = obj.substr(b1 + 1, b2 - b1 - 1);
                std::stringstream ss(gs);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    try {
                        entry.global_shape.push_back(std::stoll(tok));
                    } catch (...) {}
                }
            }
        }

        auto ls_pos = obj.find("\"local_shape\"");
        if (ls_pos != std::string::npos) {
            auto b1 = obj.find('[', ls_pos);
            auto b2 = obj.find(']', b1);
            std::stringstream ss(obj.substr(b1 + 1, b2 - b1 - 1));
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                try {
                    entry.local_shape.push_back(std::stoll(tok));
                } catch (...) {}
            }
        }

        auto offset_pos = obj.find("\"global_offset\"");
        if (offset_pos != std::string::npos) {
            auto b1 = obj.find('[', offset_pos);
            auto b2 = obj.find(']', b1);
            std::stringstream ss(obj.substr(b1 + 1, b2 - b1 - 1));
            std::string token;
            while (std::getline(ss, token, ',')) {
                try {
                    entry.global_offset.push_back(std::stoll(token));
                } catch (...) {}
            }
        }

        auto fragments_pos = obj.find("\"axis_fragmentations\"");
        if (fragments_pos != std::string::npos) {
            auto b1 = obj.find('[', fragments_pos);
            auto b2 = obj.find(']', b1);
            std::stringstream ss(obj.substr(b1 + 1, b2 - b1 - 1));
            std::string token;
            while (std::getline(ss, token, ',')) {
                try {
                    entry.axis_fragmentations.push_back(std::stoi(token));
                } catch (...) {}
            }
        }

        auto extract_int64_array = [&](const char *name) {
            std::vector<int64_t> values;
            auto field_pos = obj.find(std::string("\"") + name + "\"");
            if (field_pos == std::string::npos) {
                return values;
            }
            auto b1 = obj.find('[', field_pos);
            auto b2 = obj.find(']', b1);
            if (b1 == std::string::npos || b2 == std::string::npos) {
                return values;
            }
            std::stringstream ss(obj.substr(b1 + 1, b2 - b1 - 1));
            std::string token;
            while (std::getline(ss, token, ',')) {
                try {
                    values.push_back(std::stoll(token));
                } catch (...) {}
            }
            return values;
        };
        const auto segment_global_offsets = extract_int64_array("segment_global_offsets");
        const auto segment_local_offsets = extract_int64_array("segment_local_offsets");
        const auto segment_lengths = extract_int64_array("segment_lengths");
        CHECK_EQ(segment_global_offsets.size(), segment_local_offsets.size());
        CHECK_EQ(segment_global_offsets.size(), segment_lengths.size());
        for (size_t i = 0; i < segment_lengths.size(); ++i) {
            entry.segments.push_back({.global_offset = segment_global_offsets[i],
                                      .local_offset = segment_local_offsets[i],
                                      .length = segment_lengths[i]});
        }

        auto ranks_pos = obj.find("\"stored_on_ranks\"");
        if (ranks_pos != std::string::npos) {
            auto b1 = obj.find('[', ranks_pos);
            auto b2 = obj.find(']', b1);
            std::stringstream ss(obj.substr(b1 + 1, b2 - b1 - 1));
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                try {
                    entry.stored_on_ranks.push_back(std::stoi(tok));
                } catch (...) {}
            }
        }

        meta.tensors.push_back(std::move(entry));
        obj_pos = obj_end + 1;
    }

    LOG(INFO) << "[CKPT] Loaded metadata.json: " << meta.tensors.size() << " tensors, iteration=" << meta.iteration;
    return meta;
}

Checkpoint::CheckpointMetadata Checkpoint::LoadMetadata(const std::filesystem::path &checkpoint_dir) {
    if (std::filesystem::exists(checkpoint_dir / "metadata.json")) {
        return LoadSingleMetadata(checkpoint_dir);
    }

    CheckpointMetadata merged;
    for (const auto &entry : std::filesystem::directory_iterator(checkpoint_dir)) {
        if (!entry.is_directory() || !entry.path().filename().string().starts_with("rank_")
            || !std::filesystem::exists(entry.path() / "metadata.json")) {
            continue;
        }
        auto rank_metadata = LoadSingleMetadata(entry.path());
        if (!rank_metadata.has_metadata) {
            continue;
        }
        if (!merged.has_metadata) {
            merged = rank_metadata;
            merged.tensors.clear();
        }
        for (auto &tensor : rank_metadata.tensors) {
            tensor.file = (entry.path().filename() / tensor.file).generic_string();
            merged.tensors.push_back(std::move(tensor));
        }
    }
    LOG(INFO) << "[CKPT] Aggregated " << merged.tensors.size() << " tensor shards from rank manifests";
    return merged;
}

void Checkpoint::SaveMetadataFile(const std::filesystem::path &path, const CheckpointMetadata &metadata) {
    std::ofstream ofs(path);
    CHECK(ofs.is_open()) << "Failed to write checkpoint metadata: " << path;
    ofs << "{\n";
    ofs << "  \"version\": 3,\n";
    ofs << "  \"format\": \"infinitrain_sharded\",\n";
    ofs << "  \"iteration\": " << metadata.iteration << ",\n";
    ofs << "  \"parallel_config\": {\n";
    ofs << "    \"tp_size\": " << metadata.parallel_config.tp_size << ",\n";
    ofs << "    \"pp_size\": " << metadata.parallel_config.pp_size << ",\n";
    ofs << "    \"dp_size\": " << metadata.parallel_config.dp_size << ",\n";
    ofs << "    \"sp_size\": " << metadata.parallel_config.sp_size << "\n";
    ofs << "  },\n";
    ofs << "  \"tensors\": [\n";
    for (size_t i = 0; i < metadata.tensors.size(); ++i) {
        const auto &tensor = metadata.tensors[i];
        ofs << "    {\n";
        ofs << "      \"key\": \"" << tensor.key << "\",\n";
        ofs << "      \"dtype\": \"" << tensor.dtype_str << "\",\n";
        auto write_shape = [&](const char *name, const std::vector<int64_t> &shape) {
            ofs << "      \"" << name << "\": [";
            for (size_t d = 0; d < shape.size(); ++d) { ofs << shape[d] << (d + 1 < shape.size() ? ", " : ""); }
            ofs << "],\n";
        };
        write_shape("global_shape", tensor.global_shape);
        write_shape("local_shape", tensor.local_shape);
        write_shape("global_offset", tensor.global_offset);
        ofs << "      \"axis_fragmentations\": [";
        for (size_t d = 0; d < tensor.axis_fragmentations.size(); ++d) {
            ofs << tensor.axis_fragmentations[d] << (d + 1 < tensor.axis_fragmentations.size() ? ", " : "");
        }
        ofs << "],\n";
        auto write_segments = [&](const char *name, auto member) {
            ofs << "      \"" << name << "\": [";
            for (size_t d = 0; d < tensor.segments.size(); ++d) {
                ofs << tensor.segments[d].*member << (d + 1 < tensor.segments.size() ? ", " : "");
            }
            ofs << "],\n";
        };
        write_segments("segment_global_offsets", &checkpoint::ShardSegment::global_offset);
        write_segments("segment_local_offsets", &checkpoint::ShardSegment::local_offset);
        write_segments("segment_lengths", &checkpoint::ShardSegment::length);
        ofs << "      \"file\": \"" << tensor.file << "\",\n";
        ofs << "      \"offset\": " << tensor.offset << ",\n";
        ofs << "      \"byte_size\": " << tensor.byte_size << ",\n";
        ofs << "      \"pp_rank\": " << tensor.pp_rank << ",\n";
        ofs << "      \"stored_on_ranks\": [";
        for (size_t r = 0; r < tensor.stored_on_ranks.size(); ++r) {
            ofs << tensor.stored_on_ranks[r] << (r + 1 < tensor.stored_on_ranks.size() ? ", " : "");
        }
        ofs << "]\n";
        ofs << "    }" << (i + 1 < metadata.tensors.size() ? "," : "") << "\n";
    }
    ofs << "  ]\n";
    ofs << "}\n";
    CHECK(ofs.good()) << "Failed while writing checkpoint metadata: " << path;
}
} // namespace infini_train
