#include "infini_train/include/checkpoint.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include "glog/logging.h"

#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
namespace {
constexpr uint32_t kCkptMagic = 0x54504B43; // CKPT
constexpr uint32_t kCkptVersion = 1;

uint32_t PeekMagic(const std::filesystem::path &path) {
    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open checkpoint file: " << path;
    uint32_t magic = 0;
    ifs.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    return magic;
}

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

std::string ExtractStringField(const std::string &content, const std::string &key, const std::string &fallback) {
    const auto token = std::string("\"") + key + "\"";
    const auto key_pos = content.find(token);
    if (key_pos == std::string::npos) {
        return fallback;
    }
    const auto colon_pos = content.find(':', key_pos);
    const auto first_quote = content.find('"', colon_pos + 1);
    const auto second_quote = content.find('"', first_quote + 1);
    if (first_quote == std::string::npos || second_quote == std::string::npos) {
        return fallback;
    }
    return content.substr(first_quote + 1, second_quote - first_quote - 1);
}

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

void Checkpoint::Save(const std::filesystem::path &checkpoint_dir, const nn::Module &model, const Optimizer &optimizer,
                      const TrainerState &state, const CheckpointOptions &options) {
    CHECK(options.format == "bin" || options.format == "pth")
        << "Unsupported checkpoint format: " << options.format;
    std::filesystem::create_directories(checkpoint_dir);
    LOG(ERROR) << "[CKPT] Save begin: dir=" << checkpoint_dir << ", format=" << options.format
               << ", global_step=" << state.global_step;

    const auto model_path = checkpoint_dir / (options.format == "pth" ? "model.pth" : "model.bin");
    if (options.format == "bin" && options.model_bin_writer) {
        options.model_bin_writer(model, model_path);
    } else {
        SaveStateDictBinary(model_path, model.StateDict());
    }

    if (options.save_optimizer_state) {
        auto opt_state = optimizer.StateDict();
        if (!opt_state.empty()) {
            const auto opt_path = checkpoint_dir / (options.format == "pth" ? "optimizer.pth" : "optimizer.bin");
            SaveStateDictBinary(opt_path, opt_state);
        }
    }

    SaveTrainerState(checkpoint_dir / "trainer_state.json", state);
    LOG(ERROR) << "[CKPT] Save done: dir=" << checkpoint_dir;
}

void Checkpoint::Load(const std::filesystem::path &checkpoint_dir, nn::Module *model, Optimizer *optimizer,
                      TrainerState *state, const CheckpointLoadOptions &options) {
    CHECK(model != nullptr);
    CHECK(state != nullptr);

    const std::string format = InferFormat(checkpoint_dir);
    const auto model_path = checkpoint_dir / (format == "pth" ? "model.pth" : "model.bin");
    LOG(ERROR) << "[CKPT] Load begin: dir=" << checkpoint_dir << ", format=" << format;
    LOG(ERROR) << "[CKPT] Loading model: " << model_path;
    if (format == "bin" && options.model_bin_loader) {
        const uint32_t magic = PeekMagic(model_path);
        if (magic == kCkptMagic) {
            LOG(ERROR) << "[CKPT] Model format detected: native checkpoint binary.";
            model->LoadStateDict(LoadStateDictBinary(model_path));
        } else {
            LOG(ERROR) << "[CKPT] Model format detected: external model.bin (magic=" << magic
                       << "), use model_bin_loader callback.";
            options.model_bin_loader(model, model_path);
        }
    } else {
        model->LoadStateDict(LoadStateDictBinary(model_path));
    }

    if (optimizer != nullptr && options.load_optimizer_state) {
        const auto opt_path = checkpoint_dir / (format == "pth" ? "optimizer.pth" : "optimizer.bin");
        if (std::filesystem::exists(opt_path)) {
            LOG(ERROR) << "[CKPT] Loading optimizer: " << opt_path;
            optimizer->LoadStateDict(LoadStateDictBinary(opt_path));
        } else {
            LOG(ERROR) << "[CKPT] Optimizer state not found, skip: " << opt_path;
        }
    } else if (optimizer == nullptr) {
        LOG(ERROR) << "[CKPT] No optimizer instance, skip optimizer state loading.";
    } else {
        LOG(ERROR) << "[CKPT] load_optimizer_state=false, skip optimizer state loading.";
    }

    *state = LoadTrainerState(checkpoint_dir / "trainer_state.json");
    LOG(ERROR) << "[CKPT] Load done: global_step=" << state->global_step << ", data_batch_idx="
               << state->data_batch_idx << ", data_batch_stride=" << state->data_batch_stride
               << ", best_loss=" << state->best_loss << ", last_lr=" << state->last_lr
               << ", optimizer_type=" << state->optimizer_type
               << ", topology(ddp,tp,sp,pp)=(" << state->ddp_size << "," << state->tp_size << ","
               << state->sp_size << "," << state->pp_size << ")";
}

void Checkpoint::SaveStateDictBinary(const std::filesystem::path &path,
                                     const std::unordered_map<std::string, std::shared_ptr<Tensor>> &state_dict) {
    std::ofstream ofs(path, std::ios::binary);
    CHECK(ofs.is_open()) << "Failed to open checkpoint file: " << path;

    uint32_t magic = kCkptMagic;
    uint32_t version = kCkptVersion;
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
        for (const auto dim : dims) {
            ofs.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
        }

        Tensor cpu_tensor = tensor->To(Device());
        uint64_t bytes = static_cast<uint64_t>(cpu_tensor.SizeInBytes());
        ofs.write(reinterpret_cast<const char *>(&bytes), sizeof(bytes));
        ofs.write(reinterpret_cast<const char *>(cpu_tensor.DataPtr()), static_cast<std::streamsize>(bytes));
    }
}

std::unordered_map<std::string, std::shared_ptr<Tensor>> Checkpoint::LoadStateDictBinary(const std::filesystem::path &path) {
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
        for (uint32_t d = 0; d < ndim; ++d) {
            ifs.read(reinterpret_cast<char *>(&dims[d]), sizeof(dims[d]));
        }

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
    ofs << "  \"global_step\": " << state.global_step << ",\n";
    ofs << "  \"data_batch_idx\": " << state.data_batch_idx << ",\n";
    ofs << "  \"data_batch_stride\": " << state.data_batch_stride << ",\n";
    ofs << "  \"best_loss\": " << state.best_loss << ",\n";
    ofs << "  \"last_lr\": " << state.last_lr << ",\n";
    ofs << "  \"optimizer_type\": \"" << state.optimizer_type << "\",\n";
    ofs << "  \"checkpoint_format\": \"" << state.checkpoint_format << "\",\n";
    ofs << "  \"ddp_size\": " << state.ddp_size << ",\n";
    ofs << "  \"tp_size\": " << state.tp_size << ",\n";
    ofs << "  \"sp_size\": " << state.sp_size << ",\n";
    ofs << "  \"pp_size\": " << state.pp_size << "\n";
    ofs << "}\n";
}

TrainerState Checkpoint::LoadTrainerState(const std::filesystem::path &path) {
    std::ifstream ifs(path);
    CHECK(ifs.is_open()) << "Failed to open trainer state file: " << path;
    const std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    TrainerState state;
    state.global_step = ExtractNumberField<int64_t>(content, "global_step", 0);
    state.data_batch_idx = ExtractNumberField<int64_t>(content, "data_batch_idx", 0);
    state.data_batch_stride = ExtractNumberField<int64_t>(content, "data_batch_stride", 1);
    state.best_loss = ExtractNumberField<float>(content, "best_loss", std::numeric_limits<float>::infinity());
    state.last_lr = ExtractNumberField<double>(content, "last_lr", 0.0);
    state.optimizer_type = ExtractStringField(content, "optimizer_type", "unknown");
    state.checkpoint_format = ExtractStringField(content, "checkpoint_format", "bin");
    state.ddp_size = ExtractNumberField<int>(content, "ddp_size", 1);
    state.tp_size = ExtractNumberField<int>(content, "tp_size", 1);
    state.sp_size = ExtractNumberField<int>(content, "sp_size", 1);
    state.pp_size = ExtractNumberField<int>(content, "pp_size", 1);
    return state;
}

std::string Checkpoint::InferFormat(const std::filesystem::path &checkpoint_dir) {
    if (std::filesystem::exists(checkpoint_dir / "model.pth")) {
        return "pth";
    }
    if (std::filesystem::exists(checkpoint_dir / "model.bin")) {
        return "bin";
    }
    LOG(FATAL) << "Failed to infer checkpoint format from path: " << checkpoint_dir;
    return "bin";
}

} // namespace infini_train
