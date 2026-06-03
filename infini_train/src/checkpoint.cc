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

void Checkpoint::Save(const std::filesystem::path &checkpoint_dir, const nn::Module &model, const Optimizer *optimizer,
                      const TrainerState &state, bool no_save_optim) {
    std::filesystem::create_directories(checkpoint_dir);
    LOG(INFO) << "[CKPT] Save begin: dir=" << checkpoint_dir << ", global_step=" << state.global_step;

    const auto model_path = checkpoint_dir / ("model.ckpt");

    SaveStateDictBinary(model_path, model.StateDict());

    if (!no_save_optim) {
        CHECK(optimizer != nullptr) << "Optimizer pointer is null, cannot save optimizer state.";
        auto opt_state = optimizer->StateDict();
        if (!opt_state.empty()) {
            const auto opt_path = checkpoint_dir / "optimizer.ckpt";
            SaveStateDictBinary(opt_path, opt_state);
        }
    }

    SaveTrainerState(checkpoint_dir / "trainer_state.json", state);
    LOG(ERROR) << "[CKPT] Save done: dir=" << checkpoint_dir;
}

void Checkpoint::Load(const std::filesystem::path &checkpoint_dir, nn::Module &model, Optimizer *optimizer,
                      TrainerState &state, bool load_optimizer_state) {
    const auto model_path = checkpoint_dir / "model.ckpt";
    LOG(INFO) << "[CKPT] Loading model: " << model_path;

    model.LoadStateDict(LoadStateDictBinary(model_path));

    if (optimizer == nullptr) {
        LOG(ERROR) << "[CKPT] No optimizer instance, skip optimizer state loading.";
    } else if (load_optimizer_state) {
        const auto opt_path = checkpoint_dir / "optimizer.ckpt";
        if (std::filesystem::exists(opt_path)) {
            LOG(ERROR) << "[CKPT] Loading optimizer: " << opt_path;
            optimizer->LoadStateDict(LoadStateDictBinary(opt_path));
        } else {
            LOG(ERROR) << "[CKPT] Optimizer state not found, skip: " << opt_path;
        }
    } else {
        LOG(ERROR) << "[CKPT] load_optimizer_state=false, skip optimizer state loading.";
    }

    state = LoadTrainerState(checkpoint_dir / "trainer_state.json");
    LOG(ERROR) << "[CKPT] Load done: global_step=" << state.global_step
               << ", consumed_batches =" << state.consumed_batches << ", last_lr=" << state.last_lr
               << ", topology(ddp,tp,sp,pp)=(" << state.ddp_size << "," << state.tp_size << "," << state.sp_size << ","
               << state.pp_size << ")";
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
        for (const auto dim : dims) { ofs.write(reinterpret_cast<const char *>(&dim), sizeof(dim)); }

        Tensor cpu_tensor = tensor->To(Device());
        uint64_t bytes = static_cast<uint64_t>(cpu_tensor.SizeInBytes());
        ofs.write(reinterpret_cast<const char *>(&bytes), sizeof(bytes));
        ofs.write(reinterpret_cast<const char *>(cpu_tensor.DataPtr()), static_cast<std::streamsize>(bytes));
    }
}

std::unordered_map<std::string, std::shared_ptr<Tensor>>
Checkpoint::LoadStateDictBinary(const std::filesystem::path &path) {
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
    ofs << "  \"vocab_size\": " << state.vocab_size << "\n";
    ofs << "  \"global_step\": " << state.global_step << ",\n";
    ofs << "  \"consumed_batches \": " << state.consumed_batches << ",\n";
    ofs << "  \"last_lr\": " << state.last_lr << ",\n";
    ofs << "  \"ddp_size\": " << state.ddp_size << ",\n";
    ofs << "  \"tp_size\": " << state.tp_size << ",\n";
    ofs << "  \"sp_size\": " << state.sp_size << ",\n";
    ofs << "  \"pp_size\": " << state.pp_size << "\n";
    ofs << "}\n";
}

// TODO(jym): Add TrainerState JSON version compatibility, referencing PyTorch's checkpoint
// versioning.
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
    state.consumed_batches = ExtractNumberField<int64_t>(content, "consumed_batches ", 0);
    state.last_lr = ExtractNumberField<double>(content, "last_lr", 0.0);
    state.ddp_size = ExtractNumberField<int>(content, "ddp_size", 1);
    state.tp_size = ExtractNumberField<int>(content, "tp_size", 1);
    state.sp_size = ExtractNumberField<int>(content, "sp_size", 1);
    state.pp_size = ExtractNumberField<int>(content, "pp_size", 1);
    return state;
}
} // namespace infini_train
