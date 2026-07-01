#include "infini_train/include/core/ccl/ccl_utils.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <thread>

#include "glog/logging.h"

namespace infini_train::core {
namespace {

constexpr int64_t kDefaultUniqueIdTimeoutSec = 600;

std::string GetEnvString(const char *name, const std::string &default_value) {
    const char *value = std::getenv(name);
    return value == nullptr ? default_value : std::string(value);
}

int64_t GetEnvInt64(const char *name, int64_t default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr) {
        return default_value;
    }
    try {
        return std::stoll(value);
    } catch (...) {
        LOG(WARNING) << "Invalid integer environment variable " << name << "=" << value << ", fallback to "
                     << default_value;
        return default_value;
    }
}

std::string SanitizeFileComponent(std::string value) {
    std::replace_if(
        value.begin(), value.end(),
        [](unsigned char c) { return !(std::isalnum(c) || c == '_' || c == '-' || c == '.'); }, '_');
    return value;
}

std::filesystem::path UniqueIdFilePath(const std::string &pg_name) {
    std::string file_name = "cclUniqueId_";
    const std::string name_space = GetEnvString("INFINITRAIN_CCL_ID_NAMESPACE", "");
    if (!name_space.empty()) {
        file_name += SanitizeFileComponent(name_space) + "_";
    }
    file_name += SanitizeFileComponent(pg_name) + ".bin";
    return std::filesystem::path(GetEnvString("INFINITRAIN_CCL_ID_DIR", ".")) / file_name;
}

std::filesystem::path UniqueIdTmpFilePath(const std::filesystem::path &file_path) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string global_proc_rank = GetEnvString("GLOBAL_PROC_RANK", "0");
    return std::filesystem::path(file_path.string() + ".tmp." + global_proc_rank + "." + std::to_string(now));
}

void RemoveTmpFilesFor(const std::filesystem::path &file_path) {
    const auto parent = file_path.parent_path();
    std::error_code ec;
    if (!std::filesystem::exists(parent, ec)) {
        return;
    }

    const std::string prefix = file_path.filename().string() + ".tmp.";
    for (const auto &entry : std::filesystem::directory_iterator(parent, ec)) {
        if (ec) {
            return;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0) {
            std::error_code remove_ec;
            std::filesystem::remove(entry.path(), remove_ec);
        }
    }
}
} // namespace

void WriteUniqueIdFile(const CclUniqueId &unique_id, const std::string &pg_name) {
    const auto file_path = UniqueIdFilePath(pg_name);
    const auto tmp_path = UniqueIdTmpFilePath(file_path);

    std::error_code ec;
    std::filesystem::create_directories(file_path.parent_path(), ec);
    CHECK(!ec) << "Failed to create CCL unique_id directory: " << file_path.parent_path() << ", error=" << ec.message();
    RemoveTmpFilesFor(file_path);

    std::ofstream ofs(tmp_path, std::ios::binary);
    CHECK(ofs.good()) << "Failed to open unique_id tmp file for write: " << tmp_path;
    const size_t size = unique_id.Size();
    ofs.write(reinterpret_cast<const char *>(unique_id.Data()), static_cast<std::streamsize>(size));
    CHECK(ofs.good()) << "Failed to write unique_id tmp file: " << tmp_path;
    ofs.close();
    CHECK(!ofs.fail()) << "Failed to close unique_id tmp file: " << tmp_path;

    const auto tmp_size = std::filesystem::file_size(tmp_path, ec);
    CHECK(!ec && tmp_size == size) << "Invalid unique_id tmp file size. file=" << tmp_path << ", expected=" << size
                                   << ", got=" << (ec ? 0 : tmp_size) << ", error=" << ec.message();

    std::filesystem::rename(tmp_path, file_path, ec);
    if (ec) {
        std::error_code remove_ec;
        std::filesystem::remove(tmp_path, remove_ec);
        LOG(FATAL) << "Failed to publish unique_id file. tmp=" << tmp_path << ", final=" << file_path
                   << ", error=" << ec.message();
    }
}

void ReadUniqueIdFile(CclUniqueId *unique_id, const std::string &pg_name) {
    CHECK_NOTNULL(unique_id);
    const auto file_path = UniqueIdFilePath(pg_name);
    const auto timeout_sec
        = std::max<int64_t>(1, GetEnvInt64("INFINITRAIN_CCL_ID_TIMEOUT_SEC", kDefaultUniqueIdTimeoutSec));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_sec);
    const size_t expected_size = unique_id->Size();
    uintmax_t last_observed_size = 0;

    while (std::chrono::steady_clock::now() < deadline) {
        std::error_code ec;
        if (std::filesystem::exists(file_path, ec)) {
            const auto file_size = std::filesystem::file_size(file_path, ec);
            if (!ec) {
                last_observed_size = file_size;
                if (file_size == expected_size) {
                    std::ifstream ifs(file_path, std::ios::binary);
                    if (ifs.good()) {
                        std::string bytes((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
                        ifs.close();
                        if (bytes.size() == expected_size) {
                            unique_id->Load(bytes.data(), bytes.size());
                            return;
                        }
                        last_observed_size = bytes.size();
                    }
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    LOG(FATAL) << "Timed out waiting for CCL unique_id file. file=" << file_path << ", expected_size=" << expected_size
               << ", last_observed_size=" << last_observed_size << ", timeout_sec=" << timeout_sec
               << ". Set INFINITRAIN_CCL_ID_NAMESPACE for concurrent jobs sharing a directory.";
}

void CleanupUniqueIdFile(const std::string &pg_name) {
    const auto file_path = UniqueIdFilePath(pg_name);
    std::error_code ec;
    std::filesystem::remove(file_path, ec);
    RemoveTmpFilesFor(file_path);
}

} // namespace infini_train::core
