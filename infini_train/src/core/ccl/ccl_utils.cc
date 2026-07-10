#include "infini_train/include/core/ccl/ccl_utils.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>

#include "glog/logging.h"

namespace infini_train::core {
namespace {
std::string UniqueIdPath(const std::string &pg_name) {
    const char *run_id = std::getenv("INFINI_RUN_ID");
    const std::string prefix = run_id == nullptr ? "" : std::string(run_id) + "_";
    return "cclUniqueId_" + prefix + pg_name + ".bin";
}

std::string UniqueIdTmpPath(const std::string &pg_name) {
    const char *run_id = std::getenv("INFINI_RUN_ID");
    const std::string prefix = run_id == nullptr ? "" : std::string(run_id) + "_";
    return "cclUniqueId_" + prefix + pg_name + ".tmp";
}
} // namespace

void WriteUniqueIdFile(const CclUniqueId &unique_id, const std::string &pg_name) {
    const std::string tmp_path = UniqueIdTmpPath(pg_name);

    std::ofstream ofs(tmp_path, std::ios::binary);
    CHECK(ofs.good()) << "Failed to open unique_id tmp file for write: " << tmp_path;
    const size_t size = unique_id.Size();
    ofs.write(reinterpret_cast<const char *>(unique_id.Data()), static_cast<std::streamsize>(size));
    ofs.close();

    const std::string file_path = UniqueIdPath(pg_name);
    CHECK_EQ(std::rename(tmp_path.c_str(), file_path.c_str()), 0)
        << "Failed to rename unique_id file from " << tmp_path << " to " << file_path;
}

void ReadUniqueIdFile(CclUniqueId *unique_id, const std::string &pg_name) {
    CHECK_NOTNULL(unique_id);
    const std::string file_path = UniqueIdPath(pg_name);

    while (!std::filesystem::exists(file_path)) { std::this_thread::sleep_for(std::chrono::microseconds(1000)); }

    std::ifstream ifs(file_path, std::ios::binary);
    CHECK(ifs.good()) << "Failed to open unique_id file for read: " << file_path;

    std::string bytes((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    CHECK_EQ(bytes.size(), unique_id->Size())
        << "Mismatched unique_id size in file. expected=" << unique_id->Size() << ", got=" << bytes.size();
    unique_id->Load(bytes.data(), bytes.size());
}

void CleanupUniqueIdFile(const std::string &pg_name) {
    const std::string file_path = UniqueIdPath(pg_name);
    if (std::filesystem::exists(file_path)) {
        std::filesystem::remove(file_path);
    }

    const std::string tmp_path = UniqueIdTmpPath(pg_name);
    if (std::filesystem::exists(tmp_path)) {
        std::filesystem::remove(tmp_path);
    }
}

} // namespace infini_train::core
