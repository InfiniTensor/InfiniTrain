#include "infini_train/include/core/ccl/ccl_utils.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>

#include "glog/logging.h"

namespace infini_train::core {
namespace {
std::string UniqueIdFileName(const std::string &name, bool tmp = false) {
    return "cclUniqueId_" + name + (tmp ? ".tmp" : ".bin");
}
} // namespace

void WriteUniqueIdFile(const CclUniqueId &unique_id, const std::string &pg_name) {
    const std::string tmp_path = UniqueIdFileName(pg_name, true);

    std::ofstream ofs(tmp_path, std::ios::binary);
    CHECK(ofs.good()) << "Failed to open unique_id tmp file for write: " << tmp_path;
    const size_t size = unique_id.Size();
    ofs.write(reinterpret_cast<const char *>(unique_id.Data()), static_cast<std::streamsize>(size));
    ofs.close();

    std::rename(tmp_path.c_str(), UniqueIdFileName(pg_name).c_str());
}

void ReadUniqueIdFile(CclUniqueId *unique_id, const std::string &pg_name) {
    CHECK_NOTNULL(unique_id);
    const std::string file_path = UniqueIdFileName(pg_name);

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
    const std::string file_path = UniqueIdFileName(pg_name);
    if (std::filesystem::exists(file_path)) {
        std::filesystem::remove(file_path);
    }
}

} // namespace infini_train::core
