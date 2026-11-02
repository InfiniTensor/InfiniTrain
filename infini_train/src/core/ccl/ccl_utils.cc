#include "infini_train/include/core/ccl/ccl_utils.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string_view>
#include <thread>

#include "glog/logging.h"

#include "infini_train/include/core/ccl/ccl_common.h"

#include <netdb.h>

namespace infini_train::core {
namespace {
constexpr uint32_t kRendezvousMagic = 0x49545244;
struct RendezvousHeader {
  uint32_t magic, operation, key_size, value_size;
};

int GetEnvInt(const char *name, int fallback) {
  const char *value = std::getenv(name);
  return value == nullptr ? fallback : std::stoi(value);
}

bool UseTcpRendezvous() {
  return GetEnvInt("WORLD_SIZE", 1) > GetEnvInt("LOCAL_WORLD_SIZE", 1);
}

bool ReadAll(int fd, void *buffer, size_t size) {
  auto *bytes = static_cast<char *>(buffer);
  while (size > 0) {
    const ssize_t count = read(fd, bytes, size);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return false;
    bytes += count;
    size -= static_cast<size_t>(count);
  }
  return true;
}

bool WriteAll(int fd, const void *buffer, size_t size) {
  const auto *bytes = static_cast<const char *>(buffer);
  while (size > 0) {
    const ssize_t count = write(fd, bytes, size);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return false;
    bytes += count;
    size -= static_cast<size_t>(count);
  }
  return true;
}

int ConnectRendezvous() {
  const char *host = std::getenv("MASTER_ADDR");
  const char *port = std::getenv("MASTER_PORT");
  CHECK(host != nullptr && port != nullptr)
      << "MASTER_ADDR and MASTER_PORT are required for multi-node rendezvous";
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo *addresses = nullptr;
  CHECK_EQ(getaddrinfo(host, port, &hints, &addresses), 0)
      << "Failed to resolve rendezvous endpoint " << host << ':' << port;
  const int timeout = GetEnvInt("INFINI_RDZV_TIMEOUT_SEC", 120);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeout);
  int fd = -1;
  while (std::chrono::steady_clock::now() < deadline && fd < 0) {
    for (addrinfo *address = addresses; address != nullptr;
         address = address->ai_next) {
      const int candidate = socket(address->ai_family, address->ai_socktype,
                                   address->ai_protocol);
      if (candidate < 0) continue;
      if (connect(candidate, address->ai_addr, address->ai_addrlen) == 0) {
        fd = candidate;
        break;
      }
      close(candidate);
    }
    if (fd < 0) std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  freeaddrinfo(addresses);
  CHECK_GE(fd, 0) << "Timed out connecting to rendezvous " << host << ':'
                  << port;
  return fd;
}

std::string RendezvousRequest(uint32_t operation, const std::string &key,
                              std::string_view value) {
  CHECK_LE(key.size(), std::numeric_limits<uint32_t>::max());
  CHECK_LE(value.size(), std::numeric_limits<uint32_t>::max());
  const int fd = ConnectRendezvous();
  const RendezvousHeader header{htonl(kRendezvousMagic), htonl(operation),
                                htonl(static_cast<uint32_t>(key.size())),
                                htonl(static_cast<uint32_t>(value.size()))};
  CHECK(WriteAll(fd, &header, sizeof(header)) &&
        WriteAll(fd, key.data(), key.size()) &&
        (value.empty() || WriteAll(fd, value.data(), value.size())))
      << "Failed to send rendezvous request for " << key;
  uint32_t response[2] = {};
  CHECK(ReadAll(fd, response, sizeof(response)));
  CHECK_EQ(ntohl(response[0]), 0) << "Rendezvous request failed for " << key;
  std::string result(ntohl(response[1]), '\0');
  CHECK(result.empty() || ReadAll(fd, result.data(), result.size()));
  close(fd);
  return result;
}

std::string RendezvousKey(const std::string &pg_name) {
  const char *run_id = std::getenv("INFINI_RUN_ID");
  CHECK(run_id != nullptr) << "INFINI_RUN_ID is required";
  return std::string(run_id) + "/ccl/" + pg_name;
}

std::string UniqueIdFileName(const std::string &pg_name) {
  const char *run_id = std::getenv("INFINI_RUN_ID");
  const std::string prefix = run_id == nullptr ? "" : std::string(run_id) + "_";
  return "cclUniqueId_" + prefix + pg_name + ".bin";
}

std::string UniqueIdTmpFileName(const std::string &pg_name) {
  const char *run_id = std::getenv("INFINI_RUN_ID");
  const std::string prefix = run_id == nullptr ? "" : std::string(run_id) + "_";
  return "cclUniqueId_" + prefix + pg_name + ".tmp";
}
}  // namespace

void WriteUniqueIdFile(const CclUniqueId &unique_id,
                       const std::string &pg_name) {
  if (UseTcpRendezvous()) {
    RendezvousRequest(
        1, RendezvousKey(pg_name),
        std::string_view(static_cast<const char *>(unique_id.Data()),
                         unique_id.Size()));
    return;
  }
  const std::string tmp_path = UniqueIdTmpFileName(pg_name);
  std::ofstream ofs(tmp_path, std::ios::binary);
  CHECK(ofs.good()) << "Failed to open unique_id tmp file for write: "
                    << tmp_path;
  ofs.write(reinterpret_cast<const char *>(unique_id.Data()),
            static_cast<std::streamsize>(unique_id.Size()));
  ofs.close();
  const std::string file_path = UniqueIdFileName(pg_name);
  CHECK_EQ(std::rename(tmp_path.c_str(), file_path.c_str()), 0)
      << "Failed to rename unique_id file from " << tmp_path << " to "
      << file_path;
}

void ReadUniqueIdFile(CclUniqueId *unique_id, const std::string &pg_name) {
  CHECK_NOTNULL(unique_id);
  if (UseTcpRendezvous()) {
    const std::string bytes = RendezvousRequest(2, RendezvousKey(pg_name), {});
    CHECK_EQ(bytes.size(), unique_id->Size());
    unique_id->Load(bytes.data(), bytes.size());
    return;
  }
  const std::string file_path = UniqueIdFileName(pg_name);
  const int timeout = GetEnvInt("INFINI_RDZV_TIMEOUT_SEC", 120);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(timeout);
  while (!std::filesystem::exists(file_path)) {
    CHECK(std::chrono::steady_clock::now() < deadline)
        << "Timed out waiting for unique-id file " << file_path;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::ifstream ifs(file_path, std::ios::binary);
  CHECK(ifs.good()) << "Failed to open unique_id file for read: " << file_path;
  std::string bytes((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
  CHECK_EQ(bytes.size(), unique_id->Size());
  unique_id->Load(bytes.data(), bytes.size());
}

void CleanupUniqueIdFile(const std::string &pg_name) {
  if (UseTcpRendezvous()) return;
  for (const auto &path :
       {UniqueIdFileName(pg_name), UniqueIdTmpFileName(pg_name)}) {
    if (std::filesystem::exists(path)) std::filesystem::remove(path);
  }
}
}  // namespace infini_train::core
