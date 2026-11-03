#include "infini_train/include/core/ccl/ccl_utils.h"

#include <arpa/inet.h>
#include <stdint.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <compare>
#include <cstdlib>
#include <limits>
#include <ostream>
#include <string_view>
#include <thread>

#include "glog/logging.h"

#include "infini_train/include/core/ccl/ccl_common.h"
#include "infini_train/include/core/ccl/rendezvous_protocol.h"

#include <netdb.h>

namespace infini_train::core {
namespace {
int GetEnvInt(const char *name, int fallback) {
  const char *value = std::getenv(name);
  return value == nullptr ? fallback : std::stoi(value);
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

std::string RendezvousRequest(rendezvous::Operation operation,
                              const std::string &key, std::string_view value) {
  CHECK_LE(key.size(), std::numeric_limits<uint32_t>::max());
  CHECK_LE(value.size(), std::numeric_limits<uint32_t>::max());
  const int fd = ConnectRendezvous();
  const rendezvous::Header header{htonl(rendezvous::kMagic),
                                  htonl(static_cast<uint32_t>(operation)),
                                  htonl(static_cast<uint32_t>(key.size())),
                                  htonl(static_cast<uint32_t>(value.size()))};
  CHECK(WriteAll(fd, &header, sizeof(header)) &&
        WriteAll(fd, key.data(), key.size()) &&
        (value.empty() || WriteAll(fd, value.data(), value.size())))
      << "Failed to send rendezvous request for " << key;
  rendezvous::Response response{};
  CHECK(ReadAll(fd, &response, sizeof(response)));
  CHECK_EQ(ntohl(response.status), 0)
      << "Rendezvous request failed for " << key;
  std::string result(ntohl(response.value_size), '\0');
  CHECK(result.empty() || ReadAll(fd, result.data(), result.size()));
  close(fd);
  return result;
}

std::string RendezvousKey(const std::string &pg_name) {
  const char *run_id = std::getenv("INFINI_RUN_ID");
  CHECK(run_id != nullptr) << "INFINI_RUN_ID is required";
  return std::string(run_id) + "/ccl/" + pg_name;
}

}  // namespace

void WriteUniqueId(const CclUniqueId &unique_id, const std::string &pg_name) {
  RendezvousRequest(
      rendezvous::Operation::kPut, RendezvousKey(pg_name),
      std::string_view(static_cast<const char *>(unique_id.Data()),
                       unique_id.Size()));
}

void ReadUniqueId(CclUniqueId *unique_id, const std::string &pg_name) {
  CHECK_NOTNULL(unique_id);
  const std::string bytes = RendezvousRequest(rendezvous::Operation::kGet,
                                              RendezvousKey(pg_name), {});
  CHECK_EQ(bytes.size(), unique_id->Size());
  unique_id->Load(bytes.data(), bytes.size());
}
}  // namespace infini_train::core
