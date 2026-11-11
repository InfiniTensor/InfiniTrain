#include "infini_train/include/core/ccl/rendezvous_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdint.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/ccl/rendezvous_protocol.h"

#include <netdb.h>

namespace infini_train::core {
namespace {
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
}  // namespace

class RendezvousServer::Impl {
 public:
  Impl(const std::string &host, int port, int timeout) : timeout_(timeout) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *addresses = nullptr;
    const std::string port_text = std::to_string(port);
    CHECK_EQ(getaddrinfo(host.c_str(), port_text.c_str(), &hints, &addresses),
             0)
        << "Failed to resolve rendezvous bind address " << host;
    for (addrinfo *address = addresses; address != nullptr;
         address = address->ai_next) {
      fd_ = socket(address->ai_family, address->ai_socktype,
                   address->ai_protocol);
      if (fd_ < 0) continue;
      int reuse = 1;
      setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
      if (bind(fd_, address->ai_addr, address->ai_addrlen) == 0) break;
      close(fd_);
      fd_ = -1;
    }
    freeaddrinfo(addresses);
    CHECK_GE(fd_, 0) << "Failed to bind rendezvous endpoint " << host << ':'
                     << port << ": " << strerror(errno);
    CHECK_EQ(listen(fd_, 64), 0);

    sockaddr_storage bound_address{};
    socklen_t bound_address_size = sizeof(bound_address);
    CHECK_EQ(getsockname(fd_, reinterpret_cast<sockaddr *>(&bound_address),
                         &bound_address_size),
             0);
    if (bound_address.ss_family == AF_INET) {
      port_ = ntohs(reinterpret_cast<sockaddr_in *>(&bound_address)->sin_port);
    } else {
      port_ =
          ntohs(reinterpret_cast<sockaddr_in6 *>(&bound_address)->sin6_port);
    }
    thread_ = std::thread([this] { Serve(); });
  }

  ~Impl() {
    shutdown(fd_, SHUT_RDWR);
    close(fd_);
    if (thread_.joinable()) thread_.join();
    for (auto &worker : workers_) {
      if (worker.joinable()) worker.join();
    }
  }

  int port() const { return port_; }

 private:
  void Serve() {
    while (true) {
      const int client = accept(fd_, nullptr, nullptr);
      if (client < 0) break;
      workers_.emplace_back(&Impl::Handle, this, client);
    }
  }

  void Handle(int client) {
    rendezvous::Header wire{};
    if (!ReadAll(client, &wire, sizeof(wire))) {
      close(client);
      return;
    }
    const auto operation =
        static_cast<rendezvous::Operation>(ntohl(wire.operation));
    const uint32_t key_size = ntohl(wire.key_size);
    const uint32_t value_size = ntohl(wire.value_size);
    if (ntohl(wire.magic) != rendezvous::kMagic || key_size == 0 ||
        key_size > rendezvous::kMaxKeySize ||
        value_size > rendezvous::kMaxValueSize) {
      close(client);
      return;
    }

    std::string key(key_size, '\0');
    std::string value(value_size, '\0');
    if (!ReadAll(client, key.data(), key.size()) ||
        (!value.empty() && !ReadAll(client, value.data(), value.size()))) {
      close(client);
      return;
    }

    uint32_t status = 0;
    if (operation == rendezvous::Operation::kPut) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        values_[key] = std::move(value);
      }
      condition_.notify_all();
    } else if (operation == rendezvous::Operation::kGet) {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!condition_.wait_for(lock, std::chrono::seconds(timeout_),
                               [&] { return values_.contains(key); })) {
        status = 1;
      } else {
        value = values_.at(key);
      }
    } else {
      status = 2;
    }

    const rendezvous::Response response{
        htonl(status), htonl(static_cast<uint32_t>(value.size()))};
    WriteAll(client, &response, sizeof(response));
    if (status == 0 && operation == rendezvous::Operation::kGet) {
      WriteAll(client, value.data(), value.size());
    }
    close(client);
  }

  int fd_ = -1;
  int timeout_;
  int port_ = -1;
  std::thread thread_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::map<std::string, std::string> values_;
};

RendezvousServer::RendezvousServer(const std::string &host, int port,
                                   int timeout)
    : impl_(std::make_unique<Impl>(host, port, timeout)) {}

RendezvousServer::~RendezvousServer() = default;

int RendezvousServer::port() const { return impl_->port(); }
}  // namespace infini_train::core
