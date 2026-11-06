#pragma once

#include <memory>
#include <string>

namespace infini_train::core {
class RendezvousServer {
 public:
  RendezvousServer(const std::string &host, int port, int timeout);
  ~RendezvousServer();

  RendezvousServer(const RendezvousServer &) = delete;
  RendezvousServer &operator=(const RendezvousServer &) = delete;

  int port() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace infini_train::core
