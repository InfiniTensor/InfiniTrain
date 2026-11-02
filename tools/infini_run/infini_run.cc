// IWYU pragma: no_include <bits/types/sig_atomic_t.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <stdint.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <compare>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_int32(nnodes, 1, "Total number of nodes");
DEFINE_int32(nproc_per_node, 1, "Number of processes per node");
DEFINE_int32(node_rank, 0, "Rank of this node");
DEFINE_string(rdzv_endpoint, "127.0.0.1:29500", "Rendezvous endpoint");
DEFINE_string(rdzv_id, "", "Unique job ID shared by all nodes");
DEFINE_int32(rdzv_timeout, 120, "Rendezvous timeout in seconds");
DEFINE_string(log_dir, "", "Directory for per-rank stdout/stderr logs");

namespace {
constexpr uint32_t kMagic = 0x49545244;
volatile std::sig_atomic_t g_signal = 0;
void HandleSignal(int signal) { g_signal = signal; }

bool IsLauncherFlag(const std::string &flag) {
  return flag == "--nnodes" || flag == "--nproc_per_node" ||
         flag == "--node_rank" || flag == "--rdzv_endpoint" ||
         flag == "--rdzv_id" || flag == "--rdzv_timeout" || flag == "--log_dir";
}
int FindTrainProgramIndex(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--") return i + 1;
    if (!arg.starts_with("--")) return i;
    const size_t eq = arg.find('=');
    if (IsLauncherFlag(arg.substr(0, eq)) && eq == std::string::npos) ++i;
  }
  return argc;
}
std::vector<char *> BuildLauncherArgv(int end, char **argv) {
  std::vector<char *> result{argv[0]};
  for (int i = 1; i < end; ++i) result.push_back(argv[i]);
  result.push_back(nullptr);
  return result;
}
void SetEnvInt(const char *name, int value) {
  const std::string text = std::to_string(value);
  setenv(name, text.c_str(), 1);
}
bool ReadAll(int fd, void *buffer, size_t size) {
  auto *bytes = static_cast<char *>(buffer);
  while (size > 0) {
    ssize_t n = read(fd, bytes, size);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    bytes += n;
    size -= n;
  }
  return true;
}
bool WriteAll(int fd, const void *buffer, size_t size) {
  const auto *bytes = static_cast<const char *>(buffer);
  while (size > 0) {
    ssize_t n = write(fd, bytes, size);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    bytes += n;
    size -= n;
  }
  return true;
}

class RendezvousServer {
 public:
  RendezvousServer(int port, int timeout) : timeout_(timeout) {
    fd_ = socket(AF_INET, SOCK_STREAM, 0);
    CHECK_GE(fd_, 0);
    int reuse = 1;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port);
    CHECK_EQ(bind(fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)),
             0)
        << "Failed to bind rendezvous port " << port << ": " << strerror(errno);
    CHECK_EQ(listen(fd_, 64), 0);
    thread_ = std::thread([this] { Serve(); });
  }
  ~RendezvousServer() {
    shutdown(fd_, SHUT_RDWR);
    close(fd_);
    if (thread_.joinable()) thread_.join();
    for (auto &worker : workers_)
      if (worker.joinable()) worker.join();
  }

 private:
  struct Header {
    uint32_t magic, operation, key_size, value_size;
  };
  void Serve() {
    while (true) {
      int client = accept(fd_, nullptr, nullptr);
      if (client < 0) break;
      workers_.emplace_back(&RendezvousServer::Handle, this, client);
    }
  }
  void Handle(int client) {
    Header wire{};
    if (!ReadAll(client, &wire, sizeof(wire))) {
      close(client);
      return;
    }
    uint32_t operation = ntohl(wire.operation), key_size = ntohl(wire.key_size),
             value_size = ntohl(wire.value_size);
    if (ntohl(wire.magic) != kMagic || key_size == 0 || key_size > 4096 ||
        value_size > 1024 * 1024) {
      close(client);
      return;
    }
    std::string key(key_size, '\0'), value(value_size, '\0');
    if (!ReadAll(client, key.data(), key.size()) ||
        (!value.empty() && !ReadAll(client, value.data(), value.size()))) {
      close(client);
      return;
    }
    uint32_t status = 0;
    if (operation == 1) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        values_[key] = std::move(value);
      }
      condition_.notify_all();
    } else if (operation == 2) {
      std::unique_lock<std::mutex> lock(mutex_);
      if (!condition_.wait_for(lock, std::chrono::seconds(timeout_),
                               [&] { return values_.contains(key); }))
        status = 1;
      else
        value = values_.at(key);
    } else
      status = 2;
    CHECK_LE(value.size(), std::numeric_limits<uint32_t>::max());
    uint32_t response[2] = {htonl(status),
                            htonl(static_cast<uint32_t>(value.size()))};
    WriteAll(client, response, sizeof(response));
    if (status == 0 && operation == 2)
      WriteAll(client, value.data(), value.size());
    close(client);
  }
  int fd_ = -1, timeout_;
  std::thread thread_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::map<std::string, std::string> values_;
};

int ExitCode(int status) {
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  return WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 1;
}
void SignalChildren(const std::unordered_set<pid_t> &children, int signal) {
  for (pid_t pid : children) kill(pid, signal);
}
std::string LocalRunId() {
  return std::to_string(getpid()) + "_" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count());
}
void CleanupFiles(const std::string &run_id) {
  const std::string prefix = "cclUniqueId_" + run_id + "_";
  for (const auto &entry : std::filesystem::directory_iterator(".")) {
    if (entry.is_regular_file() &&
        entry.path().filename().string().starts_with(prefix)) {
      std::error_code error;
      std::filesystem::remove(entry.path(), error);
    }
  }
}
}  // namespace

int main(int argc, char **argv) {
  const int train_index = FindTrainProgramIndex(argc, argv);
  auto launcher_argv = BuildLauncherArgv(train_index, argv);
  int launcher_argc = launcher_argv.size() - 1;
  char **launcher_argv_ptr = launcher_argv.data();
  gflags::ParseCommandLineFlags(&launcher_argc, &launcher_argv_ptr, true);
  google::InitGoogleLogging(argv[0]);
  CHECK_GT(FLAGS_nnodes, 0);
  CHECK_GT(FLAGS_nproc_per_node, 0);
  CHECK_GE(FLAGS_node_rank, 0);
  CHECK_LT(FLAGS_node_rank, FLAGS_nnodes);
  CHECK_GT(FLAGS_rdzv_timeout, 0);
  CHECK_LT(train_index, argc);
  const size_t separator = FLAGS_rdzv_endpoint.rfind(':');
  CHECK_NE(separator, std::string::npos) << "rdzv_endpoint must be host:port";
  CHECK(FLAGS_nnodes == 1 || !FLAGS_rdzv_id.empty());

  const std::string master_addr = FLAGS_rdzv_endpoint.substr(0, separator);
  const std::string master_port = FLAGS_rdzv_endpoint.substr(separator + 1);
  char *port_end = nullptr;
  errno = 0;
  const long parsed_port = std::strtol(master_port.c_str(), &port_end, 10);
  CHECK(errno == 0 && port_end != master_port.c_str() && *port_end == '\0' &&
        parsed_port > 0 && parsed_port <= 65535)
      << "Invalid rendezvous port: " << master_port;
  const std::string run_id =
      FLAGS_rdzv_id.empty() ? LocalRunId() : FLAGS_rdzv_id;
  setenv("INFINI_RUN_ID", run_id.c_str(), 1);
  setenv("INFINI_RDZV_TIMEOUT_SEC", std::to_string(FLAGS_rdzv_timeout).c_str(),
         1);
  std::unique_ptr<RendezvousServer> server;
  if (FLAGS_nnodes > 1 && FLAGS_node_rank == 0)
    server = std::make_unique<RendezvousServer>(static_cast<int>(parsed_port),
                                                FLAGS_rdzv_timeout);
  if (!FLAGS_log_dir.empty())
    std::filesystem::create_directories(FLAGS_log_dir);
  struct sigaction action {};
  action.sa_handler = HandleSignal;
  sigemptyset(&action.sa_mask);
  sigaction(SIGINT, &action, nullptr);
  sigaction(SIGTERM, &action, nullptr);

  std::vector<char *> train_argv;
  for (int i = train_index; i < argc; ++i) train_argv.push_back(argv[i]);
  train_argv.push_back(nullptr);
  std::unordered_set<pid_t> children;
  int exit_code = 0;
  const int world_size = FLAGS_nnodes * FLAGS_nproc_per_node;
  for (int local_rank = 0; local_rank < FLAGS_nproc_per_node; ++local_rank) {
    pid_t pid = fork();
    if (pid < 0) {
      PLOG(ERROR) << "fork failed after starting " << children.size()
                  << " rank processes";
      SignalChildren(children, SIGTERM);
      for (pid_t child : children) waitpid(child, nullptr, 0);
      return 1;
    }
    if (pid == 0) {
      const int rank = FLAGS_node_rank * FLAGS_nproc_per_node + local_rank;
      SetEnvInt("LOCAL_WORLD_SIZE", FLAGS_nproc_per_node);
      SetEnvInt("WORLD_SIZE", world_size);
      SetEnvInt("RANK", rank);
      SetEnvInt("LOCAL_RANK", local_rank);
      setenv("MASTER_ADDR", master_addr.c_str(), 1);
      setenv("MASTER_PORT", master_port.c_str(), 1);
      if (!FLAGS_log_dir.empty()) {
        const std::string prefix =
            FLAGS_log_dir + "/rank-" + std::to_string(rank);
        int out = open((prefix + ".stdout.log").c_str(),
                       O_WRONLY | O_CREAT | O_TRUNC, 0644);
        int err = open((prefix + ".stderr.log").c_str(),
                       O_WRONLY | O_CREAT | O_TRUNC, 0644);
        CHECK_GE(out, 0);
        CHECK_GE(err, 0);
        dup2(out, STDOUT_FILENO);
        dup2(err, STDERR_FILENO);
        close(out);
        close(err);
      }
      char hostname[256] = {};
      gethostname(hostname, sizeof(hostname) - 1);
      fprintf(stderr,
              "INFINITRAIN_RANK hostname=%s global_rank=%d local_rank=%d "
              "world_size=%d device=%d job_id=%s\n",
              hostname, rank, local_rank, world_size, local_rank,
              run_id.c_str());
      fflush(stderr);
      execvp(train_argv[0], train_argv.data());
      perror("exec failed");
      _exit(127);
    }
    children.insert(pid);
  }

  bool term_sent = false;
  auto term_time = std::chrono::steady_clock::now();
  while (!children.empty()) {
    if (g_signal && !term_sent) {
      exit_code = 128 + g_signal;
      SignalChildren(children, SIGTERM);
      term_sent = true;
      term_time = std::chrono::steady_clock::now();
    }
    if (term_sent &&
        std::chrono::steady_clock::now() - term_time > std::chrono::seconds(10))
      SignalChildren(children, SIGKILL);
    int status = 0;
    pid_t child = waitpid(-1, &status, term_sent ? WNOHANG : 0);
    if (child == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    if (child < 0) {
      if (errno == EINTR) continue;
      exit_code = 1;
      break;
    }
    children.erase(child);
    const int code = ExitCode(status);
    if (code != 0 && exit_code == 0) {
      exit_code = code;
      SignalChildren(children, SIGTERM);
      term_sent = true;
      term_time = std::chrono::steady_clock::now();
    }
  }
  if (FLAGS_nnodes == 1) CleanupFiles(run_id);
  return exit_code;
}
