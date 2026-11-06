// IWYU pragma: no_include <bits/types/sig_atomic_t.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <compare>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <ostream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/core/ccl/rendezvous_server.h"

DEFINE_int32(nnodes, 1, "Total number of nodes");
DEFINE_int32(nproc_per_node, 1, "Number of processes per node");
DEFINE_int32(node_rank, 0, "Rank of this node");
DEFINE_string(rdzv_endpoint, "127.0.0.1:29500",
              "Rendezvous endpoint (host:port)");

DEFINE_string(rdzv_id, "", "Unique job ID shared by all nodes");
DEFINE_int32(rdzv_timeout, 120, "Rendezvous timeout in seconds");
DEFINE_string(log_dir, "", "Directory for per-rank stdout/stderr logs");

namespace {
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
    if (arg == "--") {
      return i + 1;
    }
    if (arg.rfind("--", 0) != 0) {
      return i;
    }

    const size_t eq_pos = arg.find('=');
    const std::string flag = arg.substr(0, eq_pos);
    if (IsLauncherFlag(flag) && eq_pos == std::string::npos) {
      ++i;
    }
  }
  return argc;
}

std::vector<char *> BuildLauncherArgv(int train_program_index, char **argv) {
  std::vector<char *> launcher_argv;
  launcher_argv.push_back(argv[0]);
  for (int i = 1; i < train_program_index; ++i) {
    launcher_argv.push_back(argv[i]);
  }
  launcher_argv.push_back(nullptr);
  return launcher_argv;
}

void SetEnvInt(const char *name, int value) {
  const auto value_str = std::to_string(value);
  setenv(name, value_str.c_str(), 1);
}

void TerminateChildren(const std::unordered_set<pid_t> &child_pids,
                       int signal = SIGTERM) {
  for (pid_t child_pid : child_pids) {
    kill(child_pid, signal);
  }
}

int ExitCodeFromStatus(int status) {
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }
  return 1;
}

std::string GenerateLocalRunId() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::to_string(getpid()) + "_" + std::to_string(now);
}

}  // namespace

int main(int argc, char **argv) {
  gflags::SetUsageMessage(
      "Launch distributed training with InfiniTrain.\nUsage: " +
      std::string(argv[0]) +
      " [launcher options] <training program> [training options]");
  const int train_program_index = FindTrainProgramIndex(argc, argv);
  std::vector<char *> launcher_argv =
      BuildLauncherArgv(train_program_index, argv);
  int launcher_argc = static_cast<int>(launcher_argv.size()) - 1;
  char **launcher_argv_ptr = launcher_argv.data();
  gflags::ParseCommandLineFlags(&launcher_argc, &launcher_argv_ptr, true);
  google::InitGoogleLogging(argv[0]);

  CHECK_GT(FLAGS_nnodes, 0) << "nnodes must be positive";
  CHECK_GT(FLAGS_nproc_per_node, 0) << "nproc_per_node must be positive";
  CHECK_GE(FLAGS_node_rank, 0) << "node_rank must be non-negative";
  CHECK_LT(FLAGS_node_rank, FLAGS_nnodes)
      << "node_rank must be less than nnodes";
  CHECK_GT(FLAGS_rdzv_timeout, 0) << "rdzv_timeout must be positive";
  CHECK_NE(FLAGS_rdzv_endpoint.find(':'), std::string::npos)
      << "rdzv_endpoint must be host:port";
  CHECK(FLAGS_nnodes == 1 || !FLAGS_rdzv_id.empty())
      << "rdzv_id must be set to the same unique job ID on every node for "
         "multi-node training";

  CHECK_LT(train_program_index, argc) << "No training program specified!";

  std::string train_program = argv[train_program_index];
  std::vector<char *> train_argv;
  for (int i = train_program_index; i < argc; ++i) {
    train_argv.push_back(argv[i]);
  }
  train_argv.push_back(nullptr);

  int proc_world_size = FLAGS_nnodes * FLAGS_nproc_per_node;
  const size_t separator = FLAGS_rdzv_endpoint.rfind(':');
  std::string master_addr = FLAGS_rdzv_endpoint.substr(0, separator);
  std::string master_port = FLAGS_rdzv_endpoint.substr(separator + 1);
  char *port_end = nullptr;
  errno = 0;
  const long parsed_port = std::strtol(master_port.c_str(), &port_end, 10);
  CHECK(errno == 0 && port_end != master_port.c_str() && *port_end == '\0' &&
        parsed_port > 0 && parsed_port <= 65535)
      << "Invalid rendezvous port: " << master_port;
  const std::string run_id =
      FLAGS_rdzv_id.empty() ? GenerateLocalRunId() : FLAGS_rdzv_id;
  const bool endpoint_is_default =
      gflags::GetCommandLineFlagInfoOrDie("rdzv_endpoint").is_default;
  int effective_master_port = static_cast<int>(parsed_port);
  std::unique_ptr<infini_train::core::RendezvousServer> server;
  if (proc_world_size > 1 && FLAGS_node_rank == 0) {
    if (FLAGS_nnodes == 1 && endpoint_is_default) {
      master_addr = "127.0.0.1";
      effective_master_port = 0;
    }
    server = std::make_unique<infini_train::core::RendezvousServer>(
        master_addr, effective_master_port, FLAGS_rdzv_timeout);
    effective_master_port = server->port();
  }
  const std::string effective_master_port_text =
      std::to_string(effective_master_port);
  if (!FLAGS_log_dir.empty())
    std::filesystem::create_directories(FLAGS_log_dir);
  struct sigaction action {};
  action.sa_handler = HandleSignal;
  sigemptyset(&action.sa_mask);
  sigaction(SIGINT, &action, nullptr);
  sigaction(SIGTERM, &action, nullptr);

  std::unordered_set<pid_t> running_children;
  int exit_code = 0;

  for (int local_proc_rank = 0; local_proc_rank < FLAGS_nproc_per_node;
       ++local_proc_rank) {
    pid_t pid = fork();
    if (pid < 0) {
      PLOG(ERROR) << "fork failed after starting " << running_children.size()
                  << " rank processes";
      TerminateChildren(running_children);
      for (pid_t child : running_children) waitpid(child, nullptr, 0);
      return 1;
    }
    if (pid == 0) {
      int global_proc_rank =
          FLAGS_node_rank * FLAGS_nproc_per_node + local_proc_rank;
      SetEnvInt("LOCAL_WORLD_SIZE", FLAGS_nproc_per_node);

      setenv("MASTER_ADDR", master_addr.c_str(), 1);
      setenv("MASTER_PORT", effective_master_port_text.c_str(), 1);
      setenv("INFINI_RUN_ID", run_id.c_str(), 1);
      setenv("INFINI_RDZV_TIMEOUT_SEC",
             std::to_string(FLAGS_rdzv_timeout).c_str(), 1);

      SetEnvInt("RANK", global_proc_rank);
      SetEnvInt("LOCAL_RANK", local_proc_rank);

      SetEnvInt("WORLD_SIZE", proc_world_size);

      if (!FLAGS_log_dir.empty()) {
        const std::string prefix =
            FLAGS_log_dir + "/rank-" + std::to_string(global_proc_rank);
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
              hostname, global_proc_rank, local_proc_rank, proc_world_size,
              local_proc_rank, run_id.c_str());
      fflush(stderr);
      execvp(train_program.c_str(), train_argv.data());
      perror("exec failed");
      _exit(127);
    }
    running_children.insert(pid);
  }

  bool term_sent = false;
  auto term_time = std::chrono::steady_clock::now();
  while (!running_children.empty()) {
    if (g_signal && !term_sent) {
      exit_code = 128 + g_signal;
      TerminateChildren(running_children);
      term_sent = true;
      term_time = std::chrono::steady_clock::now();
    }
    if (term_sent &&
        std::chrono::steady_clock::now() - term_time > std::chrono::seconds(10))
      TerminateChildren(running_children, SIGKILL);
    int status;
    pid_t child = waitpid(-1, &status, term_sent ? WNOHANG : 0);
    if (child == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    if (child < 0) {
      if (errno == EINTR) {
        continue;
      }
      perror("wait failed");
      if (exit_code == 0) {
        exit_code = 1;
      }
      TerminateChildren(running_children);
      break;
    }

    running_children.erase(child);
    const int child_exit_code = ExitCodeFromStatus(status);
    if (child_exit_code != 0 && exit_code == 0) {
      exit_code = child_exit_code;
      TerminateChildren(running_children);
      term_sent = true;
      term_time = std::chrono::steady_clock::now();
    }
  }

  return exit_code;
}
