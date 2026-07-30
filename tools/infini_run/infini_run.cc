#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <sys/wait.h>
#include <system_error>
#include <unistd.h>
#include <unordered_set>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_int32(nnodes, 1, "Total number of nodes");
DEFINE_int32(nproc_per_node, 1, "Number of processes per node");
DEFINE_int32(node_rank, 0, "Rank of this node");
DEFINE_string(rdzv_endpoint, "127.0.0.1:29500", "Rendezvous endpoint (host:port)");

DEFINE_string(rdzv_id, "", "Unique job ID shared by all nodes");

namespace {

bool IsLauncherFlag(const std::string &flag) {
    return flag == "--nnodes" || flag == "--nproc_per_node" || flag == "--node_rank" || flag == "--rdzv_endpoint"
        || flag == "--rdzv_id";
}

int FindTrainProgramIndex(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--") {
            return i;
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
    for (int i = 1; i < train_program_index; ++i) { launcher_argv.push_back(argv[i]); }
    launcher_argv.push_back(nullptr);
    return launcher_argv;
}

void SetEnvInt(const char *name, int value) {
    const auto value_str = std::to_string(value);
    setenv(name, value_str.c_str(), 1);
}

void TerminateChildren(const std::unordered_set<pid_t> &child_pids) {
    for (pid_t child_pid : child_pids) { kill(child_pid, SIGTERM); }
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

void CleanupRunUniqueIdFiles(const std::string &run_id) {
    const std::string prefix = "cclUniqueId_" + run_id + "_";
    for (const auto &entry : std::filesystem::directory_iterator(std::filesystem::current_path())) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string filename = entry.path().filename().string();
        if (filename.rfind(prefix, 0) == 0) {
            std::error_code ec;
            std::filesystem::remove(entry.path(), ec);
            if (ec) {
                LOG(WARNING) << "Failed to remove unique-id file " << entry.path() << ": " << ec.message();
            }
        }
    }
}

std::string GenerateLocalRunId() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::to_string(getpid()) + "_" + std::to_string(now);
}

} // namespace

int main(int argc, char **argv) {
    const int train_program_index = FindTrainProgramIndex(argc, argv);
    std::vector<char *> launcher_argv = BuildLauncherArgv(train_program_index, argv);
    int launcher_argc = static_cast<int>(launcher_argv.size()) - 1;
    char **launcher_argv_ptr = launcher_argv.data();
    gflags::ParseCommandLineFlags(&launcher_argc, &launcher_argv_ptr, true);
    google::InitGoogleLogging(argv[0]);

    CHECK_GT(FLAGS_nnodes, 0) << "nnodes must be positive";
    CHECK_GT(FLAGS_nproc_per_node, 0) << "nproc_per_node must be positive";
    CHECK_NE(FLAGS_rdzv_endpoint.find(':'), std::string::npos) << "rdzv_endpoint must be host:port";
    CHECK(FLAGS_nnodes == 1 || !FLAGS_rdzv_id.empty())
        << "rdzv_id must be set to the same unique job ID on every node for multi-node training";

    CHECK_LT(train_program_index, argc) << "No training program specified!";

    std::string train_program = argv[train_program_index];
    CHECK_NE(train_program, "--") << "Explicit '--' separator is not supported; pass the training program directly "
                                     "after infini_run launcher flags";
    std::vector<char *> train_argv;
    for (int i = train_program_index; i < argc; ++i) { train_argv.push_back(argv[i]); }
    train_argv.push_back(nullptr);

    int proc_world_size = FLAGS_nnodes * FLAGS_nproc_per_node;
    std::string master_addr = FLAGS_rdzv_endpoint.substr(0, FLAGS_rdzv_endpoint.find(':'));
    std::string master_port = FLAGS_rdzv_endpoint.substr(FLAGS_rdzv_endpoint.find(':') + 1);
    const std::string run_id = FLAGS_rdzv_id.empty() ? GenerateLocalRunId() : FLAGS_rdzv_id;

    std::unordered_set<pid_t> running_children;
    int exit_code = 0;

    for (int local_proc_rank = 0; local_proc_rank < FLAGS_nproc_per_node; ++local_proc_rank) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork failed");
            exit_code = 1;
            TerminateChildren(running_children);
            break;
        }
        if (pid == 0) {
            int global_proc_rank = FLAGS_node_rank * FLAGS_nproc_per_node + local_proc_rank;
            SetEnvInt("LOCAL_WORLD_SIZE", FLAGS_nproc_per_node);

            setenv("MASTER_ADDR", master_addr.c_str(), 1);
            setenv("MASTER_PORT", master_port.c_str(), 1);
            setenv("INFINI_RUN_ID", run_id.c_str(), 1);

            SetEnvInt("RANK", global_proc_rank);
            SetEnvInt("LOCAL_RANK", local_proc_rank);

            SetEnvInt("WORLD_SIZE", proc_world_size);

            execvp(train_program.c_str(), train_argv.data());
            perror("exec failed");
            exit(1);
        }
        running_children.insert(pid);
    }

    while (!running_children.empty()) {
        int status;
        pid_t child = wait(&status);
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
        }
    }

    if (FLAGS_nnodes == 1) {
        CleanupRunUniqueIdFiles(run_id);
    }
    return exit_code;
}
