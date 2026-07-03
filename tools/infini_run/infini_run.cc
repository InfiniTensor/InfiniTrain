#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_int32(nnodes, 1, "Total number of nodes");
DEFINE_int32(nproc_per_node, 1, "Number of processes per node");
DEFINE_int32(node_rank, 0, "Rank of this node");
DEFINE_string(rdzv_endpoint, "127.0.0.1:29500", "Rendezvous endpoint (host:port)");

namespace {

bool IsLauncherFlag(const std::string &flag) {
    return flag == "--nnodes" || flag == "--nproc_per_node" || flag == "--node_rank" || flag == "--rdzv_endpoint";
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

    for (int local_proc_rank = 0; local_proc_rank < FLAGS_nproc_per_node; ++local_proc_rank) {
        pid_t pid = fork();
        if (pid == 0) {
            int global_proc_rank = FLAGS_node_rank * FLAGS_nproc_per_node + local_proc_rank;
            SetEnvInt("NNODES", FLAGS_nnodes);
            SetEnvInt("NPROC_PER_NODE", FLAGS_nproc_per_node);
            SetEnvInt("LOCAL_WORLD_SIZE", FLAGS_nproc_per_node);

            setenv("MASTER_ADDR", master_addr.c_str(), 1);
            setenv("MASTER_PORT", master_port.c_str(), 1);

            SetEnvInt("GLOBAL_PROC_RANK", global_proc_rank);
            SetEnvInt("LOCAL_PROC_RANK", local_proc_rank);
            SetEnvInt("RANK", global_proc_rank);
            SetEnvInt("LOCAL_RANK", local_proc_rank);

            SetEnvInt("PROC_WORLD_SIZE", proc_world_size);
            SetEnvInt("WORLD_SIZE", proc_world_size);
            SetEnvInt("GROUP_RANK", FLAGS_node_rank);
            SetEnvInt("ROLE_RANK", global_proc_rank);
            SetEnvInt("ROLE_WORLD_SIZE", proc_world_size);

            execvp(train_program.c_str(), train_argv.data());
            perror("exec failed");
            exit(1);
        }
    }

    int exit_code = 0;
    for (int i = 0; i < FLAGS_nproc_per_node; ++i) {
        int status;
        pid_t child = wait(&status);
        if (child < 0) {
            perror("wait failed");
            return 1;
        }

        if (WIFEXITED(status)) {
            int child_exit_code = WEXITSTATUS(status);
            if (child_exit_code != 0 && exit_code == 0) {
                exit_code = child_exit_code;
            }
        } else if (WIFSIGNALED(status)) {
            int signal = WTERMSIG(status);
            if (exit_code == 0) {
                exit_code = 128 + signal;
            }
        } else if (exit_code == 0) {
            exit_code = 1;
        }
    }

    return exit_code;
}
