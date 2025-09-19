#include <string>
#include <sys/wait.h>
#include <vector>
#include <cstdlib>
#include <unistd.h>

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_int32(nnodes, 1, "Total number of nodes");
DEFINE_int32(nproc_per_node, 1, "Number of processes per node");
DEFINE_int32(node_rank, 0, "Rank of this node");
DEFINE_string(rdzv_endpoint, "127.0.0.1:29500", "Rendezvous endpoint (host:port)");

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    CHECK_GE(argc, 2) << "No training prgram specified!";

    std::string train_program = argv[1];
    std::vector<char*> train_argv;
    for (int i = 1; i < argc; ++i) {
        train_argv.push_back(argv[i]);
    }
    train_argv.push_back(nullptr);

    int world_size = FLAGS_nnodes * FLAGS_nproc_per_node;
    std::string master_addr = FLAGS_rdzv_endpoint.substr(0, FLAGS_rdzv_endpoint.find(':'));
    std::string master_port = FLAGS_rdzv_endpoint.substr(FLAGS_rdzv_endpoint.find(':') + 1);

    for (int local_rank = 0; local_rank < FLAGS_nproc_per_node; ++local_rank) {
        pid_t pid = fork();
        if (pid == 0) {
            int global_rank = FLAGS_node_rank * FLAGS_nproc_per_node + local_rank;
            setenv("RANK", std::to_string(global_rank).c_str(), 1);
            setenv("LOCAL_RANK", std::to_string(local_rank).c_str(), 1);
            setenv("WORLD_SIZE", std::to_string(world_size).c_str(), 1);
            setenv("MASTER_ADDR", master_addr.c_str(), 1);
            setenv("MASTER_PORT", master_port.c_str(), 1);

            execvp(train_program.c_str(), train_argv.data());
            perror("exec failed");
            exit(1);
        }
    }

    for (int i = 0; i < FLAGS_nproc_per_node; ++i) {
        int status;
        wait(&status);
    }

    return 0;
}
