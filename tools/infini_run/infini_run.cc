#include <cstdlib>
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <cstdio>

#ifdef USE_NCCL
#include <nccl.h>
#endif

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_int32(nnodes, 1, "Total number of nodes");
DEFINE_int32(nproc_per_node, 1, "Number of processes per node");
DEFINE_int32(node_rank, 0, "Rank of this node");
DEFINE_string(rdzv_endpoint, "127.0.0.1:29500", "Rendezvous endpoint (host:port)");

#ifdef USE_NCCL
std::string NcclIdToString(const ncclUniqueId& id) {
    std::ostringstream oss;
    for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
        oss << std::hex << std::uppercase << std::setw(2) << std::setfill('0') << (int)(unsigned char)id.internal[i];
    }
    return oss.str();
}
#endif

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    CHECK_GE(argc, 2) << "No training prgram specified!";

    std::string train_program = argv[1];
    std::vector<char *> train_argv;
    for (int i = 1; i < argc; ++i) { train_argv.push_back(argv[i]); }
    train_argv.push_back(nullptr);

    int world_size = FLAGS_nnodes * FLAGS_nproc_per_node;
    std::string master_addr = FLAGS_rdzv_endpoint.substr(0, FLAGS_rdzv_endpoint.find(':'));
    std::string master_port = FLAGS_rdzv_endpoint.substr(FLAGS_rdzv_endpoint.find(':') + 1);

    const char* nccl_id_path = "/data/shared/InfiniTrain-dev/data/nccl_id.bin";
    const char* nccl_id_tmp_path = "/data/shared/InfiniTrain-dev/data/nccl_id.tmp";

#ifdef USE_NCCL
    if (FLAGS_node_rank == 0) {
        ncclUniqueId id;
        ncclGetUniqueId(&id);

        std::ofstream ofs(nccl_id_tmp_path, std::ios::binary);
        ofs.write((char *)&id, sizeof(id));
        ofs.close();

        // atomic operation
        rename(nccl_id_tmp_path, nccl_id_path);
    }
#endif

    for (int local_proc_rank = 0; local_proc_rank < FLAGS_nproc_per_node; ++local_proc_rank) {
        pid_t pid = fork();
        if (pid == 0) {
            int global_proc_rank = FLAGS_node_rank * FLAGS_nproc_per_node + local_proc_rank;
            setenv("GLOBAL_PROC_RANK", std::to_string(global_proc_rank).c_str(), 1);
            setenv("LOCAL_PROC_RANK", std::to_string(local_proc_rank).c_str(), 1);
            setenv("PROC_WORLD_SIZE", std::to_string(world_size).c_str(), 1);
            setenv("NPROC_PER_NODE", std::to_string(FLAGS_nproc_per_node).c_str(), 1);
            setenv("MASTER_ADDR", master_addr.c_str(), 1);
            setenv("MASTER_PORT", master_port.c_str(), 1);

#ifdef USE_NCCL
            struct stat st;
            while (stat(nccl_id_path, &st) != 0) {
                usleep(1000);
            }

            ncclUniqueId id;
            std::ifstream ifs(nccl_id_path, std::ios::binary);
            ifs.read((char*)&id, sizeof(id));

            std::string id_str = NcclIdToString(id);
            setenv("NCCL_UNIQUE_ID", id_str.c_str(), 1);
#endif

            execvp(train_program.c_str(), train_argv.data());
            perror("exec failed");
            exit(1);
        }
    }

    for (int i = 0; i < FLAGS_nproc_per_node; ++i) {
        int status;
        wait(&status);
    }

#ifdef USE_NCCL
    if (FLAGS_node_rank == 0) {
        std::remove(nccl_id_path);
    }
#endif

    return 0;
}
