#include "infini_train/include/nn/parallel/process_group.h"

#include <nccl.h>

#include "glog/logging.h"

namespace {
std::unique_ptr<infini_train::nn::parallel::ProcessGroup> g_pb = nullptr;
}

namespace infini_train::nn::parallel {

void InitProcessGroup(const ProcessGroupConfig &cfg) {
    if (cfg.backend == "nccl") {}
}

ProcessGroupNCCL::ProcessGroupNCCL(const ProcessGroupConfig &cfg) {
    ncclUniqueId id;
    if (cfg.init_method.rfind("env://", 0) == 0 || cfg.init_method.empty()) {
        const char *encoded = std::getenv("NCCL_UNIQUE_ID");
        if (!encoded) {
            LOG(FATAL) << "init_method env:// selected but NCCL_UNIQUE_ID not found";
        }
        memcpy(&id, encoded, sizeof(id));
    }

    int n_gpus = device_list_.size();
    comms_.resize(n_gpus);

    for (int i = 0; i < n_gpus; ++i) {
        cudaSetDevice(device_list_[i]);
        ncclComm_t comm;
        ncclCommInitRank(&comm, cfg.world_size, id, cfg.rank);
        comms_[i] = comm;
    }
}
} // namespace infini_train::nn::parallel
