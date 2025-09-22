#include "infini_train/include/nn/parallel/process_group.h"

#include <cstring>
#include <memory>
#include <nccl.h>
#include <sys/types.h>

#include "glog/logging.h"

namespace {
std::shared_ptr<infini_train::nn::parallel::ProcessGroup> g_pb = nullptr;
}

namespace infini_train::nn::parallel {

bool ProcessGroupNCCL::IsHexString(const std::string &s) {
    if (s.empty()) {
        return false;
    }
    for (char c : s) {
        bool ok = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
        if (!ok) {
            return false;
        }
    }
    return true;
}

std::vector<uint8_t> ProcessGroupNCCL::HexToBytes(const std::string &hex) {
    std::vector<uint8_t> out;
    out.reserve((hex.size() + 1) / 2);
    size_t i = 0;

    if (hex.size() % 2 != 0) {
        std::string hh = std::string("0") + hex;
        return HexToBytes(hh);
    }
    auto hex_val = [](char c) -> int {
        if (c >= '0' && c <= '9') {
            return c - '0';
        } else if (c >= 'a' && c <= 'f') {
            return c - 'a' + 10;
        } else if (c >= 'A' && c <= 'F') {
            return c - 'A' + 10;
        }
        return 0;
    };
    for (; i + 1 < hex.size(); i += 2) {
        uint8_t hi = (uint8_t)hex_val(hex[i]);
        uint8_t lo = (uint8_t)hex_val(hex[i + 1]);
        out.push_back((uint8_t)((hi << 4) | lo));
    }
    return out;
}

std::shared_ptr<ProcessGroup> InitProcessGroup(const ProcessGroupConfig &cfg) {
    if (g_pb) {
        LOG(WARNING) << "ProcessGroup already initialized. Returning existing instance.";
        return g_pb;
    }

    if (cfg.backend == "nccl") {
        auto pg = std::make_shared<ProcessGroupNCCL>(ProcessGroupNCCL(cfg));
        g_pb = pg;
        return pg;
    } else {
        LOG(FATAL) << "Unsupported backend: " << cfg.backend;
    }
    return nullptr;
}

void DestroyProcessGroup() {
    if (!g_pb) {
        return;
    }
    g_pb.reset();
}

ProcessGroupNCCL::ProcessGroupNCCL(const ProcessGroupConfig &cfg) {
    world_size = cfg.world_size;
    rank = cfg.rank;

    ncclUniqueId id;
    bool have_id = false;

    if (cfg.init_method.rfind("env://", 0) == 0 || cfg.init_method.empty()) {
        const char *env = std::getenv("NCCL_UNIQUE_ID");
        if (!env) {
            LOG(FATAL) << "init_method env:// selected but NCCL_UNIQUE_ID not found";
        } else {
            std::string s(env);
            if (s.size() == sizeof(ncclUniqueId)) {
                memcpy(&id, s.data(), sizeof(ncclUniqueId));
                have_id = true;
            } else if (IsHexString(s) && s.size() == sizeof(ncclUniqueId) * 2) {
                auto bytes = HexToBytes(s);
                if (bytes.size() == sizeof(ncclUniqueId)) {
                    memcpy(&id, bytes.data(), sizeof(ncclUniqueId));
                    have_id = true;
                }
            } else {
                LOG(FATAL) << "NCCL_UNIQUE_ID found in environment but is not binary or hex of expected length.";
            }
        }
    } else if (cfg.init_method.rfind("tcp://", 0) == 0) {
        LOG(FATAL) << "init_method tcp:// not supported yet";
    } else {
        LOG(FATAL) << "Unsupported init_method: " << cfg.init_method;
    }

    if (!have_id) {
        LOG(FATAL) << "Failed to obtain ncclUniqueId from init_method=" << cfg.init_method;
    }

    ncclComm_t comm = nullptr;
    ncclCommInitRank(&comm, cfg.world_size, id, cfg.rank);
    comm_ = comm;
}

ProcessGroupNCCL::~ProcessGroupNCCL() {
    if (comm_ != nullptr) {
        ncclCommDestroy(comm_);
    }
    comm_ = nullptr;
}
} // namespace infini_train::nn::parallel
