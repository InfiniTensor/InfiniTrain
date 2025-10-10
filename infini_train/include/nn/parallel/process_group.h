#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef USE_NCCL
#include <nccl.h>
#endif

#ifdef USE_CUDA
#include "infini_train/include/common/cuda/common_cuda.h"
#endif
#include "infini_train/include/nn/parallel/reduce_op_type.h"

namespace infini_train {
class Tensor;
class Device;
namespace nn {
class Module;
}
} // namespace infini_train

namespace infini_train::nn::parallel {
class ProcessGroup;

class ProcessGroup {
public:
    explicit ProcessGroup(int comm_size) {
        comms_.resize(comm_size);
        // TODO: init comm
        std::vector<int> device_indices;
        for (int i = 0; i < comm_size; i++) { device_indices.push_back(i); }
        NCCL_CHECK(ncclCommInitAll(comms_.data(), comm_size, device_indices.data()));
    }

    ncclComm_t comm(int idx) const { return comms_.at(idx); }

private:
    std::vector<ncclComm_t> comms_;
};

class ProcessGroupFactory {
public:
    static ProcessGroupFactory *Instance() {
        static std::mutex mutex;
        static std::unique_ptr<ProcessGroupFactory> instance = nullptr;
        if (instance == nullptr) {
            std::lock_guard<std::mutex> lock(mutex);
            if (instance == nullptr) {
                instance = std::make_unique<ProcessGroupFactory>();
            }
        }
        return instance.get();
    }

    void Create(const std::string &name, int comm_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        name_to_group_[name] = std::make_unique<ProcessGroup>(ProcessGroup(comm_size));
    }

    ProcessGroup *Get(const std::string &name) {
        std::lock_guard<std::mutex> lock(mutex_);
        return name_to_group_.at(name).get();
    }

    ProcessGroupFactory() {}

private:
    // TODO(dcj): maybe RWLock later?
    std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<ProcessGroup>> name_to_group_;
};
} // namespace infini_train::nn::parallel
