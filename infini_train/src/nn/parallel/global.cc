#include "infini_train/include/nn/parallel/global.h"

#include <cstdlib>
#include <string>

#include "glog/logging.h"

namespace {

int GetEnvAsInt(const std::string &name, int default_value) {
    const char *value = std::getenv(name.c_str());
    return value ? std::atoi(value) : default_value;
}

} // namespace

namespace infini_train::nn::parallel::global {

GlobalEnv &GlobalEnv::Instance() {
    static GlobalEnv instance;
    return instance;
}

void GlobalEnv::Init(int nthread_per_process, int tensor_parallel_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    CHECK(!initialized_) << "Repeated initialization of GlobalEnv!";

    world_size_ = GetEnvAsInt("PROC_WORLD_SIZE", 1) * nthread_per_process;
    nproc_per_node_ = GetEnvAsInt("NPROC_PER_NODE", 1);
    global_proc_rank_ = GetEnvAsInt("GLOBAL_PROC_RANK", 0);
    local_proc_rank_ = GetEnvAsInt("LOCAL_PROC_RANK", 0);

    nthread_per_process_ = nthread_per_process;
    tensor_parallel_size_ = tensor_parallel_size;
    data_parallel_size_ = world_size_ / tensor_parallel_size_;
    initialized_ = true;
}

int GlobalEnv::world_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return world_size_;
}

int GlobalEnv::global_proc_rank() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return global_proc_rank_;
}

int GlobalEnv::local_proc_rank() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return local_proc_rank_;
}

int GlobalEnv::nproc_per_node() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return nproc_per_node_;
}

int GlobalEnv::nthread_per_process() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return nthread_per_process_;
}

int GlobalEnv::tensor_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return tensor_parallel_size_;
}

int GlobalEnv::data_parallel_size() const {
    CHECK(initialized_) << "GlobalEnv is not initialized!";
    return data_parallel_size_;
}

} // namespace infini_train::nn::parallel::global
