#pragma once

#include <cstdlib>
#include <mutex>
#include <string>

#include "glog/logging.h"

namespace infini_train::global {

class GlobalEnv {
public:
    static GlobalEnv &Instance() {
        static GlobalEnv instance;
        return instance;
    }

    void Init(int intra_world_size, int tensor_parallel_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        inter_world_size_ = GetEnvAsInt("WORLD_SIZE", 1);
        intra_world_size_ = intra_world_size;
        tensor_parallel_size_ = tensor_parallel_size;
        initialized_ = true;
    }

    int inter_world_size() const {
        CHECK(initialized_) << "GlobalEnv is not initialized";
        return inter_world_size_;
    }

    int intra_world_size() const {
        CHECK(initialized_) << "GlobalEnv is not initialized";
        return intra_world_size_;
    }

    int tensor_parallel_size() const {
        CHECK(initialized_) << "GlobalEnv is not initialized";
        return tensor_parallel_size_;
    }

private:
    GlobalEnv() = default;
    ~GlobalEnv() = default;

    GlobalEnv(const GlobalEnv &) = delete;
    GlobalEnv &operator=(const GlobalEnv &) = delete;

    static int GetEnvAsInt(const std::string &name, int default_value) {
        const char *value = std::getenv(name.c_str());
        return value ? std::atoi(value) : default_value;
    }

    static std::string GetEnvAsStr(const std::string &name, const std::string &default_value) {
        const char *value = std::getenv(name.c_str());
        return value ? std::string(value) : default_value;
    }

private:
    int inter_world_size_ = 1;
    int intra_world_size_ = 1;
    int tensor_parallel_size_ = 1;
    mutable std::mutex mutex_;
    bool initialized_ = false;
};

inline void InitAllEnv(int data_parallel_size, int tensor_parallel_size) {
    GlobalEnv::Instance().Init(data_parallel_size, tensor_parallel_size);
}
inline int GetIntraWorldSize() { return GlobalEnv::Instance().intra_world_size(); }
inline int GetInterWorldSize() { return GlobalEnv::Instance().inter_world_size(); }
inline int GetTensorParallelSize() { return GlobalEnv::Instance().tensor_parallel_size(); }

} // namespace infini_train::global
