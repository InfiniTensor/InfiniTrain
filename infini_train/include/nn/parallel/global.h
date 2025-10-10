#pragma once

#include <cstdlib>
#include <mutex>
#include <string>

namespace infini_train::global {

class GlobalEnv {
public:
    static GlobalEnv &Instance() {
        static GlobalEnv instance;
        return instance;
    }

    void Init(int tensor_parallel_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        world_size_ = GetEnvAsInt("WORLD_SIZE", 1);
        tensor_parallel_size_ = tensor_parallel_size;
    }

    int GetWorldSize() const { return world_size_; }

    int GetTensorParallelSize() const { return tensor_parallel_size_; }

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
    int world_size_ = 1;
    int tensor_parallel_size_ = 1;
    mutable std::mutex mutex_;
};

inline void InitAllEnv(int tensor_parallel_size) { GlobalEnv::Instance().Init(tensor_parallel_size); }
inline int GetWorldSize() { return GlobalEnv::Instance().GetWorldSize(); }
inline int GetTensorParallelSize() { return GlobalEnv::Instance().GetTensorParallelSize(); }

} // namespace infini_train::global
