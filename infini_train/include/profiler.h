#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <string>

#include "glog/logging.h"

#include "infini_train/include/device.h"

namespace infini_train {

inline thread_local int g_profiling_depth = 0;

struct ProfileContext {
    std::string name;
    DeviceType device;
};

inline thread_local ProfileContext g_profile_context;

inline void SetProfileContext(const std::string &name, DeviceType device) {
    if (g_profiling_depth == 0) {
        g_profile_context.name = name;
        g_profile_context.device = device;
    }
}

inline const ProfileContext &GetProfileContext() { return g_profile_context; }

struct KernelProfileInfo {
    int64_t host_total_us = 0;
    int64_t device_total_us = 0;
    int count = 0;
};

struct KernelCallRecord {
    std::string tag;
    std::string timestamp;
    int64_t rank;
    std::string name;
    std::string device;
    int64_t host_us;
    int64_t device_us;
    int64_t max_device_mem_usage_mb;
};

class Profiler {
public:
    enum class SortBy {
        HostTimeTotal,
        HostTimePercentage,
        HostTimeAverage,
        DeviceTimeTotal,
        DeviceTimePercentage,
        DeviceTimeAverage,
        Count,
        NotSorted
    };

    static Profiler &Instance();

    void StartRecord(const std::string &name, DeviceType device);
    void EndRecord(const std::string &name, DeviceType device);

    void Report(std::ostream &os = std::cout, SortBy sort_by = SortBy::NotSorted) const;
    void Report(const std::string &file_path, SortBy sort_by = SortBy::NotSorted) const;
    void PrintRecords(std::ostream &os = std::cout) const;
    void PrintRecords(const std::string &file_path) const;

    void Reset();
    void SetTag(const std::string &tag);

private:
    void RecordKernel(const std::string &name, const int &rank, const std::string &device, int64_t host_us,
                      int64_t device_us = 0, int64_t max_device_mem_usage_mb = 0);
    void ReportGroupedByRank(std::function<std::ostream &(int64_t)> get_os, SortBy sort_by) const;
    void PrintRecordsGroupedByRank(std::function<std::ostream &(int64_t)> get_os) const;

    std::mutex mtx_;
    std::vector<KernelCallRecord> call_records_;
    std::string current_tag_ = "Untagged";

#ifdef USE_CUDA
    struct EventPair {
        void *start;
        void *stop;
    };
#endif

    // thread-local tracking
    thread_local static inline std::map<std::string, std::chrono::high_resolution_clock::time_point> cpu_timing_map_;

#ifdef USE_CUDA
    thread_local static inline std::map<std::string, EventPair> cuda_timing_map_;
#endif
};
} // namespace infini_train
