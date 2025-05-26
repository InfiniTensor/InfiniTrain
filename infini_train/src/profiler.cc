#include "infini_train/include/profiler.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif

namespace infini_train {
namespace {
inline std::string GetCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &now_time);
#else
    localtime_r(&now_time, &tm_buf);
#endif
    char buffer[20];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_buf);
    return std::string(buffer);
}
} // namespace

Profiler &Profiler::Instance() {
    static Profiler profiler;
    return profiler;
}

void Profiler::StartRecord(const std::string &name, DeviceType device) {
    cpu_timing_map_[name] = std::chrono::high_resolution_clock::now();

    switch (device) {
    case DeviceType::kCPU:
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        cuda_timing_map_[name] = {reinterpret_cast<void *>(start), reinterpret_cast<void *>(stop)};
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type.";
        break;
    }
}

void Profiler::EndRecord(const std::string &name, DeviceType device) {
    int64_t host_us = 0;
    int64_t device_us = 0;

    auto cpu_start = cpu_timing_map_[name];
    auto cpu_end = std::chrono::high_resolution_clock::now();
    host_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    cpu_timing_map_.erase(name);

    switch (device) {
    case DeviceType::kCPU:
        device_us = host_us;
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        auto it = cuda_timing_map_.find(name);
        if (it != cuda_timing_map_.end()) {
            auto event_pair = it->second;
            cudaEvent_t start = reinterpret_cast<cudaEvent_t>(event_pair.start);
            cudaEvent_t stop = reinterpret_cast<cudaEvent_t>(event_pair.stop);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float elapsed_ms;
            cudaEventElapsedTime(&elapsed_ms, start, stop);
            device_us = static_cast<int64_t>(elapsed_ms * 1000);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cuda_timing_map_.erase(it);
        }
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type.";
        break;
    }

    RecordKernel(name, host_us, device_us);
}

void Profiler::RecordKernel(const std::string &name, int64_t host_us, int64_t device_us) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto &entry = stats_[name];
        entry.host_total_us += host_us;
        entry.device_total_us += device_us;
        entry.count += 1;
    }

    call_records_.emplace_back(
        KernelCallRecord{current_tag_, GetCurrentTimestamp(), name, GetProfileContext().device, host_us, device_us});
}

void Profiler::Reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    stats_.clear();
    call_records_.clear();
    current_tag_ = "Untagged";
}

void Profiler::SetTag(const std::string &tag) { current_tag_ = tag; }

void Profiler::Report(std::ostream &os, SortBy sort_by) const {
    os << "\n--- Profiler Report ---\n";

    std::map<std::string, std::map<std::string, KernelProfileInfo>> grouped_stats;

    for (const auto &rec : call_records_) {
        auto &entry = grouped_stats[rec.tag][rec.name];
        entry.host_total_us += rec.host_us;
        entry.device_total_us += rec.device_us;
        entry.count += 1;
    }

    for (const auto &[tag, kernel_map] : grouped_stats) {
        os << "\nTag: " << tag << "\n";
        os << std::left << std::setw(24) << "Name" << std::setw(10) << "Count" << std::setw(18) << "Host Total(us)"
           << std::setw(10) << "Host %" << std::setw(20) << "Device Total(us)" << std::setw(10) << "Device %"
           << std::setw(16) << "Avg Host(us)" << std::setw(18) << "Avg Device(us)"
           << "\n";

        int64_t host_sum = 0;
        int64_t dev_sum = 0;
        for (const auto &[_, info] : kernel_map) {
            host_sum += info.host_total_us;
            dev_sum += info.device_total_us;
        }

        std::vector<std::pair<std::string, KernelProfileInfo>> records(kernel_map.begin(), kernel_map.end());

        auto compare = [&](const auto &a, const auto &b) {
            const auto &[_, A] = a;
            const auto &[__, B] = b;
            switch (sort_by) {
            case SortBy::HostTimeTotal:
                return A.host_total_us > B.host_total_us;
            case SortBy::HostTimePercentage:
                return A.host_total_us * dev_sum > B.host_total_us * dev_sum;
            case SortBy::HostTimeAverage:
                return A.host_total_us / A.count > B.host_total_us / B.count;
            case SortBy::DeviceTimeTotal:
                return A.device_total_us > B.device_total_us;
            case SortBy::DeviceTimePercentage:
                return A.device_total_us * host_sum > B.device_total_us * host_sum;
            case SortBy::DeviceTimeAverage:
                return A.device_total_us / A.count > B.device_total_us / B.count;
            case SortBy::Count:
                return A.count > B.count;
            case SortBy::NotSorted:
                return false;
            }
            return false;
        };

        if (sort_by != SortBy::NotSorted) {
            std::sort(records.begin(), records.end(), compare);
        }

        for (const auto &[name, info] : records) {
            double host_pct = host_sum > 0 ? 100.0 * info.host_total_us / host_sum : 0.0;
            double dev_pct = dev_sum > 0 ? 100.0 * info.device_total_us / dev_sum : 0.0;
            double avg_host = static_cast<double>(info.host_total_us) / info.count;
            double avg_dev = static_cast<double>(info.device_total_us) / info.count;

            os << std::left << std::setw(24) << name << std::setw(10) << info.count << std::setw(18)
               << info.host_total_us << std::setw(10) << std::fixed << std::setprecision(2) << host_pct << std::setw(20)
               << info.device_total_us << std::setw(10) << std::fixed << std::setprecision(2) << dev_pct
               << std::setw(16) << static_cast<int64_t>(avg_host) << std::setw(18) << static_cast<int64_t>(avg_dev)
               << "\n";
        }
    }
}

void Profiler::Report(const std::string &file_path, SortBy sort_by) const {
    std::ofstream ofs(file_path);
    if (!ofs) {
        LOG(ERROR) << "Failed to open file: " << file_path;
        return;
    }
    Report(ofs, sort_by);
}

void Profiler::PrintRecords(std::ostream &os) const {
    os << "\n--- Kernel Call Log ---\n";
    os << std::left << std::setw(16) << "Tag" << std::setw(8) << "Idx" << std::setw(24) << "Timestamp" << std::setw(24)
       << "Name" << std::setw(10) << "Device" << std::setw(12) << "Host(us)" << std::setw(12) << "Device(us)"
       << "\n";

    std::map<std::string, int> tag_counters;

    for (const auto &rec : call_records_) {
        int idx = tag_counters[rec.tag]++;
        os << std::left << std::setw(16) << rec.tag << std::setw(8) << idx << std::setw(24) << rec.timestamp
           << std::setw(24) << rec.name << std::setw(10) << static_cast<int>(rec.device) << std::setw(12) << rec.host_us
           << std::setw(12) << rec.device_us << "\n";
    }
}

void Profiler::PrintRecords(const std::string &file_path) const {
    std::ofstream ofs(file_path);
    if (!ofs) {
        LOG(ERROR) << "Failed to open file: " << file_path;
        return;
    }
    PrintRecords(ofs);
}

} // namespace infini_train
