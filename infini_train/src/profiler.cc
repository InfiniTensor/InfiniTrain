#include "infini_train/include/profiler.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"

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

int GetRank(Device::DeviceType device) {
    auto impl = core::GetDeviceGuardImpl(device);
    return impl->GetDevice().index();
}

void Profiler::StartRecord(const std::string &name, Device::DeviceType device) {
    if (g_profiling_depth++ > 0) {
        return;
    }
    cpu_timing_map_[name] = std::chrono::high_resolution_clock::now();

    if (device == Device::DeviceType::kCPU) {
        return;
    }

    auto *impl = core::GetDeviceGuardImpl(device);
    const int device_id = impl->GetDevice().index();
    auto current_device = Device(device, static_cast<int8_t>(device_id));
    auto *stream = impl->GetStream(current_device);

    auto it = device_timing_map_.find(name);
    if (it != device_timing_map_.end()) {
        impl->EventDestroy(it->second.start);
        impl->EventDestroy(it->second.stop);
        device_timing_map_.erase(it);
    }

    core::Event *start = nullptr;
    core::Event *stop = nullptr;
    impl->EventCreate(&start);
    impl->EventCreate(&stop);

    // Make sure the compute stream has done waiting, and ready for the execution of next op
    impl->SynchronizeStream(stream);
    // Start record after waiting
    cpu_timing_map_[name] = std::chrono::high_resolution_clock::now();
    impl->EventRecord(start, stream);
    device_timing_map_[name] = {start, stop};
}

void Profiler::EndRecord(const std::string &name, Device::DeviceType device) {
    if (--g_profiling_depth > 0) {
        return;
    }
    int64_t host_us = 0, device_us = 0;
    int64_t peak_mem_mb = 0;
    std::string device_str = "cpu";
    int rank = GetRank(device);

    if (device != Device::DeviceType::kCPU) {
        auto *impl = core::GetDeviceGuardImpl(device);
        auto current_device = Device(device, static_cast<int8_t>(rank));
        auto *stream = impl->GetStream(current_device);

        auto it = device_timing_map_.find(name);
        if (it == device_timing_map_.end()) {
            LOG(FATAL) << "Start time of " + name + " is not recorded.";
        }

        auto event_pair = it->second;
        impl->EventRecord(event_pair.stop, stream);
        impl->EventSynchronize(event_pair.stop);
        device_us = static_cast<int64_t>(impl->EventElapsedTime(event_pair.start, event_pair.stop) * 1000.0f);
        impl->EventDestroy(event_pair.start);
        impl->EventDestroy(event_pair.stop);
        device_timing_map_.erase(it);

        auto [peak_used_mb, peak_reserved_mb] = impl->GetMemPoolPeakMB(current_device);
        (void)peak_used_mb;
        peak_mem_mb = static_cast<int64_t>(peak_reserved_mb);
        device_str = current_device.ToString();
    }

    auto cpu_start = cpu_timing_map_[name];
    auto cpu_end = std::chrono::high_resolution_clock::now();
    host_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    cpu_timing_map_.erase(name);

    RecordKernel(name, rank, device_str, host_us, device_us, peak_mem_mb);
}

void Profiler::RecordKernel(const std::string &name, const int &rank, const std::string &device, int64_t host_us,
                            int64_t device_us, int64_t max_device_mem_usage_mb) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        call_records_.emplace_back(KernelCallRecord{current_tag_, GetCurrentTimestamp(), rank, name, device, host_us,
                                                    device_us, max_device_mem_usage_mb});
    }
}

void Profiler::Reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    call_records_.clear();
    current_tag_ = "Untagged";
}

void Profiler::SetTag(const std::string &tag) { current_tag_ = tag; }

void Profiler::ReportGroupedByRank(std::function<std::ostream &(int64_t)> get_os, SortBy sort_by) const {
    std::vector<KernelCallRecord> records_snapshot;
    {
        // Prevent call_records_ from being modified by other threads
        std::lock_guard<std::mutex> lock(mtx_);
        records_snapshot = call_records_;
    }

    std::map<int64_t, std::map<std::string, std::map<std::string, KernelProfileInfo>>> grouped_stats;

    for (const auto &rec : records_snapshot) {
        auto &entry = grouped_stats[rec.rank][rec.tag][rec.name];
        entry.host_total_us += rec.host_us;
        entry.device_total_us += rec.device_us;
        entry.count += 1;
    }

    for (const auto &[rank, tag_map] : grouped_stats) {
        std::ostream &os = get_os(rank);
        if (!os) {
            continue;
        }

        os << "\n=== Profiler Report for Rank " << rank << " ===\n";

        for (const auto &[tag, kernel_map] : tag_map) {
            os << "\nTag: " << tag << "\n";

            // Peak memory usage by tag
            int64_t tag_peak_mb = 0;
            for (const auto &rec : records_snapshot) {
                if (rec.rank == rank && rec.tag == tag) {
                    tag_peak_mb = std::max(tag_peak_mb, rec.max_device_mem_usage_mb);
                }
            }
            os << "Peak Device Memory Usage: " << tag_peak_mb << " MB\n";

            os << std::left << std::setw(40) << "Name" << std::setw(10) << "Count" << std::setw(18) << "Host Total(us)"
               << std::setw(10) << "Host %" << std::setw(20) << "Device Total(us)" << std::setw(10) << "Device %"
               << std::setw(16) << "Avg Host(us)" << std::setw(18) << "Avg Device(us)"
               << "\n";

            int64_t host_sum = 0, dev_sum = 0;
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

                os << std::left << std::setw(40) << name << std::setw(10) << info.count << std::setw(18)
                   << info.host_total_us << std::setw(10) << std::fixed << std::setprecision(2) << host_pct
                   << std::setw(20) << info.device_total_us << std::setw(10) << std::fixed << std::setprecision(2)
                   << dev_pct << std::setw(16) << static_cast<int64_t>(avg_host) << std::setw(18)
                   << static_cast<int64_t>(avg_dev) << "\n";
            }
        }
    }
}

void Profiler::Report(std::ostream &os, SortBy sort_by) const {
    auto get_stream = [&](int64_t) -> std::ostream & { return os; };
    ReportGroupedByRank(get_stream, sort_by);
}

void Profiler::Report(const std::string &file_prefix, SortBy sort_by) const {
    std::map<int64_t, std::ofstream> file_map;

    auto get_stream = [&](int64_t rank) -> std::ostream & {
        auto &file = file_map[rank];
        if (!file.is_open()) {
            std::string filename = std::format("{}.rank{}", file_prefix, rank);
            file.open(filename);
            if (!file) {
                LOG(ERROR) << "Failed to open file: " << filename;
                static std::ofstream null_ofs;
                return null_ofs;
            }
        }
        return file;
    };

    ReportGroupedByRank(get_stream, sort_by);
}

void Profiler::PrintRecordsGroupedByRank(std::function<std::ostream &(int64_t)> get_os) const {
    std::vector<KernelCallRecord> records_snapshot;
    {
        // Prevent call_records_ from being modified by other threads
        std::lock_guard<std::mutex> lock(mtx_);
        records_snapshot = call_records_;
    }

    std::map<int64_t, std::map<std::string, std::vector<const KernelCallRecord *>>> grouped;

    for (const auto &rec : records_snapshot) { grouped[rec.rank][rec.tag].push_back(&rec); }

    for (const auto &[rank, tag_map] : grouped) {
        std::ostream &os = get_os(rank);
        if (!os) {
            continue;
        }

        os << "\n=== Kernel Call Log for Rank " << rank << " ===\n";

        for (const auto &[tag, records] : tag_map) {
            os << "\nTag: " << tag << "\n";

            os << std::left << std::setw(8) << "Idx" << std::setw(24) << "Timestamp" << std::setw(40) << "Name"
               << std::setw(10) << "Device" << std::setw(12) << "Host(us)" << std::setw(12) << "Device(us)"
               << std::setw(16) << "Peak Mem(MB)"
               << "\n";

            for (size_t idx = 0; idx < records.size(); ++idx) {
                const auto &rec = *records[idx];
                os << std::left << std::setw(8) << idx << std::setw(24) << rec.timestamp << std::setw(40) << rec.name
                   << std::setw(10) << rec.device << std::setw(12) << rec.host_us << std::setw(12) << rec.device_us
                   << std::setw(16) << rec.max_device_mem_usage_mb << "\n";
            }
        }
    }
}

void Profiler::PrintRecords(std::ostream &os) const {
    auto get_stream = [&](int64_t) -> std::ostream & { return os; };
    PrintRecordsGroupedByRank(get_stream);
}

void Profiler::PrintRecords(const std::string &file_prefix) const {
    std::map<int64_t, std::ofstream> file_map;

    auto get_stream = [&](int64_t rank) -> std::ostream & {
        auto &file = file_map[rank];
        if (!file.is_open()) {
            std::string filename = std::format("{}.rank{}", file_prefix, rank);
            file.open(filename);
            if (!file) {
                LOG(ERROR) << "Failed to open file: " << filename;
                static std::ofstream null_ofs;
                return null_ofs;
            }
        }
        return file;
    };

    PrintRecordsGroupedByRank(get_stream);
}

} // namespace infini_train
