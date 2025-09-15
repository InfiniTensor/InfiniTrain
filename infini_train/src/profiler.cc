#include "infini_train/include/profiler.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "glog/logging.h"

#ifdef USE_CUDA
#include "infini_train/include/common/cuda/common_cuda.h"
#endif
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

int GetRank(DeviceType device) {
    if (device == DeviceType::kCPU) {
        return 0;
    }

    // Assume single-node setting, rank == device_id
    int device_id = 0;
#ifdef USE_CUDA
    CUDA_CHECK(cudaGetDevice(&device_id));
#endif
    return device_id;
}

#ifdef USE_CUDA
cudaStream_t GetCudaStream() {
    int device_id = GetRank(DeviceType::kCUDA);
    // TODO(zbl): support multi-stream on single device
    return dynamic_cast<const CudaDevice *>(
               DeviceManager::Instance()->GetDevice(DeviceType::kCUDA, static_cast<int8_t>(device_id)))
        ->Stream();
}
#endif

void Profiler::StartRecord(const std::string &name, DeviceType device) {
    if (g_profiling_depth++ > 0) {
        return;
    }
    cpu_timing_map_[name] = std::chrono::high_resolution_clock::now();

    switch (device) {
    case DeviceType::kCPU:
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        auto it = cuda_timing_map_.find(name);
        if (it != cuda_timing_map_.end()) {
            // Make sure there are no conflicts
            CUDA_CHECK(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(it->second.start)));
            CUDA_CHECK(cudaEventDestroy(reinterpret_cast<cudaEvent_t>(it->second.stop)));
            cuda_timing_map_.erase(it);
        }

        cudaEvent_t start, stop;
        cudaStream_t stream = GetCudaStream();
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, stream));
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
    if (--g_profiling_depth > 0) {
        return;
    }
    int64_t host_us = 0, device_us = 0;
    int64_t peak_mem_mb = 0;
    std::string device_str = "cpu";
    int rank = GetRank(device);

    switch (device) {
    case DeviceType::kCPU:
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        auto it = cuda_timing_map_.find(name);
        if (it != cuda_timing_map_.end()) {
            auto event_pair = it->second;
            cudaEvent_t start = reinterpret_cast<cudaEvent_t>(event_pair.start);
            cudaEvent_t stop = reinterpret_cast<cudaEvent_t>(event_pair.stop);
            cudaStream_t stream = GetCudaStream();
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed_ms = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            device_us = static_cast<int64_t>(elapsed_ms * 1000);
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            cuda_timing_map_.erase(it);

            cudaMemPool_t pool;
            size_t peak_bytes = 0;
            if (cudaDeviceGetDefaultMemPool(&pool, rank) == cudaSuccess
                && cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &peak_bytes) == cudaSuccess) {
                peak_mem_mb = static_cast<int64_t>(peak_bytes) / (1024 * 1024);
            } else {
                LOG(FATAL) << "cudaMemPool not supported.";
            }
            device_str = "cuda:" + std::to_string(rank);
        } else {
            LOG(FATAL) << "Start time of " + name + " is not recorded.";
        }
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type.";
        break;
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
    std::map<int64_t, std::map<std::string, std::map<std::string, KernelProfileInfo>>> grouped_stats;

    for (const auto &rec : call_records_) {
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
            for (const auto &rec : call_records_) {
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
    std::map<int64_t, std::map<std::string, std::vector<const KernelCallRecord *>>> grouped;

    for (const auto &rec : call_records_) { grouped[rec.rank][rec.tag].push_back(&rec); }

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
