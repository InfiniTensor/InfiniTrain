#include "infini_train/include/utils/precision_checker.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_map>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/utils/precision_check_context.h"

namespace infini_train::utils {

namespace {

// Simple MD5 implementation
class MD5 {
public:
    MD5() { Init(); }

    void Update(const void *data, size_t len) {
        const uint8_t *ptr = static_cast<const uint8_t *>(data);
        size_t buffer_space = 64 - buffer_len_;

        if (len >= buffer_space) {
            memcpy(buffer_ + buffer_len_, ptr, buffer_space);
            Transform(buffer_);
            ptr += buffer_space;
            len -= buffer_space;
            total_len_ += buffer_space;
            buffer_len_ = 0;

            while (len >= 64) {
                Transform(ptr);
                ptr += 64;
                len -= 64;
                total_len_ += 64;
            }
        }

        memcpy(buffer_ + buffer_len_, ptr, len);
        buffer_len_ += len;
        total_len_ += len;
    }

    std::string Finalize() {
        uint8_t padding[64] = {0x80};
        uint64_t bits = total_len_ * 8;

        size_t pad_len = (buffer_len_ < 56) ? (56 - buffer_len_) : (120 - buffer_len_);
        Update(padding, pad_len);

        uint8_t len_bytes[8];
        for (int i = 0; i < 8; ++i) { len_bytes[i] = (bits >> (i * 8)) & 0xff; }
        Update(len_bytes, 8);

        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        for (int i = 0; i < 4; ++i) { oss << std::setw(2) << ((state_[0] >> (i * 8)) & 0xff); }
        for (int i = 0; i < 4; ++i) { oss << std::setw(2) << ((state_[1] >> (i * 8)) & 0xff); }
        for (int i = 0; i < 4; ++i) { oss << std::setw(2) << ((state_[2] >> (i * 8)) & 0xff); }
        for (int i = 0; i < 4; ++i) { oss << std::setw(2) << ((state_[3] >> (i * 8)) & 0xff); }
        return oss.str();
    }

private:
    void Init() {
        state_[0] = 0x67452301;
        state_[1] = 0xefcdab89;
        state_[2] = 0x98badcfe;
        state_[3] = 0x10325476;
        buffer_len_ = 0;
        total_len_ = 0;
    }

    static uint32_t F(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
    static uint32_t G(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
    static uint32_t H(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
    static uint32_t I(uint32_t x, uint32_t y, uint32_t z) { return y ^ (x | ~z); }
    static uint32_t RotateLeft(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

    void Transform(const uint8_t *block) {
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t x[16];
        for (int i = 0; i < 16; ++i) {
            x[i] = block[i * 4] | (block[i * 4 + 1] << 8) | (block[i * 4 + 2] << 16) | (block[i * 4 + 3] << 24);
        }

        static const uint32_t k[]
            = {0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
               0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
               0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
               0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
               0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
               0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
               0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
               0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
        static const int s[] = {7,  12, 17, 22, 7,  12, 17, 22, 7,  12, 17, 22, 7,  12, 17, 22, 5,  9,  14, 20, 5,  9,
                                14, 20, 5,  9,  14, 20, 5,  9,  14, 20, 4,  11, 16, 23, 4,  11, 16, 23, 4,  11, 16, 23,
                                4,  11, 16, 23, 6,  10, 15, 21, 6,  10, 15, 21, 6,  10, 15, 21, 6,  10, 15, 21};

        for (int i = 0; i < 64; ++i) {
            uint32_t f, g;
            if (i < 16) {
                f = F(b, c, d);
                g = i;
            } else if (i < 32) {
                f = G(b, c, d);
                g = (5 * i + 1) % 16;
            } else if (i < 48) {
                f = H(b, c, d);
                g = (3 * i + 5) % 16;
            } else {
                f = I(b, c, d);
                g = (7 * i) % 16;
            }
            uint32_t temp = d;
            d = c;
            c = b;
            b = b + RotateLeft(a + f + k[i] + x[g], s[i]);
            a = temp;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
    }

    uint32_t state_[4];
    uint8_t buffer_[64];
    size_t buffer_len_;
    uint64_t total_len_;
};

std::string ComputeMD5(const void *data, size_t size) {
    MD5 md5;
    md5.Update(data, size);
    return md5.Finalize();
}

// Baseline storage
std::unordered_map<std::string, std::string> &GetBaseline() {
    static std::unordered_map<std::string, std::string> baseline;
    static bool loaded = false;
    static std::mutex load_mutex;

    if (!loaded) {
        std::lock_guard<std::mutex> lock(load_mutex);
        if (!loaded) {
            const auto &config = nn::parallel::global::GlobalEnv::Instance().GetPrecisionCheckConfig();
            if (!config.baseline_path.empty()) {
                std::ifstream file(config.baseline_path);
                if (!file.is_open()) {
                    std::cerr << "[PrecisionCheck] Failed to open baseline file: " << config.baseline_path << std::endl;
                } else {
                    std::string line;
                    while (std::getline(file, line)) {
                        // Try format 1: key|md5
                        auto pipe_pos = line.rfind('|');
                        if (pipe_pos != std::string::npos) {
                            std::string key = line.substr(0, pipe_pos);
                            std::string md5 = line.substr(pipe_pos + 1);
                            baseline[key] = md5;
                        } else {
                            // Try format 2: simple log format with "md5="
                            auto md5_pos = line.find("md5=");
                            if (md5_pos != std::string::npos) {
                                // Extract md5 value
                                std::string md5 = line.substr(md5_pos + 4);

                                // Extract key: find text between "][PrecisionCheck] " and ": md5="
                                auto check_pos = line.find("][PrecisionCheck] ");
                                if (check_pos != std::string::npos) {
                                    size_t key_start = check_pos + 18; // length of "][PrecisionCheck] "
                                    size_t key_end = line.find(": md5=", key_start);
                                    if (key_end != std::string::npos) {
                                        std::string key = line.substr(key_start, key_end - key_start);
                                        baseline[key] = md5;
                                    }
                                }
                            }
                        }
                    }
                    std::cout << "[PrecisionCheck] Loaded " << baseline.size() << " baseline entries from "
                              << config.baseline_path << std::endl;
                }
            }
            loaded = true;
        }
    }
    return baseline;
}

// Table header printed flag
bool &TableHeaderPrinted() {
    thread_local bool printed = false;
    return printed;
}

std::ostream &GetLogStream() {
    thread_local std::ofstream log_file;
    thread_local std::mutex init_mutex;
    thread_local bool initialized = false;
    thread_local bool use_console = false;

    if (!initialized) {
        std::lock_guard<std::mutex> lock(init_mutex);
        if (!initialized) {
            const auto &config = nn::parallel::global::GlobalEnv::Instance().GetPrecisionCheckConfig();

            if (config.output_path.empty()) {
                use_console = true;
            } else {
                // Create output directory if it doesn't exist
                mkdir(config.output_path.c_str(), 0755);

                int global_rank = nn::parallel::global::thread_global_rank;
                std::string filename
                    = config.output_path + "/precision_check_rank_" + std::to_string(global_rank) + ".log";
                log_file.open(filename, std::ios::out | std::ios::trunc);
                if (!log_file.is_open()) {
                    std::cerr << "[Rank " << global_rank << "] Failed to open precision check log file: " << filename
                              << std::endl;
                    use_console = true;
                } else {
                    use_console = false;
                    std::cout << "[Rank " << global_rank << "] Precision check output: " << filename << std::endl;
                }
            }
            initialized = true;
        }
    }

    return use_console ? std::cout : log_file;
}

bool ShouldPrint() {
    const auto &config = nn::parallel::global::GlobalEnv::Instance().GetPrecisionCheckConfig();
    if (!config.output_path.empty()) {
        return true;
    }
    return nn::parallel::global::GlobalEnv::Instance().global_proc_rank() == 0;
}

std::string GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm;
    localtime_r(&time_t, &tm);

    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << (tm.tm_mon + 1) << std::setw(2) << tm.tm_mday << ' ' << std::setw(2)
        << tm.tm_hour << ':' << std::setw(2) << tm.tm_min << ':' << std::setw(2) << tm.tm_sec << '.' << std::setw(3)
        << ms.count();
    return oss.str();
}

std::string FormatShape(const std::vector<int64_t> &shape) {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << ")";
    return oss.str();
}

std::string DataTypeToString(DataType dtype) {
    switch (dtype) {
    case DataType::kFLOAT32:
        return "float32";
    case DataType::kFLOAT16:
        return "float16";
    case DataType::kBFLOAT16:
        return "bfloat16";
    case DataType::kINT32:
        return "int32";
    case DataType::kINT64:
        return "int64";
    default:
        return "unknown";
    }
}

void PrintTableHeader(std::ostream &os) {
    if (TableHeaderPrinted()) {
        return;
    }
    TableHeaderPrinted() = true;

    os << "+" << std::string(50, '-') << "+" << std::string(7, '-') << "+" << std::string(18, '-') << "+"
       << std::string(15, '-') << "+" << std::string(10, '-') << "+\n";
    os << "| " << std::left << std::setw(49) << "key"
       << "| " << std::setw(6) << "level"
       << "| " << std::setw(17) << "shape"
       << "| " << std::setw(14) << "dtype"
       << "| " << std::setw(9) << "same_hash"
       << "|\n";
    os << "+" << std::string(50, '-') << "+" << std::string(7, '-') << "+" << std::string(18, '-') << "+"
       << std::string(15, '-') << "+" << std::string(10, '-') << "+\n";
}

void PrintTableRow(std::ostream &os, const std::string &key, int level, const std::string &shape,
                   const std::string &dtype, const std::string &same_hash) {
    os << "| " << std::left << std::setw(49) << key.substr(0, 49) << "| " << std::setw(6) << level << "| "
       << std::setw(17) << shape.substr(0, 17) << "| " << std::setw(14) << dtype << "| " << std::setw(9) << same_hash
       << "|\n";
}

// Calculate diff order between two tensors (returns string like "1e-3" or "0")
std::string CalculateDiffOrder(const float *data1, const float *data2, size_t size) {
    if (!data1 || !data2 || size == 0) {
        return "N/A";
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double diff = std::abs(static_cast<double>(data1[i]) - static_cast<double>(data2[i]));
        max_diff = std::max(max_diff, diff);
    }

    if (max_diff == 0.0) {
        return "0";
    }

    int order = static_cast<int>(std::floor(std::log10(max_diff)));
    return "1e" + std::to_string(order);
}

} // namespace

void PrecisionChecker::CheckTensors(const std::string &stage, const std::string &name,
                                    const std::vector<std::shared_ptr<Tensor>> &tensors, const Config &config) {
    if (!ShouldPrint()) {
        return;
    }

    const auto &global_config = nn::parallel::global::GlobalEnv::Instance().GetPrecisionCheckConfig();
    int rank = nn::parallel::global::thread_global_rank;
    int level = global_config.level;
    auto &baseline = GetBaseline();

    for (size_t i = 0; i < tensors.size(); ++i) {
        if (!tensors[i]) {
            continue;
        }

        auto &tensor = tensors[i];

        // Copy tensor to CPU if it's on GPU
        std::shared_ptr<Tensor> cpu_tensor;
        if (tensor->GetDevice()->Type() == DeviceType::kCUDA) {
            auto cpu_device = DeviceManager::Instance()->GetDevice(DeviceType::kCPU);
            cpu_tensor = std::make_shared<Tensor>(tensor->To(cpu_device));
        } else {
            cpu_tensor = tensor;
        }

        const void *data = cpu_tensor->DataPtr();
        size_t byte_size = cpu_tensor->SizeInBytes();
        size_t num_elements = cpu_tensor->NumElements();

        // Build key
        std::string context_key = PrecisionCheckContext::Instance().GetKey();
        std::string full_key = context_key.empty() ? (stage + " " + name + " tensor[" + std::to_string(i) + "]")
                                                   : (context_key + " " + stage + " " + name);

        // Only compute MD5 if needed (for output or baseline comparison)
        bool need_md5 = global_config.output_md5 || !baseline.empty();
        std::string md5;
        if (need_md5) {
            md5 = ComputeMD5(data, byte_size);
        }

        // Check baseline
        bool has_baseline = !baseline.empty();
        bool same_hash = true;
        if (has_baseline) {
            auto it = baseline.find(full_key);
            if (it == baseline.end() && !context_key.empty()) {
                // Try without context: "stage name tensor[i]"
                std::string key_without_context = stage + " " + name + " tensor[" + std::to_string(i) + "]";
                it = baseline.find(key_without_context);
            }
            if (it != baseline.end()) {
                same_hash = (it->second == md5);
            }
        }

        auto &log_stream = GetLogStream();

        if (global_config.format == "table") {
            thread_local bool header_printed = false;
            if (!header_printed) {
                PrintTableHeader(log_stream);
                header_printed = true;
            }
            std::string same_hash_str = has_baseline ? (same_hash ? "True" : "False") : "--";
            PrintTableRow(log_stream, full_key, level, FormatShape(cpu_tensor->Dims()),
                          DataTypeToString(cpu_tensor->Dtype()), same_hash_str);

            // Save to baseline file if output_path is set and output_md5 is true
            if (!global_config.output_path.empty() && global_config.output_md5) {
                log_stream << full_key << "|" << md5 << std::endl;
            }
        } else {
            // Simple format
            const float *float_data = static_cast<const float *>(data);

            bool has_nan = false;
            bool has_inf = false;
            for (size_t j = 0; j < num_elements; ++j) {
                float val = float_data[j];
                if (std::isnan(val)) {
                    has_nan = true;
                }
                if (std::isinf(val)) {
                    has_inf = true;
                }
            }

            bool has_error = (config.check_nan && has_nan) || (config.check_inf && has_inf);

            // When output_path is set, always write to file; otherwise only write on error or if print_stats is enabled
            bool should_output = !global_config.output_path.empty() || has_error || config.print_stats;

            if (should_output) {
                std::string log_level = has_error ? "E" : "I";

                log_stream << log_level << GetTimestamp() << " [Rank " << rank << "][PrecisionCheck] " << stage << " "
                           << name << " tensor[" << i << "]: ";

                if (global_config.output_md5) {
                    log_stream << "md5=" << md5;
                    if (!same_hash) {
                        log_stream << " (MISMATCH)";
                    }
                } else {
                    log_stream << "[";
                    if (has_nan) {
                        log_stream << " NaN detected!";
                    }
                    if (has_inf) {
                        log_stream << " Inf detected!";
                    }

                    if (config.print_stats) {
                        constexpr size_t max_print = 6;
                        for (size_t j = 0; j < std::min(num_elements, max_print); ++j) {
                            if (j > 0) {
                                log_stream << ", ";
                            }
                            log_stream << float_data[j];
                        }
                        if (num_elements > max_print) {
                            log_stream << ", ...";
                        }
                    }
                    log_stream << "]";
                }
                log_stream << std::endl;
            }

            if (has_error && config.abort_on_error) {
                std::cerr << "Precision check failed, aborting!" << std::endl;
                std::abort();
            }
        }
    }
}

void PrecisionChecker::RegisterForFunction(autograd::Function *func, const std::string &name, const Config &config) {
    std::string func_name = name.empty() ? "Function" : name;

    func->RegisterForwardPreHook(
        [func_name, config](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &inputs) {
            CheckTensors("Forward Input", func_name, inputs, config);
        });

    func->RegisterForwardPostHook([func_name, config](autograd::Function *,
                                                      const std::vector<std::shared_ptr<Tensor>> &,
                                                      const std::vector<std::shared_ptr<Tensor>> &outputs) {
        CheckTensors("Forward Output", func_name, outputs, config);
    });

    func->RegisterBackwardPreHook(
        [func_name, config](autograd::Function *, const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
            CheckTensors("Backward Input", func_name, grad_outputs, config);
        });

    func->RegisterBackwardPostHook([func_name, config](autograd::Function *,
                                                       const std::vector<std::shared_ptr<Tensor>> &grad_inputs,
                                                       const std::vector<std::shared_ptr<Tensor>> &) {
        CheckTensors("Backward Output", func_name, grad_inputs, config);
    });
}

void PrecisionChecker::RegisterForModule(nn::Module *module, const std::string &name, const Config &config) {
    std::string module_name = name.empty() ? module->type() : name;

    module->RegisterForwardPostHook([module_name, config](nn::Module *, const std::vector<std::shared_ptr<Tensor>> &,
                                                          const std::vector<std::shared_ptr<Tensor>> &outputs) {
        CheckTensors("Module Forward Output", module_name, outputs, config);
    });

    module->RegisterBackwardPostHook([module_name, config](nn::Module *,
                                                           const std::vector<std::shared_ptr<Tensor>> &grad_inputs,
                                                           const std::vector<std::shared_ptr<Tensor>> &) {
        CheckTensors("Module Backward Output", module_name, grad_inputs, config);
    });
}

} // namespace infini_train::utils
