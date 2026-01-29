#include "infini_train/include/utils/precision_checker.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/global.h"
#include "infini_train/include/nn/parallel/tensor_parallel.h"
#include "infini_train/include/tensor.h"
#include "infini_train/include/utils/global_module_hook_registry.h"
#include "infini_train/include/utils/precision_check_config.h"
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

std::ostream &GetLogStream() {
    thread_local std::ofstream tls_log_file;
    thread_local std::mutex tls_init_mutex;
    thread_local bool tls_initialized = false;

    if (!tls_initialized) {
        std::lock_guard<std::mutex> lock(tls_init_mutex);
        if (!tls_initialized) {
            const auto &output_path = PrecisionCheckEnv::Instance().GetOutputPath();
            int global_rank = nn::parallel::global::thread_global_rank;
            std::string filename = output_path + "/precision_check_rank_" + std::to_string(global_rank) + ".log";
            tls_log_file.open(filename, std::ios::out | std::ios::trunc);
            if (!tls_log_file.is_open()) {
                std::cerr << "[Rank " << global_rank << "] Failed to open precision check log file: " << filename
                          << std::endl;
            } else {
                std::cout << "[Rank " << global_rank << "] Precision check output: " << filename << std::endl;
            }
            tls_initialized = true;
        }
    }

    return tls_log_file.is_open() ? tls_log_file : std::cout;
}

std::string FormatShape(const std::vector<int64_t> &shape) {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            oss << ",";
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

struct TensorStats {
    float min_val = 0;
    float max_val = 0;
    float mean_val = 0;
    int nan_count = 0;
    int inf_count = 0;
};

TensorStats ComputeStats(const float *data, size_t num_elements) {
    TensorStats stats;
    if (num_elements == 0) {
        return stats;
    }

    stats.min_val = std::numeric_limits<float>::max();
    stats.max_val = std::numeric_limits<float>::lowest();
    double sum = 0;

    for (size_t i = 0; i < num_elements; ++i) {
        float val = data[i];
        if (std::isnan(val)) {
            stats.nan_count++;
            continue;
        }
        if (std::isinf(val)) {
            stats.inf_count++;
            continue;
        }
        stats.min_val = std::min(stats.min_val, val);
        stats.max_val = std::max(stats.max_val, val);
        sum += val;
    }

    size_t valid_count = num_elements - stats.nan_count - stats.inf_count;
    stats.mean_val = valid_count > 0 ? static_cast<float>(sum / valid_count) : 0;

    return stats;
}

// Quantize float data to specified tolerance for MD5 calculation
// e.g., tolerance=1e-3: 4.0003 and 4.0004 both become 4.000
std::vector<float> QuantizeData(const float *data, size_t num_elements, double tolerance) {
    std::vector<float> quantized(num_elements);
    double inv_tolerance = 1.0 / tolerance;
    for (size_t i = 0; i < num_elements; ++i) {
        quantized[i] = static_cast<float>(std::round(data[i] * inv_tolerance) * tolerance);
    }
    return quantized;
}

void SaveNpy(const std::shared_ptr<Tensor> &tensor, const std::string &name, int idx, const std::string &stage,
             int rank) {
    const auto &output_path = PrecisionCheckEnv::Instance().GetOutputPath();
    std::string dir = output_path + "/rank_" + std::to_string(rank);
    std::filesystem::create_directories(dir);
    std::string filename = dir + "/" + name + "_" + std::to_string(idx) + "_" + stage + ".npy";

    if (tensor->Dtype() == DataType::kFLOAT32) {
        tensor->SaveAsNpy(filename);
    } else {
        auto float_tensor = tensor->To(DataType::kFLOAT32);
        float_tensor.SaveAsNpy(filename);
    }
}

} // namespace

void PrecisionChecker::CheckTensors(const std::string &stage, const std::string &name,
                                    const std::vector<std::shared_ptr<Tensor>> &tensors, const Config &config) {
    const auto &global_config = PrecisionCheckEnv::Instance().GetConfig();
    if (global_config.level == PrecisionCheckLevel::OFF) {
        return;
    }

    const int rank = nn::parallel::global::thread_global_rank;

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

        const float *float_data = static_cast<const float *>(cpu_tensor->DataPtr());
        const size_t byte_size = cpu_tensor->SizeInBytes();
        const size_t num_elements = cpu_tensor->NumElements();

        // Build context key
        const std::string context_key = PrecisionCheckContext::Instance().GetKey();
        const std::string stage_short = (stage.find("Forward") != std::string::npos) ? "forward" : "backward";

        // Get tensor index for this (name, stage) combination
        std::string counter_key = name + "_" + stage_short;
        int idx = PrecisionCheckEnv::GetAndIncrementCounter(counter_key);

        // Save NPY if enabled
        if (global_config.save_tensors) {
            SaveNpy(cpu_tensor, name, idx, stage_short, rank);
        }

        // Output to log
        auto &log_stream = GetLogStream();

        if (global_config.format == "md5") {
            // MD5 format
            std::string md5;
            if (global_config.md5_tolerance > 0.0) {
                // Quantize data before computing MD5
                // Convert to float32 if needed for quantization
                std::shared_ptr<Tensor> float32_tensor = cpu_tensor;
                if (cpu_tensor->Dtype() != DataType::kFLOAT32) {
                    float32_tensor = std::make_shared<Tensor>(cpu_tensor->To(DataType::kFLOAT32));
                }
                const float *data_ptr = static_cast<const float *>(float32_tensor->DataPtr());
                size_t num_elems = float32_tensor->NumElements();
                auto quantized = QuantizeData(data_ptr, num_elems, global_config.md5_tolerance);
                md5 = ComputeMD5(quantized.data(), quantized.size() * sizeof(float));
            } else {
                // Original precision MD5
                md5 = ComputeMD5(cpu_tensor->DataPtr(), byte_size);
            }
            log_stream << context_key << " " << name << "_" << idx << "_" << stage << " tensor[" << i << "]: "
                       << "dtype=" << DataTypeToString(cpu_tensor->Dtype()) << " "
                       << "shape=" << FormatShape(cpu_tensor->Dims()) << " "
                       << "md5=" << md5 << std::endl;
        } else {
            // Simple format (default)
            TensorStats stats = ComputeStats(float_data, num_elements);

            const bool has_error
                = (config.check_nan && stats.nan_count > 0) || (config.check_inf && stats.inf_count > 0);
            const std::string error_marker = has_error ? " <- ERROR" : "";

            log_stream << context_key << " " << name << "_" << idx << "_" << stage << " tensor[" << i << "]: "
                       << "dtype=" << DataTypeToString(cpu_tensor->Dtype()) << " "
                       << "shape=" << FormatShape(cpu_tensor->Dims()) << " "
                       << "min=" << stats.min_val << " "
                       << "max=" << stats.max_val << " "
                       << "mean=" << stats.mean_val << " [";

            // Print first 6 values
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
            log_stream << "] [NaN:" << stats.nan_count << " Inf:" << stats.inf_count << "]" << error_marker
                       << std::endl;

            if (has_error && config.abort_on_error) {
                std::cerr << "Precision check failed, aborting!" << std::endl;
                std::abort();
            }
        }
    }
}

void PrecisionChecker::Init(const PrecisionCheckConfig &global_config, const Config &config) {
    if (global_config.level == PrecisionCheckLevel::OFF) {
        return;
    }

    // Register auto-hook for all modules via GlobalModuleHookRegistry
    GlobalModuleHookRegistry::Instance().RegisterHook([config](nn::Module *module) {
        RegisterForModule(module, module->type(), config);
    });
}

void PrecisionChecker::RegisterForFunction(autograd::Function *func, const std::string &name, const Config &config) {
    const std::string func_name = name.empty() ? "Function" : name;

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
    const std::string module_name = name.empty() ? module->type() : name;

    module->RegisterForwardPostHook([module_name, config](nn::Module *, const std::vector<std::shared_ptr<Tensor>> &,
                                                          const std::vector<std::shared_ptr<Tensor>> &outputs) {
        CheckTensors("Forward Output", module_name, outputs, config);
    });

    module->RegisterBackwardPostHook([module_name, config](nn::Module *,
                                                           const std::vector<std::shared_ptr<Tensor>> &grad_inputs,
                                                           const std::vector<std::shared_ptr<Tensor>> &) {
        CheckTensors("Backward Output", module_name, grad_inputs, config);
    });
}

void PrecisionChecker::ResetCounters() { PrecisionCheckEnv::ResetCounters(); }

} // namespace infini_train::utils
