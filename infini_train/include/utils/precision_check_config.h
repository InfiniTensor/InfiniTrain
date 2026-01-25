#pragma once

#include <string>
#include <unordered_map>

namespace infini_train {
namespace utils {

enum class PrecisionCheckLevel { OFF = 0, MODULE = 1, FUNCTION = 2 };

struct PrecisionCheckConfig {
    PrecisionCheckLevel level = PrecisionCheckLevel::OFF;
    std::string output_path = "./precision_check"; // Output path (default)
    std::string format = "simple";                 // "simple" or "md5"
    bool save_tensors = false;                     // Whether to output .npy file
    double md5_tolerance = 0.0;                    // MD5 tolerance for quantization (e.g., 1e-3)
                                                   // 0 means no quantization (original precision)

    // Parse from "key=value,key=value" string
    static PrecisionCheckConfig Parse(const std::string &config_str);
};

class PrecisionCheckEnv {
public:
    static PrecisionCheckEnv &Instance();
    void Init(const PrecisionCheckConfig &config);
    const PrecisionCheckConfig &GetConfig() const;
    const std::string &GetOutputPath() const;

    // Tensor counter management for file overwrite across iterations (thread-local)
    static int GetAndIncrementCounter(const std::string &key);
    static void ResetCounters();

private:
    PrecisionCheckEnv() = default;
    PrecisionCheckConfig config_;
    std::string timestamped_path_; // Actual output path (with timestamp)
};

} // namespace utils
} // namespace infini_train
