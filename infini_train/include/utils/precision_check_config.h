#pragma once

#include <string>

namespace infini_train {
namespace utils {

enum class PrecisionCheckLevel { OFF = 0, MODULE = 1, FUNCTION = 2 };

struct PrecisionCheckConfig {
    PrecisionCheckLevel level = PrecisionCheckLevel::OFF;
    std::string output_path = "";   // empty=console(rank0), non-empty=file(all ranks)
    bool output_md5 = false;        // output MD5 hash or tensor values
    std::string format = "simple";  // "simple" or "table"
    std::string baseline_path = ""; // baseline file path for comparison

    // Parse from "key=value,key=value" string
    static PrecisionCheckConfig Parse(const std::string &config_str);
};

class PrecisionCheckEnv {
public:
    static PrecisionCheckEnv &Instance();
    void Init(const PrecisionCheckConfig &config);
    const PrecisionCheckConfig &GetConfig() const;

private:
    PrecisionCheckEnv() = default;
    PrecisionCheckConfig config_;
};

} // namespace utils
} // namespace infini_train
