#include "infini_train/include/utils/precision_check_config.h"

#include <sstream>
#include <unordered_map>

namespace infini_train::utils {

PrecisionCheckConfig PrecisionCheckConfig::Parse(const std::string &config_str) {
    PrecisionCheckConfig config;
    if (config_str.empty()) {
        return config;
    }

    std::unordered_map<std::string, std::string> kv_map;
    std::istringstream ss(config_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        auto pos = item.find('=');
        if (pos != std::string::npos) {
            kv_map[item.substr(0, pos)] = item.substr(pos + 1);
        }
    }

    if (kv_map.count("level")) {
        int level_int = std::stoi(kv_map["level"]);
        config.level = static_cast<PrecisionCheckLevel>(level_int);
    }
    if (kv_map.count("output_path")) {
        config.output_path = kv_map["output_path"];
    }
    if (kv_map.count("output_md5")) {
        config.output_md5 = (kv_map["output_md5"] == "true" || kv_map["output_md5"] == "1");
    }
    if (kv_map.count("baseline")) {
        config.baseline_path = kv_map["baseline"];
    }
    if (kv_map.count("format")) {
        config.format = kv_map["format"];
    } else if (!config.baseline_path.empty()) {
        // Default to table format when baseline is specified
        config.format = "table";
    }
    return config;
}

PrecisionCheckEnv &PrecisionCheckEnv::Instance() {
    static PrecisionCheckEnv instance;
    return instance;
}

void PrecisionCheckEnv::Init(const PrecisionCheckConfig &config) { config_ = config; }

} // namespace infini_train::utils
