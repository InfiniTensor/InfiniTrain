#pragma once

#include <sstream>
#include <string>
#include <unordered_map>

namespace infini_train {
namespace utils {

struct PrecisionCheckConfig {
    int level = 0;                  // 0=off, 1=module, 2=function
    std::string output_path = "";   // empty=console(rank0), non-empty=file(all ranks)
    bool output_md5 = false;        // output MD5 hash or tensor values
    std::string format = "simple";  // "simple" or "table"
    std::string baseline_path = ""; // baseline file path for comparison

    // Parse from "key=value,key=value" string
    static PrecisionCheckConfig Parse(const std::string &config_str) {
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
            config.level = std::stoi(kv_map["level"]);
        }
        if (kv_map.count("output_path")) {
            config.output_path = kv_map["output_path"];
        }
        if (kv_map.count("output_md5")) {
            config.output_md5 = (kv_map["output_md5"] == "true" || kv_map["output_md5"] == "1");
        }
        if (kv_map.count("format")) {
            config.format = kv_map["format"];
        }
        if (kv_map.count("baseline")) {
            config.baseline_path = kv_map["baseline"];
        }

        return config;
    }
};

} // namespace utils
} // namespace infini_train
