#include "infini_train/include/utils/precision_check_config.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "infini_train/include/utils/precision_checker.h"

namespace infini_train::utils {

namespace {
// Thread-local tensor counter for precision check file indexing
thread_local std::unordered_map<std::string, int> tls_g_tensor_counter;
} // namespace

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
    if (kv_map.count("path")) {
        config.output_path = kv_map["path"];
    }
    if (kv_map.count("format")) {
        config.format = kv_map["format"];
    }
    if (kv_map.count("save_tensors")) {
        config.save_tensors = (kv_map["save_tensors"] == "true" || kv_map["save_tensors"] == "1");
    }
    if (kv_map.count("md5_tolerance")) {
        config.md5_tolerance = std::stod(kv_map["md5_tolerance"]);
    }
    return config;
}

PrecisionCheckEnv &PrecisionCheckEnv::Instance() {
    static PrecisionCheckEnv instance;
    return instance;
}

void PrecisionCheckEnv::Init(const PrecisionCheckConfig &config) {
    config_ = config;
    if (config_.level != PrecisionCheckLevel::OFF) {
        // Create timestamped subdirectory: output_path/YYYYMMDD_HHMMSS/
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
        localtime_r(&time_t, &tm);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);

        timestamped_path_ = config_.output_path + "/" + buf;
        std::filesystem::create_directories(timestamped_path_);

        // Initialize PrecisionChecker (registers global module hooks)
        PrecisionChecker::Init(config_);

        // Output precision check output path
        std::cout << "[PrecisionCheck] Output: " << timestamped_path_ << std::endl;
    }
}

const PrecisionCheckConfig &PrecisionCheckEnv::GetConfig() const { return config_; }

const std::string &PrecisionCheckEnv::GetOutputPath() const { return timestamped_path_; }

int PrecisionCheckEnv::GetAndIncrementCounter(const std::string &key) { return tls_g_tensor_counter[key]++; }

void PrecisionCheckEnv::ResetCounters() { tls_g_tensor_counter.clear(); }

} // namespace infini_train::utils
