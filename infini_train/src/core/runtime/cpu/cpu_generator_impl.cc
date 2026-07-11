#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"

#include <cstring>
#include <mutex>
#include <random>
#include <sstream>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::core::cpu {

// ============================================================
// 非确定性随机数（仿 c10::detail::getNonDeterministicRandom）
// ============================================================
// Linux 下读取 /dev/urandom；其他平台 fallback 到 std::random_device
//
static uint64_t getNonDeterministicRandom() {
    std::random_device rd;
    uint64_t val = (static_cast<uint64_t>(rd()) << 32) | rd();
    return val;
}

// ============================================================
// CPUGeneratorImpl
// ============================================================

CPUGeneratorImpl::CPUGeneratorImpl(uint64_t seed)
    : GeneratorImpl(Device(Device::DeviceType::kCPU, 0))
    , engine_(seed)
    , seed_(seed) {}

void CPUGeneratorImpl::set_current_seed(uint64_t seed) {
    seed_ = seed;
    next_float_normal_sample_.reset();
    next_double_normal_sample_.reset();
    engine_ = std::mt19937(seed);
}

uint64_t CPUGeneratorImpl::current_seed() const {
    return seed_;
}

uint64_t CPUGeneratorImpl::seed() {
    uint64_t random_seed = getNonDeterministicRandom();
    set_current_seed(random_seed);
    return random_seed;
}

// ============================================================
// 状态序列化
// ============================================================
// 二进制格式（仿 PyTorch CPUGeneratorImplState）：
//   [engine_stream: N bytes]  — mt19937 的 operator<< 输出
//   [seed_: 8 bytes]          — 当前种子
//   [has_float: 1 byte]       — 是否有缓存的 float 正态样本
//   [float_val: 4 bytes]      — 缓存的 float 正态样本
//   [has_double: 1 byte]      — 是否有缓存的 double 正态样本
//   [double_val: 8 bytes]     — 缓存的 double 正态样本
//
// 注意：engine 部分使用 operator<< / operator>>，格式是实现定义的。
// Known limitation: PyTorch 通过自己实现 mt19937_engine 暴露 data()/set_data()
// 来实现固定格式序列化。我们使用 std::mt19937，标准库不提供内部状态访问接口，
// 因此无法做到跨编译器/跨版本的固定格式。此实现在同一构建下是稳定的。

void CPUGeneratorImpl::set_state(const Tensor &state) {
    const uint8_t *data = static_cast<const uint8_t *>(state.DataPtr());
    const size_t data_size = state.SizeInBytes();
    size_t offset = 0;

    // 1. 恢复引擎（变长部分直到 seed_ 字段前 ~ 22 字节的固定尾部）
    //    引擎的 operator<< 输出是变长的，我们把剩下的 data 一起给 stream
    //    但流可能会多读。改用精确长度：data_size 减去尾部固定字段长度。
    constexpr size_t kFooterSize = 8 + 1 + 4 + 1 + 8;  // seed + has_float + float + has_double + double

    std::string engine_str;
    if (data_size >= kFooterSize) {
        engine_str.assign(reinterpret_cast<const char *>(data), data_size - kFooterSize);
        offset = data_size - kFooterSize;
    } else {
        // 旧格式：没有 footer（向后兼容），整个 data 就是 engine 状态
        engine_str.assign(reinterpret_cast<const char *>(data), data_size);
        offset = data_size;
    }

    std::istringstream iss(engine_str);
    iss >> engine_;

    // 2. 恢复种子和正态缓存
    if (offset + kFooterSize <= data_size) {
        std::memcpy(&seed_, data + offset, sizeof(seed_));
        offset += sizeof(seed_);

        bool has_float = (data[offset++] != 0);
        float float_val;
        std::memcpy(&float_val, data + offset, sizeof(float_val));
        offset += sizeof(float_val);
        next_float_normal_sample_ = has_float ? std::optional<float>(float_val) : std::nullopt;

        bool has_double = (data[offset++] != 0);
        double double_val;
        std::memcpy(&double_val, data + offset, sizeof(double_val));
        next_double_normal_sample_ = has_double ? std::optional<double>(double_val) : std::nullopt;
    }
}

std::shared_ptr<Tensor> CPUGeneratorImpl::get_state() const {
    // 1. 序列化引擎
    std::ostringstream oss;
    oss << engine_;
    std::string engine_str = oss.str();

    // 2. 计算总大小
    const size_t engine_size = engine_str.size();
    constexpr size_t kFooterSize = 8 + 1 + 4 + 1 + 8;
    const size_t total_size = engine_size + kFooterSize;

    auto state_tensor = std::make_shared<Tensor>(
        std::vector<int64_t>{static_cast<int64_t>(total_size)},
        DataType::kUINT8, Device(Device::DeviceType::kCPU, 0));

    uint8_t *data = static_cast<uint8_t *>(state_tensor->DataPtr());
    size_t offset = 0;

    // 写入引擎
    std::memcpy(data + offset, engine_str.data(), engine_size);
    offset += engine_size;

    // 写入种子
    std::memcpy(data + offset, &seed_, sizeof(seed_));
    offset += sizeof(seed_);

    // 写入 float 正态缓存
    bool has_float = next_float_normal_sample_.has_value();
    data[offset++] = has_float ? 1 : 0;
    float float_val = has_float ? *next_float_normal_sample_ : 0.0f;
    std::memcpy(data + offset, &float_val, sizeof(float_val));
    offset += sizeof(float_val);

    // 写入 double 正态缓存
    bool has_double = next_double_normal_sample_.has_value();
    data[offset++] = has_double ? 1 : 0;
    double double_val = has_double ? *next_double_normal_sample_ : 0.0;
    std::memcpy(data + offset, &double_val, sizeof(double_val));

    return state_tensor;
}

// ============================================================
// 随机数生成
// ============================================================

uint32_t CPUGeneratorImpl::random() {
    return engine_();
}

uint64_t CPUGeneratorImpl::random64() {
    uint32_t hi = engine_();
    uint32_t lo = engine_();
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

// ============================================================
// Box-Muller 正态缓存
// ============================================================

std::optional<float> CPUGeneratorImpl::next_float_normal_sample() const {
    return next_float_normal_sample_;
}

std::optional<double> CPUGeneratorImpl::next_double_normal_sample() const {
    return next_double_normal_sample_;
}

void CPUGeneratorImpl::set_next_float_normal_sample(std::optional<float> randn) {
    next_float_normal_sample_ = randn;
}

void CPUGeneratorImpl::set_next_double_normal_sample(std::optional<double> randn) {
    next_double_normal_sample_ = randn;
}

// ============================================================
// clone
// ============================================================

std::shared_ptr<CPUGeneratorImpl> CPUGeneratorImpl::clone() const {
    return std::shared_ptr<CPUGeneratorImpl>(clone_impl());
}

CPUGeneratorImpl *CPUGeneratorImpl::clone_impl() const {
    auto gen = new CPUGeneratorImpl(seed_);
    gen->set_engine(engine_);
    gen->set_next_float_normal_sample(next_float_normal_sample_);
    gen->set_next_double_normal_sample(next_double_normal_sample_);
    return gen;
}

void CPUGeneratorImpl::set_engine(std::mt19937 engine) {
    engine_ = std::move(engine);
}

// ============================================================
// 类型标识
// ============================================================

Device::DeviceType CPUGeneratorImpl::device_type() {
    return Device::DeviceType::kCPU;
}

} // namespace infini_train::core::cpu

// ============================================================
// 默认 Generator 管理（文件作用域，仿 PyTorch detail 命名空间）
// ============================================================

namespace infini_train::core::cpu {

const Generator &getDefaultCPUGenerator() {
    // 使用真随机种子初始化默认 generator（仿 PyTorch）
    static auto default_gen = createCPUGenerator(getNonDeterministicRandom());
    return default_gen;
}

Generator createCPUGenerator(uint64_t seed) {
    return make_generator<CPUGeneratorImpl>(seed);
}

void manual_seed(uint64_t seed) {
    const auto &default_gen = getDefaultCPUGenerator();
    std::lock_guard<std::mutex> lock(default_gen.mutex());
    default_gen.set_current_seed(seed);
}

} // namespace infini_train::core::cpu
