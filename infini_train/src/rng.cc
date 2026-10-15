#include "infini_train/include/rng_tracker.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/core/runtime/generator_backend.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/generator_impl.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
namespace {

constexpr char kDefaultStream[] = "default";
thread_local RngScope *current_scope = nullptr;

std::unique_lock<std::recursive_mutex> LockTracker(std::recursive_mutex &mutex) {
    std::unique_lock<std::recursive_mutex> lock(mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        throw std::logic_error("RNG tracker is in use on another thread");
    }
    return lock;
}

Device CanonicalDevice(const Device &device) {
    if (device.index() == -1) {
        return core::GetDeviceGuardImpl(device.type())->GetDevice();
    }
    return device;
}

} // namespace

class RngAccessor {
public:
    static RngTracker *CurrentTracker() { return current_scope ? current_scope->tracker_ : nullptr; }

    static const Generator *GetScopedGenerator(const Device &requested_device) {
        if (!current_scope) {
            return nullptr;
        }
        auto &scope = *current_scope;
        auto &stream = scope.tracker_->streams_.at(scope.name_);
        const Device device = CanonicalDevice(requested_device);
        const RngTracker::DeviceKey key{device.type(), device.index()};
        auto found = stream.generators.find(key);
        if (found == stream.generators.end()) {
            found = stream.generators.emplace(key, CreateGenerator(device, stream.seed)).first;
        }
        return &found->second;
    }

    static bool ManualSeedCurrentStream(uint64_t seed) {
        if (!current_scope) {
            return false;
        }
        auto &scope = *current_scope;
        scope.tracker_->ManualSeed(scope.name_, seed);
        return true;
    }
};

RngTracker::RngTracker(uint64_t seed) { streams_.emplace(kDefaultStream, RngStream{seed, {}}); }

RngTracker::~RngTracker() = default;

void RngTracker::Add(const std::string &name, uint64_t seed) {
    auto lock = LockTracker(mutex_);
    if (name.empty() || streams_.contains(name)) {
        throw std::invalid_argument("RNG stream name must be nonempty and unique");
    }
    streams_.emplace(name, RngStream{seed, {}});
}

bool RngTracker::Contains(const std::string &name) const {
    auto lock = LockTracker(mutex_);
    return streams_.contains(name);
}

void RngTracker::ManualSeed(uint64_t seed) { ManualSeed(kDefaultStream, seed); }

void RngTracker::ManualSeed(const std::string &name, uint64_t seed) {
    auto lock = LockTracker(mutex_);
    auto found = streams_.find(name);
    if (found == streams_.end()) {
        throw std::invalid_argument("Unknown RNG stream: " + name);
    }
    auto &stream = found->second;
    for (const auto &[device, generator] : stream.generators) {
        std::lock_guard<std::mutex> generator_lock(GeneratorAccessor::Mutex(generator));
        generator.set_current_seed(seed);
    }
    stream.seed = seed;
}

std::shared_ptr<Tensor> RngTracker::GetState() const {
    auto lock = LockTracker(mutex_);
    constexpr char kStateMagic[] = "ITRNGTRK";
    constexpr uint64_t kStateVersion = 1;
    std::vector<uint8_t> bytes(kStateMagic, kStateMagic + sizeof(kStateMagic) - 1);
    auto append_uint64 = [&bytes](uint64_t value) {
        for (int shift = 0; shift < 64; shift += 8) { bytes.push_back(static_cast<uint8_t>(value >> shift)); }
    };

    // Metadata uses little-endian integers; backend state bytes remain opaque.
    append_uint64(kStateVersion);
    append_uint64(streams_.size());
    for (const auto &[name, stream] : streams_) {
        append_uint64(name.size());
        bytes.insert(bytes.end(), name.begin(), name.end());
        append_uint64(stream.seed);
        append_uint64(stream.generators.size());
        for (const auto &[device, generator] : stream.generators) {
            std::lock_guard<std::mutex> generator_lock(GeneratorAccessor::Mutex(generator));
            const auto state = generator.get_state();
            append_uint64(static_cast<uint64_t>(device.first));
            append_uint64(static_cast<uint64_t>(device.second));
            append_uint64(state->SizeInBytes());
            const auto *data = static_cast<const uint8_t *>(state->DataPtr());
            bytes.insert(bytes.end(), data, data + state->SizeInBytes());
        }
    }
    if (bytes.size() > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        throw std::length_error("RNG tracker state is too large");
    }
    auto state = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(bytes.size())}, DataType::kUINT8,
                                          Device());
    std::memcpy(state->DataPtr(), bytes.data(), bytes.size());
    return state;
}

void RngTracker::SetState(const Tensor &state) {
    auto lock = LockTracker(mutex_);
    constexpr char kStateMagic[] = "ITRNGTRK";
    constexpr uint64_t kStateVersion = 1;
    if (!state.defined() || !state.GetDevice().IsCPU() || state.Dtype() != DataType::kUINT8 || state.Dims().size() != 1
        || state.SizeInBytes() < sizeof(kStateMagic) - 1) {
        throw std::invalid_argument("RNG tracker state must be a nonempty one-dimensional CPU UINT8 tensor");
    }
    const auto *data = static_cast<const uint8_t *>(state.DataPtr());
    const size_t data_size = state.SizeInBytes();
    if (std::memcmp(data, kStateMagic, sizeof(kStateMagic) - 1) != 0) {
        throw std::invalid_argument("Invalid RNG tracker state magic");
    }
    size_t offset = sizeof(kStateMagic) - 1;
    auto read_uint64 = [&] {
        if (data_size - offset < sizeof(uint64_t)) {
            throw std::invalid_argument("Truncated RNG tracker state");
        }
        uint64_t value = 0;
        for (int shift = 0; shift < 64; shift += 8) { value |= static_cast<uint64_t>(data[offset++]) << shift; }
        return value;
    };
    if (read_uint64() != kStateVersion) {
        throw std::invalid_argument("Unsupported RNG tracker state version");
    }
    const uint64_t stream_count = read_uint64();
    if (stream_count == 0 || stream_count > (data_size - offset) / (3 * sizeof(uint64_t) + 1)) {
        throw std::invalid_argument("Invalid RNG tracker stream count");
    }

    struct StreamState {
        uint64_t seed;
        std::map<DeviceKey, std::pair<size_t, size_t>> generator_ranges;
    };
    std::map<std::string, StreamState> states;
    for (uint64_t i = 0; i < stream_count; ++i) {
        const uint64_t name_size = read_uint64();
        if (name_size == 0 || name_size > data_size - offset) {
            throw std::invalid_argument("Invalid RNG stream name length");
        }
        std::string name(reinterpret_cast<const char *>(data + offset), static_cast<size_t>(name_size));
        offset += static_cast<size_t>(name_size);
        StreamState stream_state{read_uint64(), {}};
        auto [found, inserted] = states.emplace(std::move(name), std::move(stream_state));
        if (!inserted) {
            throw std::invalid_argument("Duplicate RNG stream name");
        }
        const uint64_t device_count = read_uint64();
        if (device_count > (data_size - offset) / (3 * sizeof(uint64_t) + 1)) {
            throw std::invalid_argument("Invalid RNG stream device count");
        }
        for (uint64_t j = 0; j < device_count; ++j) {
            const uint64_t type = read_uint64();
            const uint64_t index = read_uint64();
            const uint64_t byte_count = read_uint64();
            if (type >= static_cast<uint64_t>(Device::DeviceType::kCount)
                || index > static_cast<uint64_t>(std::numeric_limits<int8_t>::max())) {
                throw std::invalid_argument("Invalid RNG state device");
            }
            const auto device_type = static_cast<Device::DeviceType>(type);
            GeneratorBackendRegistry::Instance().Get(device_type);
            if (index >= static_cast<uint64_t>(core::GetDeviceGuardImpl(device_type)->DeviceCount())) {
                throw std::invalid_argument("RNG tracker state has an invalid device index");
            }
            if (byte_count == 0 || byte_count > data_size - offset
                || byte_count > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                throw std::invalid_argument("Invalid RNG generator state length");
            }
            const DeviceKey key{device_type, static_cast<int8_t>(index)};
            if (!found->second.generator_ranges.emplace(key, std::pair{offset, static_cast<size_t>(byte_count)})
                     .second) {
                throw std::invalid_argument("Duplicate RNG state device");
            }
            offset += static_cast<size_t>(byte_count);
        }
    }
    if (offset != data_size || !states.contains(kDefaultStream)) {
        throw std::invalid_argument("RNG tracker state has trailing bytes or no default stream");
    }
    for (auto *scope = current_scope; scope; scope = scope->previous_) {
        if (scope->tracker_ == this && !states.contains(scope->name_)) {
            throw std::invalid_argument("Cannot restore a state that removes an active RNG stream");
        }
    }

    Streams restored_streams;
    for (const auto &[name, stream_state] : states) {
        RngStream stream{stream_state.seed, {}};
        for (const auto &[device, range] : stream_state.generator_ranges) {
            auto generator = CreateGenerator(Device(device.first, device.second), stream_state.seed);
            Tensor generator_state(std::vector<int64_t>{static_cast<int64_t>(range.second)}, DataType::kUINT8,
                                   Device());
            std::memcpy(generator_state.DataPtr(), data + range.first, range.second);
            generator.set_state(generator_state);
            stream.generators.emplace(device, std::move(generator));
        }
        restored_streams.emplace(name, std::move(stream));
    }
    streams_.swap(restored_streams);
}

RngScope::RngScope() : RngScope(Generator::kDefaultSeed) {}

RngScope::RngScope(uint64_t seed)
    : owned_tracker_(std::make_unique<RngTracker>(seed)), tracker_(owned_tracker_.get()), name_(kDefaultStream),
      previous_(current_scope), thread_(std::this_thread::get_id()), lock_(LockTracker(tracker_->mutex_)) {
    current_scope = this;
}

RngScope::RngScope(RngTracker &tracker) : RngScope(tracker, kDefaultStream) {}

RngScope::RngScope(RngTracker &tracker, const std::string &name)
    : tracker_(&tracker), name_(name), previous_(current_scope), thread_(std::this_thread::get_id()),
      lock_(LockTracker(tracker_->mutex_)) {
    if (!tracker_->streams_.contains(name_)) {
        throw std::invalid_argument("Unknown RNG stream: " + name_);
    }
    current_scope = this;
}

RngScope::~RngScope() {
    CHECK(thread_ == std::this_thread::get_id()) << "RNG scopes must be destroyed on their creating thread";
    CHECK(current_scope == this) << "RNG scopes must be destroyed in reverse construction order";
    current_scope = previous_;
}

namespace detail {

const Generator *TryGetScopedGenerator(const Device &device) { return RngAccessor::GetScopedGenerator(device); }

bool TryManualSeedScopedRng(uint64_t seed) { return RngAccessor::ManualSeedCurrentStream(seed); }

RngTracker *GetCurrentRngTracker() { return RngAccessor::CurrentTracker(); }

} // namespace detail
} // namespace infini_train
