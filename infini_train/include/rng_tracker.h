#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"

namespace infini_train {

class Tensor;

// Owns independent, named RNG streams for each device. Child threads do not inherit scopes.
class RngTracker {
public:
    explicit RngTracker(uint64_t seed = Generator::kDefaultSeed);
    ~RngTracker();

    RngTracker(const RngTracker &) = delete;
    RngTracker &operator=(const RngTracker &) = delete;
    RngTracker(RngTracker &&) = delete;
    RngTracker &operator=(RngTracker &&) = delete;

    void Add(const std::string &name, uint64_t seed);
    bool Contains(const std::string &name) const;

    void ManualSeed(uint64_t seed);
    void ManualSeed(const std::string &name, uint64_t seed);

    // Returns an independent CPU UINT8 tensor containing all streams.
    std::shared_ptr<Tensor> GetState() const;
    // Restoring invalidates references to this tracker's default generators.
    void SetState(const Tensor &state);

private:
    friend class RngAccessor;
    friend class RngScope;

    using DeviceKey = std::pair<Device::DeviceType, int8_t>;

    struct RngStream {
        uint64_t seed;
        std::map<DeviceKey, Generator> generators;
    };

    using Streams = std::map<std::string, RngStream>;

    mutable std::recursive_mutex mutex_;
    Streams streams_;
};

// Selects a stream on the current thread. External trackers must outlive their scopes.
// Concurrent scopes on different threads cannot share a tracker; nesting on one thread is supported.
// Owning scopes start fresh; binding an existing tracker retains its progress.
class RngScope {
public:
    RngScope();
    explicit RngScope(uint64_t seed);
    explicit RngScope(RngTracker &tracker);
    RngScope(RngTracker &tracker, const std::string &name);
    ~RngScope();

    RngScope(const RngScope &) = delete;
    RngScope &operator=(const RngScope &) = delete;
    RngScope(RngScope &&) = delete;
    RngScope &operator=(RngScope &&) = delete;

private:
    friend class RngAccessor;
    friend class RngTracker;

    std::unique_ptr<RngTracker> owned_tracker_;
    RngTracker *tracker_;
    std::string name_;
    RngScope *previous_;
    std::thread::id thread_;
    std::unique_lock<std::recursive_mutex> lock_;
};

} // namespace infini_train
