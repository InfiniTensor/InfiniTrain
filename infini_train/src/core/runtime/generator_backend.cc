#include "infini_train/include/core/runtime/generator_backend.h"

#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "glog/logging.h"

namespace infini_train {

GeneratorBackendRegistry &GeneratorBackendRegistry::Instance() {
    static GeneratorBackendRegistry instance;
    return instance;
}

void GeneratorBackendRegistry::Register(Device::DeviceType type, std::unique_ptr<GeneratorBackend> backend) {
    CHECK(backend != nullptr) << "Registering a null GeneratorBackend";
    // DeviceType has no stream insertion operator.
    CHECK_EQ(static_cast<int>(type), static_cast<int>(backend->Type()))
        << "GeneratorBackend type mismatch: registered as " << static_cast<int>(type) << " but Type() is "
        << static_cast<int>(backend->Type());
    CHECK(!backends_.contains(type)) << "GeneratorBackend for device type " << static_cast<int>(type)
                                     << " is already registered";

    backends_.emplace(type, std::move(backend));
}

GeneratorBackend &GeneratorBackendRegistry::Get(Device::DeviceType type) const {
    auto it = backends_.find(type);
    if (it == backends_.end()) {
        throw std::invalid_argument("No GeneratorBackend registered for device type "
                                    + std::to_string(static_cast<int>(type)));
    }
    return *it->second;
}

void GeneratorBackendRegistry::ManualSeedAll(uint64_t seed) const {
    for (const auto &entry : backends_) { entry.second->ManualSeedAll(seed); }
}

} // namespace infini_train
