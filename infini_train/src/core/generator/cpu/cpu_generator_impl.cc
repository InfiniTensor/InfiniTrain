#include "infini_train/include/core/generator/cpu_generator_impl.h"

#include <cstdint>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/generator/generator_impl.h"
#include "infini_train/include/device.h"

namespace infini_train::core {

CPUGeneratorImpl::CPUGeneratorImpl(int8_t device_index) : GeneratorImpl(Device(Device::DeviceType::kCPU, 0)) {
    CHECK_EQ(device_index, 0) << "CPU Generator only supports device index 0";
}

void CPUGeneratorImpl::SetCurrentSeed(uint64_t seed) {
    initial_seed_ = seed;
    seed_ = seed;
    engine_.seed(seed);
}

void CPUGeneratorImpl::Reseed(uint64_t seed) {
    seed_ = seed;
    engine_.seed(seed);
}

std::vector<uint8_t> CPUGeneratorImpl::GetState() const {
    LOG(FATAL) << "CPUGeneratorImpl::GetState() not implemented yet (Task 6)";
    return {};
}

void CPUGeneratorImpl::SetState(const std::vector<uint8_t> & /*state*/) {
    LOG(FATAL) << "CPUGeneratorImpl::SetState() not implemented yet (Task 6)";
}

namespace {

std::shared_ptr<GeneratorImpl> CpuFactory(int8_t device_index) { return std::make_shared<CPUGeneratorImpl>(device_index); }

} // namespace

} // namespace infini_train::core

INFINI_TRAIN_REGISTER_GENERATOR_IMPL(kCPU, ::infini_train::core::CpuFactory)
