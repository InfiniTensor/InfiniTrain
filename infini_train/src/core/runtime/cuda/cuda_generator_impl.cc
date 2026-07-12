#include "infini_train/src/core/runtime/cuda/cuda_generator_impl.h"

#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/tensor.h"

namespace infini_train::core::cuda {
namespace {

constexpr size_t kStateSize = sizeof(uint64_t) * 2;

std::once_flag default_generators_init_flag;
std::vector<Generator> default_generators;
std::deque<std::once_flag> default_generator_init_flags;

uint64_t get_non_deterministic_random() {
    std::random_device random_device;
    return (static_cast<uint64_t>(random_device()) << 32) | random_device();
}

void init_default_generators() {
    std::call_once(default_generators_init_flag, [] {
        int device_count = 0;
        const cudaError_t status = cudaGetDeviceCount(&device_count);
        if (status == cudaErrorNoDevice) {
            cudaGetLastError();
            return;
        }
        CHECK_EQ(status, cudaSuccess) << "cudaGetDeviceCount failed: " << cudaGetErrorString(status);
        default_generators.resize(device_count);
        default_generator_init_flags.resize(device_count);
    });
}

int resolve_device_index(int8_t device_index) {
    init_default_generators();
    int index = device_index;
    if (index == -1) {
        CUDA_CHECK(cudaGetDevice(&index));
    }
    int device_count = 0;
    device_count = static_cast<int>(default_generators.size());
    CHECK(index >= 0 && index < device_count) << "Invalid CUDA device index " << index;
    return index;
}

} // namespace

CUDAGeneratorImpl::CUDAGeneratorImpl(int8_t device_index, uint64_t seed)
    : GeneratorImpl(Device(Device::DeviceType::kCUDA, device_index))
    , seed_(seed) {}

void CUDAGeneratorImpl::set_current_seed(uint64_t seed) {
    seed_ = seed;
    next_philox_subsequence_ = 0;
}

uint64_t CUDAGeneratorImpl::current_seed() const {
    return seed_;
}

uint64_t CUDAGeneratorImpl::seed() {
    const uint64_t random_seed = get_non_deterministic_random();
    set_current_seed(random_seed);
    return random_seed;
}

void CUDAGeneratorImpl::set_state(const Tensor &state) {
    ::infini_train::detail::check_rng_state(state);
    CHECK_EQ(state.SizeInBytes(), kStateSize);

    const auto *data = static_cast<const uint8_t *>(state.DataPtr());
    std::memcpy(&seed_, data, sizeof(seed_));
    std::memcpy(&next_philox_subsequence_, data + sizeof(seed_), sizeof(next_philox_subsequence_));
}

std::shared_ptr<Tensor> CUDAGeneratorImpl::get_state() const {
    auto state = std::make_shared<Tensor>(
        std::vector<int64_t>{static_cast<int64_t>(kStateSize)},
        DataType::kUINT8,
        Device(Device::DeviceType::kCPU, 0));

    auto *data = static_cast<uint8_t *>(state->DataPtr());
    std::memcpy(data, &seed_, sizeof(seed_));
    std::memcpy(data + sizeof(seed_), &next_philox_subsequence_, sizeof(next_philox_subsequence_));
    return state;
}

Device::DeviceType CUDAGeneratorImpl::device_type() {
    return Device::DeviceType::kCUDA;
}

uint64_t CUDAGeneratorImpl::philox_subsequence(uint64_t increment) {
    const uint64_t subsequence = next_philox_subsequence_;
    next_philox_subsequence_ += increment;
    return subsequence;
}

CUDAGeneratorImpl *CUDAGeneratorImpl::clone_impl() const {
    auto *generator = new CUDAGeneratorImpl(device().index(), seed_);
    generator->next_philox_subsequence_ = next_philox_subsequence_;
    return generator;
}

const Generator &getDefaultCUDAGenerator(int8_t device_index) {
    const int index = resolve_device_index(device_index);
    std::call_once(default_generator_init_flags[index], [index] {
        default_generators[index] = createCUDAGenerator(static_cast<int8_t>(index), get_non_deterministic_random());
    });
    return default_generators[index];
}

Generator createCUDAGenerator(int8_t device_index, uint64_t seed) {
    const int index = resolve_device_index(device_index);
    return make_generator<CUDAGeneratorImpl>(static_cast<int8_t>(index), seed);
}

void manual_seed_all(uint64_t seed) {
    init_default_generators();
    for (size_t index = 0; index < default_generators.size(); ++index) {
        const auto &generator = getDefaultCUDAGenerator(static_cast<int8_t>(index));
        std::lock_guard<std::mutex> lock(generator.mutex());
        generator.set_current_seed(seed);
    }
}

} // namespace infini_train::core::cuda
