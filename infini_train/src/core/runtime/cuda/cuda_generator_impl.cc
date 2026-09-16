#include "infini_train/src/core/runtime/cuda/cuda_generator_impl.h"

#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "glog/logging.h"

#include "infini_train/include/common/cuda/common_cuda.h"
#include "infini_train/include/core/runtime/generator_backend.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/generator_impl.h"
#include "infini_train/include/tensor.h"

namespace infini_train::core::cuda {
namespace {

// Backend tag used to reject states from other generator implementations.
constexpr char kCUDAStateMagic[] = "ITRNGCUD";
constexpr size_t kStateMagicSize = sizeof(kCUDAStateMagic) - 1;

// [magic: 8B] + [seed_: 8B] + [next_philox_subsequence_: 8B]
constexpr size_t kStateSize = kStateMagicSize + sizeof(uint64_t) * 2;

std::once_flag default_generators_init_flag;
std::vector<std::optional<Generator>> default_generators;
std::deque<std::once_flag> default_generator_init_flags;

uint64_t GenerateNonDeterministicSeed() {
    std::random_device random_device;
    return (static_cast<uint64_t>(random_device()) << 32) | random_device();
}

void InitDefaultGenerators() {
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

int ResolveDeviceIndex(int8_t device_index) {
    InitDefaultGenerators();
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
    : GeneratorImpl(Device(Device::DeviceType::kCUDA, device_index)), seed_(seed) {}

void CUDAGeneratorImpl::set_current_seed(uint64_t seed) {
    seed_ = seed;
    next_philox_subsequence_ = 0;
}

uint64_t CUDAGeneratorImpl::current_seed() const { return seed_; }

uint64_t CUDAGeneratorImpl::Seed() {
    const uint64_t random_seed = GenerateNonDeterministicSeed();
    set_current_seed(random_seed);
    return random_seed;
}

void CUDAGeneratorImpl::set_state(const Tensor &state) {
    ::infini_train::detail::CheckRngState(state);
    CHECK_EQ(state.SizeInBytes(), kStateSize);

    const auto *data = static_cast<const uint8_t *>(state.DataPtr());
    CHECK_EQ(std::memcmp(data, kCUDAStateMagic, kStateMagicSize), 0)
        << "Invalid RNG state: not a CUDA generator state (backend magic mismatch)";
    std::memcpy(&seed_, data + kStateMagicSize, sizeof(seed_));
    std::memcpy(&next_philox_subsequence_, data + kStateMagicSize + sizeof(seed_), sizeof(next_philox_subsequence_));
}

std::shared_ptr<Tensor> CUDAGeneratorImpl::get_state() const {
    auto state = std::make_shared<Tensor>(std::vector<int64_t>{static_cast<int64_t>(kStateSize)}, DataType::kUINT8,
                                          Device(Device::DeviceType::kCPU, 0));

    auto *data = static_cast<uint8_t *>(state->DataPtr());
    std::memcpy(data, kCUDAStateMagic, kStateMagicSize);
    std::memcpy(data + kStateMagicSize, &seed_, sizeof(seed_));
    std::memcpy(data + kStateMagicSize + sizeof(seed_), &next_philox_subsequence_, sizeof(next_philox_subsequence_));
    return state;
}

uint64_t CUDAGeneratorImpl::ReservePhiloxSubsequence(uint64_t increment) {
    CHECK_LE(increment, std::numeric_limits<uint64_t>::max() - next_philox_subsequence_)
        << "Philox subsequence counter overflow";
    const uint64_t subsequence = next_philox_subsequence_;
    next_philox_subsequence_ += increment;
    return subsequence;
}

CUDAGeneratorImpl *CUDAGeneratorImpl::CloneImpl() const {
    auto *clone = new CUDAGeneratorImpl(device().index(), seed_);
    clone->next_philox_subsequence_ = next_philox_subsequence_;
    return clone;
}

const Generator &GetDefaultCudaGenerator(int8_t device_index) {
    const int index = ResolveDeviceIndex(device_index);
    std::call_once(default_generator_init_flags[index], [index] {
        default_generators[index].emplace(
            CreateCudaGenerator(static_cast<int8_t>(index), GenerateNonDeterministicSeed()));
    });
    return *default_generators[index];
}

Generator CreateCudaGenerator(int8_t device_index, uint64_t seed) {
    const int index = ResolveDeviceIndex(device_index);
    return MakeGenerator<CUDAGeneratorImpl>(static_cast<int8_t>(index), seed);
}

void ManualSeedAll(uint64_t seed) {
    InitDefaultGenerators();
    for (size_t index = 0; index < default_generators.size(); ++index) {
        bool initialized_here = false;
        std::call_once(default_generator_init_flags[index], [index, seed, &initialized_here] {
            default_generators[index].emplace(CreateCudaGenerator(static_cast<int8_t>(index), seed));
            initialized_here = true;
        });
        if (initialized_here) {
            continue;
        }
        const Generator &generator = *default_generators[index];
        std::lock_guard<std::mutex> lock(GeneratorAccessor::Mutex(generator));
        generator.set_current_seed(seed);
    }
}

} // namespace infini_train::core::cuda

namespace infini_train::core::cuda {
namespace {

class CUDAGeneratorBackend final : public GeneratorBackend {
public:
    Device::DeviceType Type() const override { return Device::DeviceType::kCUDA; }

    Generator Create(const Device &device, uint64_t seed) override { return CreateCudaGenerator(device.index(), seed); }

    const Generator &GetDefault(const Device &device) override { return GetDefaultCudaGenerator(device.index()); }

    void ManualSeedAll(uint64_t seed) override { ::infini_train::core::cuda::ManualSeedAll(seed); }
};

INFINI_TRAIN_REGISTER_GENERATOR_BACKEND(Device::DeviceType::kCUDA, CUDAGeneratorBackend);

} // namespace
} // namespace infini_train::core::cuda
