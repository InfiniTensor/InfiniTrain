#include "infini_train/include/generator.h"

#include "glog/logging.h"

#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"

#ifdef USE_CUDA
#include "infini_train/src/core/runtime/cuda/cuda_generator_impl.h"
#endif

namespace infini_train {

Generator::Generator(std::shared_ptr<GeneratorImpl> impl) : impl_(std::move(impl)) {
    CHECK(impl_) << "GeneratorImpl with nullptr is not supported";
}

void Generator::set_state(const Tensor &state) {
    CHECK(state.defined()) << "Undefined tensor is not allowed";
    impl_->set_state(state);
}

std::shared_ptr<Tensor> Generator::get_state() const { return impl_->get_state(); }

namespace detail {

void check_rng_state(const Tensor &state) {
    CHECK(state.GetDevice().IsCPU()) << "RNG state must be a CPU tensor";
    CHECK_EQ(static_cast<int>(state.Dtype()), static_cast<int>(DataType::kUINT8)) << "RNG state must be a UINT8 tensor";
}

} // namespace detail

Generator CreateGenerator(const Device &device, uint64_t seed) {
    if (device.IsCPU()) {
        return core::cpu::createCPUGenerator(seed);
    }

#ifdef USE_CUDA
    if (device.IsCUDA()) {
        return core::cuda::createCUDAGenerator(device.index(), seed);
    }
#else
    if (device.IsCUDA()) {
        throw std::invalid_argument("CUDA generator requested but CUDA support is not enabled");
    }
#endif

    throw std::invalid_argument("Generator can only be created for CPU or CUDA devices");
}

const Generator &GetDefaultGenerator(const Device &device) {
    if (device.IsCPU()) {
        return core::cpu::getDefaultCPUGenerator();
    }

#ifdef USE_CUDA
    if (device.IsCUDA()) {
        return core::cuda::getDefaultCUDAGenerator(device.index());
    }
#else
    if (device.IsCUDA()) {
        throw std::invalid_argument("CUDA default generator requested but CUDA support is not enabled");
    }
#endif

    throw std::invalid_argument("Default generator can only be requested for CPU or CUDA devices");
}

void manual_seed(uint64_t seed) {
    core::cpu::manual_seed(seed);

#ifdef USE_CUDA
    core::cuda::manual_seed_all(seed);
#endif
}

} // namespace infini_train
