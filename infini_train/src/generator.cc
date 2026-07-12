#include "infini_train/include/generator.h"

#include "glog/logging.h"

#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/cpu/cpu_generator_impl.h"

#ifdef USE_CUDA
#include "infini_train/src/core/runtime/cuda/cuda_generator_impl.h"
#endif

namespace infini_train {

// ============================================================
// Generator 构造函数（null 检查需要 glog，放 .cc）
// ============================================================

Generator::Generator(std::shared_ptr<GeneratorImpl> impl)
    : impl_(std::move(impl)) {
    CHECK(impl_) << "GeneratorImpl with nullptr is not supported";
}

// ============================================================
// Generator — 状态序列化（需要 Tensor 完整定义，放 .cc）
// ============================================================

void Generator::set_state(const Tensor &state) {
    CHECK(state.defined()) << "Undefined tensor is not allowed";
    impl_->set_state(state);
}

std::shared_ptr<Tensor> Generator::get_state() const {
    return impl_->get_state();
}

namespace detail {

void check_rng_state(const Tensor &state) {
    CHECK(state.GetDevice().IsCPU()) << "RNG state must be a CPU tensor";
    CHECK_EQ(static_cast<int>(state.Dtype()), static_cast<int>(DataType::kUINT8))
        << "RNG state must be a UINT8 tensor";
}

} // namespace detail

void manual_seed(uint64_t seed) {
    core::cpu::manual_seed(seed);

#ifdef USE_CUDA
    core::cuda::manual_seed_all(seed);
#endif
}

} // namespace infini_train
