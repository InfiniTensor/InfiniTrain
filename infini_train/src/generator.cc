#include "infini_train/include/generator.h"
#include "infini_train/include/generator_impl.h"

#include <memory>
#include <ostream>
#include <utility>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/generator_backend.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/tensor.h"

namespace infini_train {

Generator::Generator(std::shared_ptr<GeneratorImpl> impl) : impl_(std::move(impl)) {
    CHECK(impl_) << "GeneratorImpl with nullptr is not supported";
}

void Generator::set_current_seed(uint64_t seed) const { impl_->set_current_seed(seed); }

uint64_t Generator::current_seed() const { return impl_->current_seed(); }

uint64_t Generator::Seed() { return impl_->Seed(); }

void Generator::set_state(const Tensor &state) {
    CHECK(state.defined()) << "Undefined tensor is not allowed";
    impl_->set_state(state);
}

std::shared_ptr<Tensor> Generator::get_state() const { return impl_->get_state(); }

Device Generator::device() const { return impl_->device(); }

Generator Generator::Clone() const { return Generator(impl_->Clone()); }

namespace detail {

void CheckRngState(const Tensor &state) {
    CHECK(state.GetDevice().IsCPU()) << "RNG state must be a CPU tensor";
    CHECK_EQ(static_cast<int>(state.Dtype()), static_cast<int>(DataType::kUINT8)) << "RNG state must be a UINT8 tensor";
}

} // namespace detail

Generator CreateGenerator(const Device &device, uint64_t seed) {
    return GeneratorBackendRegistry::Instance().Get(device.type()).Create(device, seed);
}

const Generator &GetDefaultGenerator(const Device &device) {
    return GeneratorBackendRegistry::Instance().Get(device.type()).GetDefault(device);
}

void ManualSeed(uint64_t seed) { GeneratorBackendRegistry::Instance().ManualSeedAll(seed); }

} // namespace infini_train
