#include "infini_train/include/generator.h"

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

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
    impl_->set_state(state);
}

std::shared_ptr<Tensor> Generator::get_state() const {
    return impl_->get_state();
}

} // namespace infini_train
