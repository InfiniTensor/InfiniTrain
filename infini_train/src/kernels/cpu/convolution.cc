#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/kernels/common/conv2d.h"

namespace infini_train::kernels::cpu {
namespace {
Conv2dShape Shape(const Tensor &x, const Tensor &w, int64_t stride, int64_t padding) {
    const auto &a = x.Dims();
    const auto &b = w.Dims();
    return {a[0],
            a[1],
            a[2],
            a[3],
            b[0],
            b[2],
            b[3],
            (a[2] + 2 * padding - b[2]) / stride + 1,
            (a[3] + 2 * padding - b[3]) / stride + 1,
            stride,
            padding};
}
} // namespace

std::shared_ptr<Tensor> Conv2dForward(const std::shared_ptr<Tensor> &x, const std::shared_ptr<Tensor> &w,
                                      const std::shared_ptr<Tensor> &b, int64_t stride, int64_t padding) {
    const auto s = Shape(*x, *w, stride, padding);
    auto y = std::make_shared<Tensor>(std::vector<int64_t>{s.n, s.co, s.oh, s.ow}, x->Dtype(), x->GetDevice());
    const auto xp = static_cast<const float *>(x->DataPtr());
    const auto wp = static_cast<const float *>(w->DataPtr());
    const auto bp = b ? static_cast<const float *>(b->DataPtr()) : nullptr;
    auto yp = static_cast<float *>(y->DataPtr());
    const int64_t count = y->NumElements();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < count; ++i) { yp[i] = Conv2dForwardAt(i, xp, wp, bp, s); }
    return y;
}

std::vector<std::shared_ptr<Tensor>> Conv2dBackward(const std::shared_ptr<Tensor> &x, const std::shared_ptr<Tensor> &w,
                                                    const std::shared_ptr<Tensor> &dy, int64_t stride, int64_t padding,
                                                    bool need_x, bool need_w, bool need_b) {
    const auto s = Shape(*x, *w, stride, padding);
    const auto xp = static_cast<const float *>(x->DataPtr());
    const auto wp = static_cast<const float *>(w->DataPtr());
    const auto gp = static_cast<const float *>(dy->DataPtr());
    std::vector<std::shared_ptr<Tensor>> grads(3);
    if (need_x) {
        grads[0] = std::make_shared<Tensor>(x->Dims(), x->Dtype(), x->GetDevice());
        auto p = static_cast<float *>(grads[0]->DataPtr());
        const int64_t count = x->NumElements();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int64_t i = 0; i < count; ++i) { p[i] = Conv2dInputGradAt(i, wp, gp, s); }
    }
    if (need_w) {
        grads[1] = std::make_shared<Tensor>(w->Dims(), w->Dtype(), w->GetDevice());
        auto p = static_cast<float *>(grads[1]->DataPtr());
        const int64_t count = w->NumElements();
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int64_t i = 0; i < count; ++i) { p[i] = Conv2dWeightGradAt(i, xp, gp, s); }
    }
    if (need_b) {
        grads[2] = std::make_shared<Tensor>(std::vector<int64_t>{s.co}, x->Dtype(), x->GetDevice());
        auto p = static_cast<float *>(grads[2]->DataPtr());
        for (int64_t i = 0; i < s.co; ++i) { p[i] = Conv2dBiasGradAt(i, gp, s); }
    }
    return grads;
}
} // namespace infini_train::kernels::cpu

REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, Conv2dForward, infini_train::kernels::cpu::Conv2dForward)
REGISTER_KERNEL(infini_train::Device::DeviceType::kCPU, Conv2dBackward, infini_train::kernels::cpu::Conv2dBackward)
