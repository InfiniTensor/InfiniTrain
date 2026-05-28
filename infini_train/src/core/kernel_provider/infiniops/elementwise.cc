#include <memory>
#include <numeric>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/kernel_provider/infiniops/adapter.h"
#include "infini_train/include/core/kernel_provider/infiniops_registry.h"
#include "infini_train/include/tensor.h"

#include <infini/ops.h>

namespace infini_train::kernel_provider::infiniops {
namespace {

std::vector<int64_t> ComputeBroadcastStrides(const std::vector<int64_t> &dims, const std::vector<int64_t> &out_dims) {
    CHECK_LE(dims.size(), out_dims.size());

    std::vector<int64_t> strides(dims.size());
    if (!dims.empty()) {
        strides.back() = 1;
        for (int i = static_cast<int>(dims.size()) - 2; i >= 0; --i) { strides[i] = strides[i + 1] * dims[i + 1]; }
    }

    const size_t pad = out_dims.size() - dims.size();
    std::vector<int64_t> out_strides(out_dims.size(), 0);
    for (size_t i = 0; i < dims.size(); ++i) {
        const int64_t dim = dims[i];
        const int64_t out_dim = out_dims[pad + i];
        CHECK(dim == out_dim || dim == 1) << "InfiniOps Add broadcast shape mismatch";
        out_strides[pad + i] = dim == 1 ? 0 : strides[i];
    }
    return out_strides;
}

infini::ops::Tensor ToBroadcastOpsTensor(const std::shared_ptr<Tensor> &tensor, const std::vector<int64_t> &out_dims,
                                         DataType dtype) {
    const auto strides = ComputeBroadcastStrides(tensor->Dims(), out_dims);
    return ToOpsTensor(tensor->DataPtr(), out_dims, dtype, tensor->GetDevice(), strides);
}

} // namespace

std::shared_ptr<Tensor> AddForward(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) {
    CHECK_GE(a->NumElements(), b->NumElements());
    CHECK_EQ(a->NumElements() % b->NumElements(), 0);

    auto a_dtype = a->Dtype();
    auto b_dtype = b->Dtype();
    DataType promoted_type = PromoteDataTypes(a_dtype, b_dtype);

    auto a_promoted = a_dtype == promoted_type ? a : std::make_shared<Tensor>(a->To(promoted_type));
    auto b_promoted = b_dtype == promoted_type ? b : std::make_shared<Tensor>(b->To(promoted_type));

    auto output = std::make_shared<Tensor>(a->Dims(), promoted_type, a->GetDevice());

    auto handle = GetHandle(a->GetDevice());
    auto a_ops = ToBroadcastOpsTensor(a_promoted, output->Dims(), promoted_type);
    auto b_ops = ToBroadcastOpsTensor(b_promoted, output->Dims(), promoted_type);
    auto c_ops = ToOpsTensor(output);

    {
        std::lock_guard<std::mutex> lock(InfiniOpsCallMutex());
        infini::ops::functional::Add(handle, {}, a_ops, b_ops, c_ops);
    }
    return output;
}

} // namespace infini_train::kernel_provider::infiniops

REGISTER_INFINIOPS_KERNEL(AddForward, infini_train::kernel_provider::infiniops::AddForward)
