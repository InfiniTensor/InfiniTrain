#include <cstdint>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

std::vector<std::shared_ptr<Tensor>> Broadcast(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                               const std::vector<const Device *> &devices) {
    std::vector<std::shared_ptr<Tensor>> outputs;
    for (int i = 0; i < devices.size(); ++i) {
        for (const auto &tensor : input_tensors) {
            outputs.push_back(std::make_shared<Tensor>(tensor->To(devices[i])));
        }
    }
    return outputs;
}

std::vector<std::shared_ptr<Tensor>> ReduceAddCoalesced(const std::vector<std::vector<std::shared_ptr<Tensor>>> &grads,
                                                        const Device *destination) {
    std::vector<std::shared_ptr<Tensor>> outputs;
    auto kernel = Dispatcher::Instance().GetKernel({destination->Type(), "AccumulateGrad"});
    std::vector<std::vector<std::shared_ptr<Tensor>>> to_destination_grads;
    for (int i = 0; i < grads[0].size(); ++i) {
        outputs.emplace_back(std::make_shared<Tensor>(grads[0][i]->Dims(), grads[0][i]->Dtype(), destination));
        outputs[i]->Fill<float>(0.0);
    }
    for (int i = 0; i < grads.size(); ++i) {
        to_destination_grads.push_back(std::vector<std::shared_ptr<Tensor>>());
        for (int j = 0; j < grads[i].size(); ++j) {
            to_destination_grads[i].push_back(std::make_shared<Tensor>(grads[i][j]->To(destination)));
        }
    }
    for (int i = 0; i < grads.size(); ++i) {
        for (int j = 0; j < grads[i].size(); ++j) {
            kernel.Call<void>(to_destination_grads[i][j], static_cast<float>(1.0), outputs[j]);
        }
    }
    return outputs;
}

std::vector<std::shared_ptr<Tensor>> Scatter(const std::shared_ptr<Tensor> &tensor, std::vector<const Device *> devices,
                                             int64_t dim) {
    std::vector<std::shared_ptr<Tensor>> outputs;
    // FIXME(dcj): do split without autograd
    std::vector<std::shared_ptr<Tensor>> split_tensors = tensor->Split(tensor->Dims()[dim] / devices.size(), dim);
    for (auto i = 0; i < devices.size(); ++i) {
        outputs.push_back(std::make_shared<Tensor>(split_tensors[i]->To(devices[i])));
    }
    return outputs;
}

std::shared_ptr<Tensor> Gather(const std::vector<std::shared_ptr<Tensor>> &tensors, const Device *destination,
                               int64_t dim) {
    std::vector<std::shared_ptr<Tensor>> outputs;
    for (const auto &tensor : tensors) { outputs.push_back(std::make_shared<Tensor>(tensor->To(destination))); }
    auto kernel = Dispatcher::Instance().GetKernel({tensors[0]->GetDevice()->Type(), "StackForward"});
    auto gathered_tensor = kernel.Call<std::shared_ptr<Tensor>>(outputs, dim);
    auto old_dims = gathered_tensor->Dims();
    std::vector<int64_t> new_dims{old_dims[0] * old_dims[1]};
    for (int i = 2; i < old_dims.size(); ++i) { new_dims.push_back(old_dims[i]); }
    auto view_kernel = Dispatcher::Instance().GetKernel({destination->Type(), "NoOpForward"});
    return view_kernel.Call<std::shared_ptr<Tensor>>(gathered_tensor, new_dims);
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_COMM_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, Comm##kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_COMM_KERNEL(Broadcast)
REGISTER_CUDA_COMM_KERNEL(Scatter)
REGISTER_CUDA_COMM_KERNEL(Gather)
REGISTER_CUDA_COMM_KERNEL(ReduceAddCoalesced)

#undef REGISTER_CUDA_COMM_KERNEL
