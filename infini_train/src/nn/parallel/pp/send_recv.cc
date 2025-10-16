#include "infini_train/src/nn/parallel/pp/send_recv.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::pipeline {

namespace functions {
class ISend : public autograd::Function {
public:
    static constexpr char kType[] = "ISendFunction";

    explicit ISend(const Device *target_device, int cur_rank, int peer_rank, std::vector<std::vector<int64_t>> shape)
        : autograd::Function(kType), target_device_(target_device), cur_rank_(cur_rank), peer_rank_(peer_rank),
          shapes_(shape) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

    std::vector<bool> InputRequiresGrad(const std::vector<std::shared_ptr<Tensor>> &inputs) override {
        std::vector<bool> requires_grads(inputs.size());
        for (int i = 0; i < requires_grads.size(); i++) { requires_grads[i] = (i == 0 ? true : false); }
        return requires_grads;
    }

    std::vector<bool> OutputRequiresGrad(const std::vector<std::shared_ptr<Tensor>> &inputs) override {
        std::vector<bool> requires_grads(inputs.size());
        for (int i = 0; i < requires_grads.size(); i++) { requires_grads[i] = (i == 0 ? true : false); }
        return requires_grads;
    }

private:
    const Device *target_device_;
    const Device *input_device_ = nullptr;
    int cur_rank_ = -1;
    int peer_rank_ = -1;
    std::vector<std::vector<int64_t>> shapes_;
};

class IRecv : public autograd::Function {
public:
    static constexpr char kType[] = "IRecvFunction";

    explicit IRecv(const Device *src_device, int cur_rank, int peer_rank)
        : autograd::Function(kType), src_device_(src_device), cur_rank_(cur_rank), peer_rank_(peer_rank) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

    std::vector<bool> InputRequiresGrad(const std::vector<std::shared_ptr<Tensor>> &inputs) override {
        std::vector<bool> requires_grads(inputs.size());
        for (int i = 0; i < requires_grads.size(); i++) { requires_grads[i] = (i == 0 ? true : false); }
        return requires_grads;
    }

    std::vector<bool> OutputRequiresGrad(const std::vector<std::shared_ptr<Tensor>> &inputs) override {
        std::vector<bool> requires_grads(inputs.size());
        for (int i = 0; i < requires_grads.size(); i++) { requires_grads[i] = (i == 0 ? true : false); }
        return requires_grads;
    }

private:
    const Device *src_device_ = nullptr;
    const Device *cur_device_ = nullptr;
    int cur_rank_ = -1;
    int peer_rank_ = -1;
};

std::vector<std::shared_ptr<Tensor>> ISend::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    // printf("ISend::Forward Entry!!! %d\n", cur_rank_);
    const auto &input = input_tensors[0];
    input_device_ = input->GetDevice();

    auto device_type = input_device_->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device_type, "CommNcclSend"});
    kernel.Call<std::vector<std::shared_ptr<Tensor>>>(input_tensors, peer_rank_);
    // auto outputs = kernel.Call<std::vector<std::shared_ptr<Tensor>>>(input_tensors, peer_rank_);
    std::vector<std::shared_ptr<Tensor>> outputs;
    for (int i = 0; i < input_tensors.size(); i++) {
        auto &t = input_tensors[i];
        auto t_item = std::make_shared<Tensor>(*t);
        i == 0 ? t_item->set_requires_grad(true) : t_item->set_requires_grad(false);
        outputs.push_back(t_item);
    }
    // printf("ISend::Forward OK!!! %d\n", cur_rank_);
    return outputs;
}

void ISend::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                         const std::vector<std::shared_ptr<Tensor>> &output_tensors) {}

std::vector<std::shared_ptr<Tensor>> ISend::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // printf("[stage: %d] ISend::Backward Entry!!!!!\n", cur_rank_);

    auto device_type = input_device_->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device_type, "CommNcclRecv"});

    auto shapes = shapes_;
    std::vector<std::shared_ptr<Tensor>> recv_tensors;
    for (int shape_i = 0; shape_i < shapes.size(); ++shape_i) {
        auto r_tensor = std::make_shared<Tensor>(shapes[shape_i], DataType::kFLOAT32, input_device_);

        recv_tensors.push_back(r_tensor);
        // TODO:hack
        break;
    }
    // printf("[stages %d / %d] 反向接收的对象是 [stages %d]\n", cur_rank_, input_device_->Index(), peer_rank_);
    auto grad_inputs = kernel.Call<std::vector<std::shared_ptr<Tensor>>>(recv_tensors, peer_rank_);
    // printf("[stages %d] ISend::Backward OK!!!!!\n", cur_rank_);

    // if (grad_inputs.empty()) {
    //     printf("[Rank %d] ERROR: grad_inputs 为空！\n", cur_rank_);
    // } else {
    //     printf("[Rank %d] 成功接收到 %zu 个梯度张量。\n", cur_rank_, grad_inputs.size());

    //     for (size_t i = 0; i < grad_inputs.size(); ++i) {
    //         auto &grad_tensor = grad_inputs[i];

    //         // 检查指针是否为空
    //         if (!grad_tensor) {
    //             printf("[Rank %d] ERROR: grad_inputs[%zu] 是空指针！\n", cur_rank_, i);
    //             continue;
    //         }

    //         // 打印张量的基本信息
    //         printf("[Rank %d] grad_inputs[%zu]: ", cur_rank_, i);
    //         printf("设备=%d, ", grad_tensor->GetDevice()->Index());
    //         printf("数据类型=%d, ", static_cast<int>(grad_tensor->Dtype()));
    //         printf("维度=[");

    //         auto dims = grad_tensor->Dims();
    //         for (size_t dim = 0; dim < dims.size(); ++dim) {
    //             printf("%ld", dims[dim]);
    //             if (dim < dims.size() - 1) {
    //                 printf(", ");
    //             }
    //         }
    //         printf("], ");
    //         printf("元素数量=%ld\n", grad_tensor->NumElements());

    //         auto grad_tensor_copy = grad_tensor->To(DeviceManager::Instance()->GetDefaultDevice());
    //         if (grad_tensor->NumElements() > 0 && grad_tensor->Dtype() == DataType::kFLOAT32) {
    //             float *data_ptr = static_cast<float *>(grad_tensor_copy.DataPtr());
    //             if (!data_ptr) {
    //                 printf("接收的梯度是空\n");
    //                 break;
    //             }

    //             printf(", 前n个值=");
    //             for (int i = 0; i < 20; i++) { std::cout << data_ptr[i] << " "; }
    //         }

    //         printf("\n");
    //     }
    //     printf("验证OK\n");
    // }

    return grad_inputs;
}

std::vector<std::shared_ptr<Tensor>> IRecv::Forward(const std::vector<std::shared_ptr<Tensor>> &recv_tensors) {
    CHECK_NE(src_device_, nullptr) << "src_device_ must be set";

    auto device_type = src_device_->Type();
    // printf("[stage %d] IRecv::Forward %d\n", cur_rank_, device_type);
    auto kernel = Dispatcher::Instance().GetKernel({device_type, "CommNcclRecv"});

    kernel.Call<std::vector<std::shared_ptr<Tensor>>>(recv_tensors, peer_rank_);
    std::vector<std::shared_ptr<Tensor>> outputs;
    for (auto t : recv_tensors) {
        auto t_item = std::make_shared<Tensor>(*t);
        t_item->set_requires_grad(true);
        outputs.push_back(t_item);
    }

    // printf("[stage:  %d] IRecv::Forward OK!!!\n", cur_rank_);
    return outputs;
}

void IRecv::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                         const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    if (output_tensors.empty()) {
        return;
    }
    auto device = output_tensors[0]->GetDevice();
    cur_device_ = device;
}

std::vector<std::shared_ptr<Tensor>> IRecv::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // printf("IRecv::Backward Entry!!!!!\n");

    auto device_type = cur_device_->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device_type, "CommNcclSend"});

    // for (size_t i = 0; i < grad_outputs.size(); ++i) {
    //     auto &grad_tensor = grad_outputs[i];
    //     // 打印张量的基本信息
    //     printf("[Rank %d] grad_inputs[%zu]: ", cur_rank_, i);
    //     printf("设备=%d, ", grad_tensor->GetDevice()->Index());
    //     printf("数据类型=%d, ", static_cast<int>(grad_tensor->Dtype()));
    //     printf("维度=[");
    //     auto dims = grad_tensor->Dims();
    //     for (size_t dim = 0; dim < dims.size(); ++dim) {
    //         printf("%ld", dims[dim]);
    //         if (dim < dims.size() - 1) {
    //             printf(", ");
    //         }
    //     }
    //     printf("], ");
    //     printf("元素数量=%ld\n", grad_tensor->NumElements());

    //     auto grad_tensor_copy = grad_tensor->To(DeviceManager::Instance()->GetDefaultDevice());
    //     if (grad_tensor->NumElements() > 0 && grad_tensor->Dtype() == DataType::kFLOAT32) {
    //         float *data_ptr = static_cast<float *>(grad_tensor_copy.DataPtr());

    //         printf(", 前n个值=");
    //         for (int i = 0; i < 20; i++) { std::cout << data_ptr[i] << " "; }
    //     }

    //     printf("\n");
    // }
    // printf("[stages %d / %d] 发送的对象是 [stages %d]\n", cur_rank_, cur_device_->Index(), peer_rank_);
    auto grad_remote = kernel.Call<std::vector<std::shared_ptr<Tensor>>>(grad_outputs, peer_rank_);
    // printf("[stage %d] IRecv::Backward OK!!!!! \n", cur_rank_);
    return grad_remote;
}
} // namespace functions

std::vector<std::shared_ptr<Tensor>> ISend(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                           const Device *target_device, int cur_rank, int peer_rank,
                                           std::vector<std::vector<int64_t>> shape) {
    auto func = std::make_shared<functions::ISend>(target_device, cur_rank, peer_rank, shape);
    return func->Apply(input_tensors);
}

std::vector<std::shared_ptr<Tensor>> IRecv(const std::vector<std::shared_ptr<Tensor>> &outputs,
                                           const Device *src_device, int cur_rank, int peer_rank) {
    auto func = std::make_shared<functions::IRecv>(src_device, cur_rank, peer_rank);
    return func->Apply(outputs);
}
} // namespace infini_train::nn::pipeline
