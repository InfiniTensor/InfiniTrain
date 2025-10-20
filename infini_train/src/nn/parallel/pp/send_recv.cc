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
    const auto &input = input_tensors[0];
    input_device_ = input->GetDevice();

    auto device_type = input_device_->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device_type, "CommNcclSend"});
    kernel.Call<std::vector<std::shared_ptr<Tensor>>>(input_tensors, peer_rank_);
    std::vector<std::shared_ptr<Tensor>> outputs;
    for (int i = 0; i < input_tensors.size(); i++) {
        auto &t = input_tensors[i];
        auto t_item = std::make_shared<Tensor>(*t);
        i == 0 ? t_item->set_requires_grad(true) : t_item->set_requires_grad(false);
        outputs.push_back(t_item);
    }
    return outputs;
}

void ISend::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                         const std::vector<std::shared_ptr<Tensor>> &output_tensors) {}

std::vector<std::shared_ptr<Tensor>> ISend::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    auto device_type = input_device_->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device_type, "CommNcclRecv"});

    auto shapes = shapes_;
    std::vector<std::shared_ptr<Tensor>> recv_tensors;
    auto need_grad = OutputRequiresGrad(grad_outputs);
    for (int shape_i = 0; shape_i < shapes.size(); ++shape_i) {
        auto r_tensor = std::make_shared<Tensor>(shapes[shape_i], DataType::kFLOAT32, input_device_);
        if (need_grad[shape_i]) {
            recv_tensors.push_back(r_tensor);
        }
    }

    auto grad_inputs = kernel.Call<std::vector<std::shared_ptr<Tensor>>>(recv_tensors, peer_rank_);

    return grad_inputs;
}

std::vector<std::shared_ptr<Tensor>> IRecv::Forward(const std::vector<std::shared_ptr<Tensor>> &recv_tensors) {
    CHECK_NE(src_device_, nullptr) << "src_device_ must be set";

    auto device_type = src_device_->Type();

    auto kernel = Dispatcher::Instance().GetKernel({device_type, "CommNcclRecv"});

    kernel.Call<std::vector<std::shared_ptr<Tensor>>>(recv_tensors, peer_rank_);
    std::vector<std::shared_ptr<Tensor>> outputs;
    for (auto t : recv_tensors) {
        auto t_item = std::make_shared<Tensor>(*t);
        t_item->set_requires_grad(true);
        outputs.push_back(t_item);
    }

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
    auto device_type = cur_device_->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device_type, "CommNcclSend"});

    auto grad_remote = kernel.Call<std::vector<std::shared_ptr<Tensor>>>(grad_outputs, peer_rank_);
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
