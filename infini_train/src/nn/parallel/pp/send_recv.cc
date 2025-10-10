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

    explicit ISend(const Device *target_device, int cur_rank, int target_rank)
        : autograd::Function(kType), target_device_(target_device), cur_rank_(cur_rank), target_rank_(target_rank) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const Device *target_device_;
    const Device *input_device_ = nullptr;
    int cur_rank_ = -1;
    int target_rank_ = -1;
};

class IRecv : public autograd::Function {
public:
    static constexpr char kType[] = "IRecvFunction";

    explicit IRecv(const Device *src_device, int cur_rank, int src_rank)
        : autograd::Function(kType), src_device_(src_device), cur_rank_(cur_rank), src_rank_(src_rank) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const Device *src_device_ = nullptr;
    const Device *cur_device_ = nullptr;
    int cur_rank_ = -1;
    int src_rank_ = -1;
};

std::vector<std::shared_ptr<Tensor>> ISend::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    const auto &input = input_tensors[0];
    input_device_ = input->GetDevice();

    auto device_type = input_device_->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device_type, "CommNcclSend"});

    auto output = kernel.Call<std::vector<std::shared_ptr<Tensor>>>(input_tensors, target_rank_);
    printf("ISend::Forward OK!!!\n");
    return {};
}

void ISend::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                         const std::vector<std::shared_ptr<Tensor>> &output_tensors) {}

std::vector<std::shared_ptr<Tensor>> ISend::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];

    auto device_type = target_device_->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device_type, "CommNcclSend"});

    auto grad_input = kernel.Call<std::vector<std::shared_ptr<Tensor>>>(grad_output, cur_rank_);
    printf("ISend::Backward OK!!!!!\n");
    return grad_input;
}

std::vector<std::shared_ptr<Tensor>> IRecv::Forward(const std::vector<std::shared_ptr<Tensor>> &recv_tensors) {
    CHECK_NE(src_device_, nullptr) << "src_device_ must be set";

    auto device_type = src_device_->Type();
    // printf("[stage %d] IRecv::Forward %d\n", cur_rank_, device_type);
    auto kernel = Dispatcher::Instance().GetKernel({device_type, "CommNcclRecv"});

    auto output = kernel.Call<std::vector<std::shared_ptr<Tensor>>>(recv_tensors, src_rank_);

    return output;
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
    const auto &grad_output = grad_outputs[0];

    auto device_type = grad_output->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device_type, "CommNcclSend"});

    auto grad_to_send = grad_output;

    auto grad_remote = kernel.Call<std::vector<std::shared_ptr<Tensor>>>(grad_to_send, src_rank_);
    printf("IRecv::Backward OK!!!!!\n");
    return grad_remote;
}
} // namespace functions

std::vector<std::shared_ptr<Tensor>> ISend(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                           const Device *target_device, int cur_rank, int target_rank) {
    auto func = std::make_shared<functions::ISend>(target_device, cur_rank, target_rank);
    return func->Forward(input_tensors);
}

std::vector<std::shared_ptr<Tensor>> IRecv(const std::vector<std::shared_ptr<Tensor>> &outputs,
                                           const Device *src_device, int cur_rank, int src_rank) {
    auto func = std::make_shared<functions::IRecv>(src_device, cur_rank, src_rank);
    return func->Apply(outputs);
}
} // namespace infini_train::nn::pipeline