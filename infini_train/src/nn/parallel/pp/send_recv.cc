#include "infini_train/include/nn/parallel/pp/send_recv.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

namespace functions {
class ISend : public autograd::Function {
public:
    static constexpr char kType[] = "ISendFunction";

    explicit ISend(const Device *target_device, int cur_rank, int peer_rank,
                   const std::vector<std::vector<int64_t>> &shape)
        : autograd::Function(kType), target_device_(target_device), cur_rank_(cur_rank), peer_rank_(peer_rank),
          shapes_(shape) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const Device *target_device_ = nullptr;
    const Device *input_device_ = nullptr;
    int cur_rank_ = -1;
    int peer_rank_ = -1;
    const std::vector<std::vector<int64_t>> &shapes_;
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

private:
    const Device *src_device_ = nullptr;
    const Device *cur_device_ = nullptr;
    int cur_rank_ = -1;
    int peer_rank_ = -1;
};

std::vector<std::shared_ptr<Tensor>> ISend::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    const auto &input = input_tensors[0];
    input_device_ = input->GetDevice();

    auto pp_group
        = ProcessGroupFactory::Instance()->Get(GetPipelineParallelProcessGroupName(input_device_->rank().GlobalRank()));

    pp_group->Send(input_tensors, peer_rank_, false);

    return input_tensors;
}

std::vector<std::shared_ptr<Tensor>> ISend::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    std::vector<std::shared_ptr<Tensor>> recv_tensors;
    for (int shape_i = 0; shape_i < shapes_.size(); ++shape_i) {
        // FIXME(jym): The data type between stages is not float32, which will cause a crash
        auto r_tensor = std::make_shared<Tensor>(shapes_[shape_i], DataType::kFLOAT32, input_device_);
        recv_tensors.push_back(r_tensor);
    }

    auto pp_group
        = ProcessGroupFactory::Instance()->Get(GetPipelineParallelProcessGroupName(input_device_->rank().GlobalRank()));

    pp_group->Recv(recv_tensors, peer_rank_, false);

    return recv_tensors;
}

std::vector<std::shared_ptr<Tensor>> IRecv::Forward(const std::vector<std::shared_ptr<Tensor>> &recv_tensors) {
    CHECK_NOTNULL(src_device_);
    auto pp_group
        = ProcessGroupFactory::Instance()->Get(GetPipelineParallelProcessGroupName(src_device_->rank().GlobalRank()));
    pp_group->Recv(recv_tensors, peer_rank_, false);

    return recv_tensors;
}

void IRecv::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                         const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    if (output_tensors.empty()) {
        return;
    }
    cur_device_ = output_tensors[0]->GetDevice();
}

std::vector<std::shared_ptr<Tensor>> IRecv::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    auto pp_group
        = ProcessGroupFactory::Instance()->Get(GetPipelineParallelProcessGroupName(cur_device_->rank().GlobalRank()));

    pp_group->Send(grad_outputs, peer_rank_, false);

    return grad_outputs;
}
} // namespace functions

std::vector<std::shared_ptr<Tensor>> ISend(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                           const Device *target_device, int cur_rank, int peer_rank,
                                           const std::vector<std::vector<int64_t>> &shape) {
    auto func = std::make_shared<functions::ISend>(target_device, cur_rank, peer_rank, shape);
    return func->Apply(input_tensors);
}

std::vector<std::shared_ptr<Tensor>> IRecv(const std::vector<std::shared_ptr<Tensor>> &outputs,
                                           const Device *src_device, int cur_rank, int peer_rank) {
    auto func = std::make_shared<functions::IRecv>(src_device, cur_rank, peer_rank);
    return func->Apply(outputs);
}
} // namespace infini_train::nn::parallel
