#include "infini_train/include/nn/parallel/pp/send_recv.h"

#include <stddef.h>

#include <algorithm>
#include <memory>
#include <ostream>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/nn/parallel/process_group.h"
#include "infini_train/include/nn/parallel/rank.h"
#include "infini_train/include/nn/parallel/utils.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {

namespace functions {
class ISend : public autograd::Function {
 public:
  static constexpr char kType[] = "ISendFunction";

  explicit ISend(Device target_device, int peer_rank,
                 const std::vector<std::vector<int64_t>> &shape,
                 DataType pipeline_dtype)
      : autograd::Function(kType),
        target_device_(target_device),
        peer_rank_(peer_rank),
        shapes_(shape),
        pipeline_dtype_(pipeline_dtype) {}

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

  void SetupContext(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors,
      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

  std::vector<std::shared_ptr<Tensor>> Backward(
      const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

 private:
  Device target_device_;
  Device input_device_;
  int peer_rank_ = -1;
  const std::vector<std::vector<int64_t>> &shapes_;
  // Dtype shared by activation and gradient payloads on the pipeline wire.
  DataType pipeline_dtype_ = DataType::kFLOAT32;
  // Forward tensors converted to the pipeline dtype before sending.
  std::vector<std::shared_ptr<Tensor>> wire_tensors_;
  // Backward receive buffers allocated with the pipeline dtype.
  std::vector<std::shared_ptr<Tensor>> wire_grad_tensors_;
  // Original forward input dtypes used to restore received input gradients.
  std::vector<DataType> input_dtypes_;
};

class IRecv : public autograd::Function {
 public:
  static constexpr char kType[] = "IRecvFunction";

  explicit IRecv(Device src_device, int peer_rank, DataType pipeline_dtype)
      : autograd::Function(kType),
        src_device_(src_device),
        peer_rank_(peer_rank),
        pipeline_dtype_(pipeline_dtype) {}

  std::vector<std::shared_ptr<Tensor>> Forward(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

  void SetupContext(
      const std::vector<std::shared_ptr<Tensor>> &input_tensors,
      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

  std::vector<std::shared_ptr<Tensor>> Backward(
      const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

 private:
  Device src_device_;
  Device cur_device_;
  int peer_rank_ = -1;
  // Dtype shared by activation and gradient payloads on the pipeline wire.
  DataType pipeline_dtype_ = DataType::kFLOAT32;
  // Number of forward outputs expected by backward.
  size_t output_count_ = 0;
  // Backward gradients converted to the pipeline dtype before sending.
  std::vector<std::shared_ptr<Tensor>> wire_grad_tensors_;
};

std::vector<std::shared_ptr<Tensor>> ISend::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  CHECK_EQ(input_tensors.size(), shapes_.size())
      << "Pipeline send tensor count does not match the configured receive "
         "shapes";

  CHECK(!input_tensors.empty());
  const auto &input = input_tensors[0];
  CHECK(input != nullptr);
  input_device_ = input->GetDevice();

  wire_tensors_.clear();
  wire_tensors_.reserve(input_tensors.size());
  for (const auto &tensor : input_tensors) {
    CHECK(tensor != nullptr);
    if (tensor->Dtype() == pipeline_dtype_) {
      wire_tensors_.push_back(tensor);
    } else {
      wire_tensors_.push_back(
          std::make_shared<Tensor>(tensor->To(pipeline_dtype_)));
    }
    CHECK(wire_tensors_.back()->Dtype() == pipeline_dtype_);
  }

  auto pp_group = ProcessGroupFactory::Instance(input_device_.type())
                      ->Get(GetPipelineParallelProcessGroupName(
                          input_device_.Rank().GlobalRank()));

  pp_group->Send(wire_tensors_, peer_rank_, false);

  return input_tensors;
}

void ISend::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors,
    const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
  input_dtypes_.clear();
  input_dtypes_.reserve(input_tensors.size());
  for (const auto &tensor : input_tensors) {
    CHECK(tensor != nullptr);
    input_dtypes_.push_back(tensor->Dtype());
  }
}

std::vector<std::shared_ptr<Tensor>> ISend::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
  wire_grad_tensors_.clear();
  CHECK_EQ(input_dtypes_.size(), shapes_.size());
  for (size_t shape_i = 0; shape_i < shapes_.size(); ++shape_i) {
    auto r_tensor = std::make_shared<Tensor>(shapes_[shape_i], pipeline_dtype_,
                                             input_device_);
    wire_grad_tensors_.push_back(r_tensor);
  }

  auto pp_group = ProcessGroupFactory::Instance(input_device_.type())
                      ->Get(GetPipelineParallelProcessGroupName(
                          input_device_.Rank().GlobalRank()));

  pp_group->Recv(wire_grad_tensors_, peer_rank_, false);

  std::vector<std::shared_ptr<Tensor>> input_grads;
  input_grads.reserve(wire_grad_tensors_.size());
  for (size_t i = 0; i < wire_grad_tensors_.size(); ++i) {
    if (wire_grad_tensors_[i]->Dtype() == input_dtypes_[i]) {
      input_grads.push_back(wire_grad_tensors_[i]);
    } else {
      input_grads.push_back(std::make_shared<Tensor>(
          wire_grad_tensors_[i]->To(input_dtypes_[i])));
    }
  }
  return input_grads;
}

std::vector<std::shared_ptr<Tensor>> IRecv::Forward(
    const std::vector<std::shared_ptr<Tensor>> &recv_tensors) {
  auto pp_group = ProcessGroupFactory::Instance(src_device_.type())
                      ->Get(GetPipelineParallelProcessGroupName(
                          src_device_.Rank().GlobalRank()));
  pp_group->Recv(recv_tensors, peer_rank_, false);
  return recv_tensors;
}

void IRecv::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors,
    const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
  output_count_ = output_tensors.size();
  if (output_tensors.empty()) {
    return;
  }
  cur_device_ = output_tensors[0]->GetDevice();
}

std::vector<std::shared_ptr<Tensor>> IRecv::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
  auto pp_group = ProcessGroupFactory::Instance(cur_device_.type())
                      ->Get(GetPipelineParallelProcessGroupName(
                          cur_device_.Rank().GlobalRank()));

  CHECK_EQ(grad_outputs.size(), output_count_);
  wire_grad_tensors_ = grad_outputs;
  for (size_t i = 0; i < wire_grad_tensors_.size(); ++i) {
    if (wire_grad_tensors_[i]->Dtype() != pipeline_dtype_) {
      wire_grad_tensors_[i] =
          std::make_shared<Tensor>(wire_grad_tensors_[i]->To(pipeline_dtype_));
    }
  }
  pp_group->Send(wire_grad_tensors_, peer_rank_, false);

  return wire_grad_tensors_;
}
}  // namespace functions

std::vector<std::shared_ptr<Tensor>> ISend(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors,
    Device target_device, int peer_rank,
    const std::vector<std::vector<int64_t>> &shape, DataType pipeline_dtype) {
  auto func = std::make_shared<functions::ISend>(target_device, peer_rank,
                                                 shape, pipeline_dtype);
  return func->Apply(input_tensors);
}

std::vector<std::shared_ptr<Tensor>> IRecv(
    const std::vector<std::shared_ptr<Tensor>> &outputs, Device src_device,
    int peer_rank, DataType pipeline_dtype) {
  auto func =
      std::make_shared<functions::IRecv>(src_device, peer_rank, pipeline_dtype);
  return func->Apply(outputs);
}
}  // namespace infini_train::nn::parallel
