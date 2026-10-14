#include "infini_train/include/nn/parallel/pp/send_recv.h"

#include <algorithm>
#include <cstddef>
#include <memory>
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

namespace {
std::shared_ptr<Tensor> MakeDtypeMetadata(
    const std::vector<std::shared_ptr<Tensor>> &tensors, Device device) {
  std::vector<float> dtype_values;
  dtype_values.reserve(tensors.size());
  for (const auto &tensor : tensors) {
    CHECK(tensor != nullptr);
    dtype_values.push_back(static_cast<float>(tensor->Dtype()));
  }
  return std::make_shared<Tensor>(
      dtype_values.data(),
      std::vector<int64_t>{static_cast<int64_t>(tensors.size())},
      DataType::kFLOAT32, device);
}
}  // namespace

namespace functions {
class ISend : public autograd::Function {
 public:
  static constexpr char kType[] = "ISendFunction";

  explicit ISend(Device target_device, int peer_rank,
                 const std::vector<std::vector<int64_t>> &shape)
      : autograd::Function(kType),
        target_device_(target_device),
        peer_rank_(peer_rank),
        shapes_(shape) {}

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
  std::vector<DataType> input_dtypes_;
};

class IRecv : public autograd::Function {
 public:
  static constexpr char kType[] = "IRecvFunction";

  explicit IRecv(Device src_device, int peer_rank)
      : autograd::Function(kType),
        src_device_(src_device),
        peer_rank_(peer_rank) {}

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
};

std::vector<std::shared_ptr<Tensor>> ISend::Forward(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
  const auto &input = input_tensors[0];
  input_device_ = input->GetDevice();

  auto pp_group = ProcessGroupFactory::Instance(input_device_.type())
                      ->Get(GetPipelineParallelProcessGroupName(
                          input_device_.Rank().GlobalRank()));

  pp_group->Send({MakeDtypeMetadata(input_tensors, input_device_)}, peer_rank_,
                 false);
  pp_group->Send(input_tensors, peer_rank_, false);

  return input_tensors;
}

void ISend::SetupContext(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors,
    const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
  CHECK_EQ(input_tensors.size(), shapes_.size())
      << "Pipeline send tensor count does not match the configured receive "
         "shapes";
  input_dtypes_.clear();
  input_dtypes_.reserve(input_tensors.size());
  for (const auto &tensor : input_tensors) {
    CHECK(tensor != nullptr);
    input_dtypes_.push_back(tensor->Dtype());
  }
}

std::vector<std::shared_ptr<Tensor>> ISend::Backward(
    const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
  std::vector<std::shared_ptr<Tensor>> recv_tensors;
  CHECK_EQ(input_dtypes_.size(), shapes_.size());
  for (size_t shape_i = 0; shape_i < shapes_.size(); ++shape_i) {
    auto r_tensor = std::make_shared<Tensor>(
        shapes_[shape_i], input_dtypes_[shape_i], input_device_);
    recv_tensors.push_back(r_tensor);
  }

  auto pp_group = ProcessGroupFactory::Instance(input_device_.type())
                      ->Get(GetPipelineParallelProcessGroupName(
                          input_device_.Rank().GlobalRank()));

  pp_group->Recv(recv_tensors, peer_rank_, false);

  return recv_tensors;
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

  pp_group->Send(grad_outputs, peer_rank_, false);

  return grad_outputs;
}
}  // namespace functions

std::vector<std::shared_ptr<Tensor>> ISend(
    const std::vector<std::shared_ptr<Tensor>> &input_tensors,
    Device target_device, int peer_rank,
    const std::vector<std::vector<int64_t>> &shape) {
  auto func =
      std::make_shared<functions::ISend>(target_device, peer_rank, shape);
  return func->Apply(input_tensors);
}

std::vector<DataType> ReceiveDtypes(size_t tensor_count, Device device,
                                    int peer_rank) {
  CHECK_GT(tensor_count, 0);
  auto metadata = std::make_shared<Tensor>(
      std::vector<int64_t>{static_cast<int64_t>(tensor_count)},
      DataType::kFLOAT32, device);
  auto pp_group = ProcessGroupFactory::Instance(device.type())
                      ->Get(GetPipelineParallelProcessGroupName(
                          device.Rank().GlobalRank()));
  pp_group->Recv({metadata}, peer_rank, false);

  auto host_metadata = metadata->To(Device());
  const auto *dtype_values =
      static_cast<const float *>(host_metadata.DataPtr());
  std::vector<DataType> dtypes;
  dtypes.reserve(tensor_count);
  for (size_t i = 0; i < tensor_count; ++i) {
    const auto dtype_value = static_cast<int>(dtype_values[i]);
    CHECK_GE(dtype_value, static_cast<int>(DataType::kBOOL));
    CHECK_LE(dtype_value, static_cast<int>(DataType::kFLOAT64));
    dtypes.push_back(static_cast<DataType>(dtype_value));
  }
  return dtypes;
}

std::vector<std::shared_ptr<Tensor>> IRecv(
    const std::vector<std::shared_ptr<Tensor>> &outputs, Device src_device,
    int peer_rank) {
  auto func = std::make_shared<functions::IRecv>(src_device, peer_rank);
  return func->Apply(outputs);
}
}  // namespace infini_train::nn::parallel
