#include "infini_train/src/nn/parallel/scatter_gather.h"

#include <memory>
#include <vector>

#include "infini_train/include/autograd/function.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::parallel {
namespace functions {
class Scatter : public autograd::Function {
public:
    static constexpr char kType[] = "ScatterFunction";

    explicit Scatter(const std::vector<const Device *> &target_gpus, int64_t dim)
        : autograd::Function(kType), target_gpus_(target_gpus), dim_(dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    std::vector<const Device *> target_gpus_;
    const Device *input_device_ = nullptr;
    int64_t dim_ = 0;
};

class Gather : public autograd::Function {
public:
    static constexpr char kType[] = "GatherFunction";

    explicit Gather(const Device *target_device, int64_t dim)
        : autograd::Function(kType), target_device_(target_device), dim_(dim) {}

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

    void SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                      const std::vector<std::shared_ptr<Tensor>> &output_tensors) override;

    std::vector<std::shared_ptr<Tensor>> Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) override;

private:
    const Device *target_device_ = nullptr;
    std::vector<const Device *> input_gpus_;
    int64_t dim_ = 0;
    bool unsqueezed_scalar_ = false;
};

std::vector<std::shared_ptr<Tensor>> Scatter::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    const auto &input = input_tensors[0];
    std::vector<std::shared_ptr<Tensor>> output_tensors;
    auto device = input->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "CommScatter"});
    output_tensors = kernel.Call<std::vector<std::shared_ptr<Tensor>>>(input, target_gpus_, dim_);
    return output_tensors;
}

void Scatter::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                           const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    input_device_ = input_tensors[0]->GetDevice();
}

std::vector<std::shared_ptr<Tensor>> Scatter::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    return std::make_shared<Gather>(input_device_, dim_)->Apply(grad_outputs);
}

std::vector<std::shared_ptr<Tensor>> Gather::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    for (const auto &tensor : input_tensors) {
        CHECK_NE(static_cast<int>(tensor->GetDevice()->Type()), static_cast<int>(DeviceType::kCPU))
            << "Gather function not implemented for CPU tensors";
    }
    if (dim_ == 0 && input_tensors[0]->Dims().size() == 0) {
        // FIXME(dcj): Here it is assumed that all tensors involved in the gather operation have the same shape.
        unsqueezed_scalar_ = true;
        LOG(WARNING) << "Was asked to gather along dimension 0, but all "
                        "input tensors were scalars; will instead unsqueeze "
                        "and return a vector.";
        // TODO(dcj): do unsqueeze here
    } else {
        unsqueezed_scalar_ = false;
    }
    auto device = input_tensors[0]->GetDevice()->Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "CommGather"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input_tensors, target_device_, dim_)};
}

void Gather::SetupContext(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                          const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    for (const auto &tensor : input_tensors) { input_gpus_.push_back(tensor->GetDevice()); }
}

std::vector<std::shared_ptr<Tensor>> Gather::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // TODO(dcj): do squeeze here if unsqueezed_scalar_ is true
    return std::make_shared<Scatter>(std::vector<const Device *>{input_gpus_}, dim_)->Apply(grad_outputs);
}
} // namespace functions

std::vector<std::vector<std::shared_ptr<Tensor>>> Scatter(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                                          const std::vector<const Device *> &devices, int dim) {
    std::vector<std::vector<std::shared_ptr<Tensor>>> output_tensors;
    for (const auto &tensor : input_tensors) {
        output_tensors.emplace_back(std::make_shared<functions::Scatter>(devices, dim)->Apply({tensor}));
    }
    std::vector<std::vector<std::shared_ptr<Tensor>>> transposed_output_tensors;
    transposed_output_tensors.resize(devices.size());
    for (int i = 0; i < devices.size(); ++i) {
        transposed_output_tensors[i].resize(input_tensors.size());
        for (int j = 0; j < input_tensors.size(); ++j) { transposed_output_tensors[i][j] = output_tensors[j][i]; }
    }
    return transposed_output_tensors;
}

std::vector<std::shared_ptr<Tensor>> Gather(const std::vector<std::vector<std::shared_ptr<Tensor>>> &outputs,
                                            const Device *target_device, int dim) {
    // FIXME(dcj): implement this
    std::vector<std::shared_ptr<Tensor>> gather_tensors;
    for (const auto &output : outputs) { gather_tensors.push_back(output[0]); }
    return std::make_shared<functions::Gather>(target_device, dim)->Apply(gather_tensors);
}
} // namespace infini_train::nn::parallel
