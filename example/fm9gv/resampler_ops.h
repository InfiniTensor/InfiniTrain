#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "infini_train/include/autograd/function.h"

namespace infini_train {
class Tensor;
}

namespace fm9gv {

class InterpolateFunction : public infini_train::autograd::Function {
public:
    InterpolateFunction(std::vector<int64_t> size, std::string mode, std::optional<bool> align_corners,
                        bool antialias);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs) override;
    void SetupContext(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs,
                      const std::vector<std::shared_ptr<infini_train::Tensor>> &outputs) override;
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Backward(const std::vector<std::shared_ptr<infini_train::Tensor>> &grad_outputs) override;

private:
    std::vector<int64_t> size_;
    std::string mode_;
    std::optional<bool> align_corners_;
    bool antialias_;
    std::vector<int64_t> input_dims_;
};

class RoiAlignFunction : public infini_train::autograd::Function {
public:
    RoiAlignFunction(std::vector<int64_t> output_size, double spatial_scale, int64_t sampling_ratio,
                     bool aligned);

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs) override;
    void SetupContext(const std::vector<std::shared_ptr<infini_train::Tensor>> &inputs,
                      const std::vector<std::shared_ptr<infini_train::Tensor>> &outputs) override;
    std::vector<std::shared_ptr<infini_train::Tensor>>
    Backward(const std::vector<std::shared_ptr<infini_train::Tensor>> &grad_outputs) override;

private:
    std::vector<int64_t> output_size_;
    double spatial_scale_;
    int64_t sampling_ratio_;
    bool aligned_;
    std::vector<int64_t> input_dims_;
};

// PyTorch-compatible subset of torch.nn.functional.interpolate. The DCU
// implementation currently supports a 4-D NCHW tensor, explicit 2-D size,
// mode="bilinear", align_corners=false and antialias=false.
std::shared_ptr<infini_train::Tensor>
Interpolate(const std::shared_ptr<infini_train::Tensor> &input,
            const std::optional<std::vector<int64_t>> &size = std::nullopt,
            const std::optional<std::vector<double>> &scale_factor = std::nullopt,
            const std::string &mode = "nearest", std::optional<bool> align_corners = std::nullopt,
            std::optional<bool> recompute_scale_factor = std::nullopt, bool antialias = false);

// Matches torchvision.ops.roi_align(input, boxes, output_size,
// spatial_scale=1.0, sampling_ratio=-1, aligned=false). Boxes use [K, 5]
// format: batch_index, x1, y1, x2, y2.
std::shared_ptr<infini_train::Tensor>
RoiAlign(const std::shared_ptr<infini_train::Tensor> &input,
         const std::shared_ptr<infini_train::Tensor> &boxes, const std::vector<int64_t> &output_size,
         double spatial_scale = 1.0, int64_t sampling_ratio = -1, bool aligned = false);

} // namespace fm9gv
