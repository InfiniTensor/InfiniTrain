#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/common/dcu/common_dcu.h"
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/core/runtime/dcu/dcu_dispatch.h"
#include "infini_train/src/core/runtime/dcu/dcu_runtime_common.h"

namespace infini_train::kernels::dcu {
namespace {

hipStream_t GetStream(const Device &device) {
    return dynamic_cast<core::dcu::DcuStream *>(core::GetDeviceGuardImpl(device.type())->GetStream(device))
        ->hip_stream();
}

template <typename T>
__global__ void InterpolateBilinearForwardKernel(const T *__restrict__ input, T *__restrict__ output,
                                                 int64_t channels, int64_t input_height,
                                                 int64_t input_width, int64_t output_height,
                                                 int64_t output_width, int64_t total) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) { return; }
    const int64_t ox = idx % output_width;
    const int64_t oy = (idx / output_width) % output_height;
    const int64_t c = (idx / output_width / output_height) % channels;
    const int64_t b = idx / output_width / output_height / channels;
    float iy = (static_cast<float>(oy) + 0.5f) * input_height / output_height - 0.5f;
    float ix = (static_cast<float>(ox) + 0.5f) * input_width / output_width - 0.5f;
    iy = fmaxf(iy, 0.0f);
    ix = fmaxf(ix, 0.0f);
    const int64_t y0 = min(static_cast<int64_t>(iy), input_height - 1);
    const int64_t x0 = min(static_cast<int64_t>(ix), input_width - 1);
    const int64_t y1 = min(y0 + 1, input_height - 1);
    const int64_t x1 = min(x0 + 1, input_width - 1);
    const float ly = iy - y0;
    const float lx = ix - x0;
    const int64_t base = (b * channels + c) * input_height * input_width;
    const float v00 = common::dcu::Cast<float>(input[base + y0 * input_width + x0]);
    const float v01 = common::dcu::Cast<float>(input[base + y0 * input_width + x1]);
    const float v10 = common::dcu::Cast<float>(input[base + y1 * input_width + x0]);
    const float v11 = common::dcu::Cast<float>(input[base + y1 * input_width + x1]);
    output[idx] = common::dcu::Cast<T>((1 - ly) * ((1 - lx) * v00 + lx * v01)
                                      + ly * ((1 - lx) * v10 + lx * v11));
}

template <typename T>
__global__ void InterpolateBilinearBackwardKernel(const T *__restrict__ grad_output,
                                                  T *__restrict__ grad_input, int64_t channels,
                                                  int64_t input_height, int64_t input_width,
                                                  int64_t output_height, int64_t output_width,
                                                  int64_t total) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) { return; }
    const int64_t ox = idx % output_width;
    const int64_t oy = (idx / output_width) % output_height;
    const int64_t c = (idx / output_width / output_height) % channels;
    const int64_t b = idx / output_width / output_height / channels;
    float iy = (static_cast<float>(oy) + 0.5f) * input_height / output_height - 0.5f;
    float ix = (static_cast<float>(ox) + 0.5f) * input_width / output_width - 0.5f;
    iy = fmaxf(iy, 0.0f);
    ix = fmaxf(ix, 0.0f);
    const int64_t y0 = min(static_cast<int64_t>(iy), input_height - 1);
    const int64_t x0 = min(static_cast<int64_t>(ix), input_width - 1);
    const int64_t y1 = min(y0 + 1, input_height - 1);
    const int64_t x1 = min(x0 + 1, input_width - 1);
    const float ly = iy - y0;
    const float lx = ix - x0;
    const float grad = common::dcu::Cast<float>(grad_output[idx]);
    const int64_t base = (b * channels + c) * input_height * input_width;
    common::dcu::AtomicAdd(&grad_input[base + y0 * input_width + x0], common::dcu::Cast<T>((1 - ly) * (1 - lx) * grad));
    common::dcu::AtomicAdd(&grad_input[base + y0 * input_width + x1], common::dcu::Cast<T>((1 - ly) * lx * grad));
    common::dcu::AtomicAdd(&grad_input[base + y1 * input_width + x0], common::dcu::Cast<T>(ly * (1 - lx) * grad));
    common::dcu::AtomicAdd(&grad_input[base + y1 * input_width + x1], common::dcu::Cast<T>(ly * lx * grad));
}

__device__ bool RoiBilinear(float y, float x, int64_t height, int64_t width, int64_t &y0,
                            int64_t &x0, int64_t &y1, int64_t &x1, float &w00, float &w01,
                            float &w10, float &w11) {
    if (y < -1.0f || y > height || x < -1.0f || x > width) { return false; }
    y = fmaxf(y, 0.0f);
    x = fmaxf(x, 0.0f);
    y0 = min(static_cast<int64_t>(y), height - 1);
    x0 = min(static_cast<int64_t>(x), width - 1);
    y1 = min(y0 + 1, height - 1);
    x1 = min(x0 + 1, width - 1);
    const float ly = y - y0;
    const float lx = x - x0;
    w00 = (1 - ly) * (1 - lx);
    w01 = (1 - ly) * lx;
    w10 = ly * (1 - lx);
    w11 = ly * lx;
    return true;
}

template <typename T>
__global__ void RoiAlignForwardKernel(const T *__restrict__ input, const float *__restrict__ boxes,
                                      T *__restrict__ output, int64_t channels, int64_t height,
                                      int64_t width, int64_t pooled_height, int64_t pooled_width,
                                      float spatial_scale, int64_t sampling_ratio, bool aligned,
                                      int64_t total) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) { return; }
    const int64_t pw = idx % pooled_width;
    const int64_t ph = (idx / pooled_width) % pooled_height;
    const int64_t c = (idx / pooled_width / pooled_height) % channels;
    const int64_t roi = idx / pooled_width / pooled_height / channels;
    const float offset = aligned ? 0.5f : 0.0f;
    const int64_t batch = static_cast<int64_t>(boxes[roi * 5]);
    const float start_x = boxes[roi * 5 + 1] * spatial_scale - offset;
    const float start_y = boxes[roi * 5 + 2] * spatial_scale - offset;
    const float end_x = boxes[roi * 5 + 3] * spatial_scale - offset;
    const float end_y = boxes[roi * 5 + 4] * spatial_scale - offset;
    const float roi_width = aligned ? end_x - start_x : fmaxf(end_x - start_x, 1.0f);
    const float roi_height = aligned ? end_y - start_y : fmaxf(end_y - start_y, 1.0f);
    const float bin_width = roi_width / pooled_width;
    const float bin_height = roi_height / pooled_height;
    const int64_t sample_width = sampling_ratio > 0 ? sampling_ratio : max(static_cast<int64_t>(ceilf(roi_width / pooled_width)), int64_t{1});
    const int64_t sample_height = sampling_ratio > 0 ? sampling_ratio : max(static_cast<int64_t>(ceilf(roi_height / pooled_height)), int64_t{1});
    const int64_t base = (batch * channels + c) * height * width;
    float sum = 0.0f;
    for (int64_t iy = 0; iy < sample_height; ++iy) {
        const float y = start_y + ph * bin_height + (iy + 0.5f) * bin_height / sample_height;
        for (int64_t ix = 0; ix < sample_width; ++ix) {
            const float x = start_x + pw * bin_width + (ix + 0.5f) * bin_width / sample_width;
            int64_t y0, x0, y1, x1;
            float w00, w01, w10, w11;
            if (!RoiBilinear(y, x, height, width, y0, x0, y1, x1, w00, w01, w10, w11)) { continue; }
            sum += w00 * common::dcu::Cast<float>(input[base + y0 * width + x0])
                   + w01 * common::dcu::Cast<float>(input[base + y0 * width + x1])
                   + w10 * common::dcu::Cast<float>(input[base + y1 * width + x0])
                   + w11 * common::dcu::Cast<float>(input[base + y1 * width + x1]);
        }
    }
    output[idx] = common::dcu::Cast<T>(sum / (sample_height * sample_width));
}

template <typename T>
__global__ void RoiAlignBackwardKernel(const T *__restrict__ grad_output,
                                       const float *__restrict__ boxes, T *__restrict__ grad_input,
                                       int64_t channels, int64_t height, int64_t width,
                                       int64_t pooled_height, int64_t pooled_width, float spatial_scale,
                                       int64_t sampling_ratio, bool aligned, int64_t total) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) { return; }
    const int64_t pw = idx % pooled_width;
    const int64_t ph = (idx / pooled_width) % pooled_height;
    const int64_t c = (idx / pooled_width / pooled_height) % channels;
    const int64_t roi = idx / pooled_width / pooled_height / channels;
    const float offset = aligned ? 0.5f : 0.0f;
    const int64_t batch = static_cast<int64_t>(boxes[roi * 5]);
    const float start_x = boxes[roi * 5 + 1] * spatial_scale - offset;
    const float start_y = boxes[roi * 5 + 2] * spatial_scale - offset;
    const float end_x = boxes[roi * 5 + 3] * spatial_scale - offset;
    const float end_y = boxes[roi * 5 + 4] * spatial_scale - offset;
    const float roi_width = aligned ? end_x - start_x : fmaxf(end_x - start_x, 1.0f);
    const float roi_height = aligned ? end_y - start_y : fmaxf(end_y - start_y, 1.0f);
    const float bin_width = roi_width / pooled_width;
    const float bin_height = roi_height / pooled_height;
    const int64_t sample_width = sampling_ratio > 0 ? sampling_ratio : max(static_cast<int64_t>(ceilf(roi_width / pooled_width)), int64_t{1});
    const int64_t sample_height = sampling_ratio > 0 ? sampling_ratio : max(static_cast<int64_t>(ceilf(roi_height / pooled_height)), int64_t{1});
    const float grad = common::dcu::Cast<float>(grad_output[idx]) / (sample_height * sample_width);
    const int64_t base = (batch * channels + c) * height * width;
    for (int64_t iy = 0; iy < sample_height; ++iy) {
        const float y = start_y + ph * bin_height + (iy + 0.5f) * bin_height / sample_height;
        for (int64_t ix = 0; ix < sample_width; ++ix) {
            const float x = start_x + pw * bin_width + (ix + 0.5f) * bin_width / sample_width;
            int64_t y0, x0, y1, x1;
            float w00, w01, w10, w11;
            if (!RoiBilinear(y, x, height, width, y0, x0, y1, x1, w00, w01, w10, w11)) { continue; }
            common::dcu::AtomicAdd(&grad_input[base + y0 * width + x0], common::dcu::Cast<T>(grad * w00));
            common::dcu::AtomicAdd(&grad_input[base + y0 * width + x1], common::dcu::Cast<T>(grad * w01));
            common::dcu::AtomicAdd(&grad_input[base + y1 * width + x0], common::dcu::Cast<T>(grad * w10));
            common::dcu::AtomicAdd(&grad_input[base + y1 * width + x1], common::dcu::Cast<T>(grad * w11));
        }
    }
}
} // namespace

std::shared_ptr<Tensor> InterpolateForward(const std::shared_ptr<Tensor> &input,
                                           const std::vector<int64_t> &size, const std::string &mode,
                                           std::optional<bool> align_corners, bool antialias) {
    CHECK_EQ(input->Dims().size(), 4);
    CHECK_EQ(size.size(), 2);
    CHECK_EQ(mode, "bilinear");
    CHECK(!align_corners.value_or(false));
    CHECK(!antialias);
    auto output = std::make_shared<Tensor>(
        std::vector<int64_t>{input->Dims()[0], input->Dims()[1], size[0], size[1]}, input->Dtype(),
        input->GetDevice());
    const int64_t total = output->NumElements();
    const auto stream = GetStream(input->GetDevice());
    core::dcu::DispatchDcuFunc<INFINI_ALL_FLOATING_TYPES>(input->Dtype(), [=]<typename T>() {
        InterpolateBilinearForwardKernel<T><<<(total + 255) / 256, 256, 0, stream>>>(
            static_cast<const T *>(input->DataPtr()), static_cast<T *>(output->DataPtr()), input->Dims()[1],
            input->Dims()[2], input->Dims()[3], size[0], size[1], total);
    }, "HIP InterpolateForward");
    HIP_CHECK(hipGetLastError());
    return output;
}

std::shared_ptr<Tensor> InterpolateBackward(const std::shared_ptr<Tensor> &grad_output,
                                            const std::vector<int64_t> &input_dims,
                                            const std::string &mode,
                                            std::optional<bool> align_corners, bool antialias) {
    CHECK_EQ(mode, "bilinear");
    CHECK(!align_corners.value_or(false));
    CHECK(!antialias);
    auto grad_input = std::make_shared<Tensor>(input_dims, grad_output->Dtype(), grad_output->GetDevice());
    const auto stream = GetStream(grad_output->GetDevice());
    HIP_CHECK(hipMemsetAsync(grad_input->DataPtr(), 0, grad_input->SizeInBytes(), stream));
    const int64_t total = grad_output->NumElements();
    core::dcu::DispatchDcuFunc<INFINI_ALL_FLOATING_TYPES>(grad_output->Dtype(), [=]<typename T>() {
        InterpolateBilinearBackwardKernel<T><<<(total + 255) / 256, 256, 0, stream>>>(
            static_cast<const T *>(grad_output->DataPtr()), static_cast<T *>(grad_input->DataPtr()),
            input_dims[1], input_dims[2], input_dims[3], grad_output->Dims()[2], grad_output->Dims()[3], total);
    }, "HIP InterpolateBackward");
    HIP_CHECK(hipGetLastError());
    return grad_input;
}

std::shared_ptr<Tensor> RoiAlignForward(const std::shared_ptr<Tensor> &input,
                                        const std::shared_ptr<Tensor> &boxes,
                                        const std::vector<int64_t> &output_size, double spatial_scale,
                                        int64_t sampling_ratio, bool aligned) {
    CHECK_EQ(input->Dims().size(), 4);
    CHECK_EQ(boxes->Dims().size(), 2);
    CHECK_EQ(boxes->Dims()[1], 5);
    CHECK(boxes->Dtype() == DataType::kFLOAT32);
    CHECK(input->GetDevice() == boxes->GetDevice());
    auto output = std::make_shared<Tensor>(
        std::vector<int64_t>{boxes->Dims()[0], input->Dims()[1], output_size[0], output_size[1]},
        input->Dtype(), input->GetDevice());
    const int64_t total = output->NumElements();
    const auto stream = GetStream(input->GetDevice());
    core::dcu::DispatchDcuFunc<INFINI_ALL_FLOATING_TYPES>(input->Dtype(), [=]<typename T>() {
        RoiAlignForwardKernel<T><<<(total + 255) / 256, 256, 0, stream>>>(
            static_cast<const T *>(input->DataPtr()), static_cast<const float *>(boxes->DataPtr()),
            static_cast<T *>(output->DataPtr()), input->Dims()[1], input->Dims()[2], input->Dims()[3],
            output_size[0], output_size[1], static_cast<float>(spatial_scale), sampling_ratio, aligned, total);
    }, "HIP RoiAlignForward");
    HIP_CHECK(hipGetLastError());
    return output;
}

std::shared_ptr<Tensor> RoiAlignBackward(const std::shared_ptr<Tensor> &grad_output,
                                         const std::shared_ptr<Tensor> &boxes,
                                         const std::vector<int64_t> &input_dims,
                                         const std::vector<int64_t> &output_size, double spatial_scale,
                                         int64_t sampling_ratio, bool aligned) {
    auto grad_input = std::make_shared<Tensor>(input_dims, grad_output->Dtype(), grad_output->GetDevice());
    const auto stream = GetStream(grad_output->GetDevice());
    HIP_CHECK(hipMemsetAsync(grad_input->DataPtr(), 0, grad_input->SizeInBytes(), stream));
    const int64_t total = grad_output->NumElements();
    core::dcu::DispatchDcuFunc<INFINI_ALL_FLOATING_TYPES>(grad_output->Dtype(), [=]<typename T>() {
        RoiAlignBackwardKernel<T><<<(total + 255) / 256, 256, 0, stream>>>(
            static_cast<const T *>(grad_output->DataPtr()), static_cast<const float *>(boxes->DataPtr()),
            static_cast<T *>(grad_input->DataPtr()), input_dims[1], input_dims[2], input_dims[3],
            output_size[0], output_size[1], static_cast<float>(spatial_scale), sampling_ratio, aligned, total);
    }, "HIP RoiAlignBackward");
    HIP_CHECK(hipGetLastError());
    return grad_input;
}

} // namespace infini_train::kernels::dcu

#define REGISTER_HIP_RESAMPLING_KERNEL(name)                                                                    \
    REGISTER_KERNEL(infini_train::Device::DeviceType::kDCU, name, infini_train::kernels::dcu::name)

REGISTER_HIP_RESAMPLING_KERNEL(InterpolateForward)
REGISTER_HIP_RESAMPLING_KERNEL(InterpolateBackward)
REGISTER_HIP_RESAMPLING_KERNEL(RoiAlignForward)
REGISTER_HIP_RESAMPLING_KERNEL(RoiAlignBackward)

#undef REGISTER_HIP_RESAMPLING_KERNEL
