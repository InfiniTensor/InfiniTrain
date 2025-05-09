#include <cmath>
#include <functional>
#include <memory>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> SliceForward(const std::shared_ptr<Tensor> &input, const std::vector<int64_t> &starts,
                                     const std::vector<int64_t> &ends, const std::vector<int64_t> &steps) {
    CHECK_EQ(starts.size(), ends.size());
    CHECK_EQ(starts.size(), steps.size());
    auto &dims = input->Dims();
    CHECK_EQ(starts.size(), dims.size());

    std::vector<int64_t> new_dims;
    for (int i = 0; i < starts.size(); i++) {
        CHECK_LE(starts[i], ends[i]);
        CHECK_LE(0, steps[i]);
        new_dims.push_back((ends[i] - starts[i] + steps[i] - 1) / steps[i]);
    }

    auto new_tensor = std::make_shared<Tensor>(new_dims, input->Dtype(), input->GetDevice());

    std::vector<int64_t> src_strides(dims.size());
    int64_t stride = 1;
    for (int i = src_strides.size() - 1; i >= 0; --i) {
        src_strides[i] = stride;
        stride *= dims[i];
    }

    std::vector<int64_t> dst_strides(new_dims.size());
    stride = 1;
    for (int i = dst_strides.size() - 1; i >= 0; --i) {
        dst_strides[i] = stride;
        stride *= new_dims[i];
    }

    std::vector<int64_t> dst_index(new_dims.size(), 0);
    std::vector<int64_t> src_index(dims.size(), 0);

    std::function<void(int)> recurse = [&](int d) {
        if (d == dims.size()) {
            int64_t src_offset = 0;
            int64_t dst_offset = 0;
            for (int i = dims.size() - 1; i >= 0; --i) {
                src_offset += src_index[i] * src_strides[i];
                dst_offset += dst_index[i] * dst_strides[i];
            }
            static_cast<float *>(new_tensor->DataPtr())[dst_offset]
                = static_cast<const float *>(input->DataPtr())[src_offset];
            return;
        }

        // fill in the src_index and dst_index
        int64_t out_i = 0;
        for (int64_t i = starts[d]; i < ends[d]; i += steps[d]) {
            src_index[d] = i;
            dst_index[d] = out_i++;
            recurse(d + 1);
        }
    };

    recurse(0);
    return new_tensor;
}

std::shared_ptr<Tensor> SliceBackward(const std::shared_ptr<Tensor> &grad_output, const std::shared_ptr<Tensor> &input,
                                      const std::vector<int64_t> &starts, const std::vector<int64_t> &ends,
                                      const std::vector<int64_t> &steps) {
    CHECK_EQ(starts.size(), ends.size());
    CHECK_EQ(starts.size(), steps.size());
    auto &dims = input->Dims();
    CHECK_EQ(starts.size(), dims.size());

    std::vector<int64_t> new_dims;
    for (int i = 0; i < starts.size(); i++) {
        CHECK_LE(starts[i], ends[i]);
        CHECK_LE(0, steps[i]);
        new_dims.push_back((ends[i] - starts[i] + steps[i] - 1) / steps[i]);
    }

    auto new_tensor = std::make_shared<Tensor>(input->Dims(), input->Dtype(), input->GetDevice());
    new_tensor->Fill<float>(0.0);

    std::vector<int64_t> src_strides(dims.size());
    int64_t stride = 1;
    for (int i = src_strides.size() - 1; i >= 0; --i) {
        src_strides[i] = stride;
        stride *= dims[i];
    }

    std::vector<int64_t> dst_strides(new_dims.size());
    stride = 1;
    for (int i = dst_strides.size() - 1; i >= 0; --i) {
        dst_strides[i] = stride;
        stride *= new_dims[i];
    }

    std::vector<int64_t> dst_index(new_dims.size(), 0);
    std::vector<int64_t> src_index(dims.size(), 0);

    std::function<void(int)> recurse = [&](int d) {
        if (d == dims.size()) {
            int64_t src_offset = 0;
            int64_t dst_offset = 0;
            for (int i = dims.size() - 1; i >= 0; --i) {
                src_offset += src_index[i] * src_strides[i];
                dst_offset += dst_index[i] * dst_strides[i];
            }
            static_cast<float *>(new_tensor->DataPtr())[src_offset]
                = static_cast<const float *>(grad_output->DataPtr())[dst_offset];
            return;
        }

        // fill in the src_index and dst_index
        int64_t out_i = 0;
        for (int64_t i = starts[d]; i < ends[d]; i += steps[d]) {
            src_index[d] = i;
            dst_index[d] = out_i++;
            recurse(d + 1);
        }
    };

    recurse(0);
    return new_tensor;
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_SLICE_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_SLICE_KERNEL(SliceForward)
REGISTER_CPU_SLICE_KERNEL(SliceBackward)

#undef REGISTER_CPU_SLICE_KERNEL
