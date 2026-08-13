#include "infini_train/include/nn/init.h"

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <unordered_set>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/generator.h"
#include "infini_train/include/tensor.h"
#include "infini_train/src/generator_internal.h"
#include "infini_train/src/random_utils.h"

#ifdef USE_CUDA
namespace infini_train::kernels::cuda {
void RandomUniformFloat32(void *data, int64_t num_elements, float from, float to, uint64_t seed, uint64_t offset,
                          Device device);
void RandomNormalFloat32(void *data, int64_t num_elements, float mean, float stddev, uint64_t seed, uint64_t offset,
                         Device device);
} // namespace infini_train::kernels::cuda
#endif

namespace infini_train::nn::init {

namespace {
int64_t CheckedMultiplyFanFactors(int64_t lhs, int64_t rhs) {
    CHECK_GE(lhs, 0) << "Fan calculation requires non-negative tensor dimensions";
    CHECK_GE(rhs, 0) << "Fan calculation requires non-negative tensor dimensions";
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    CHECK_LE(lhs, std::numeric_limits<int64_t>::max() / rhs) << "Fan calculation overflow";
    return lhs * rhs;
}
} // namespace

std::shared_ptr<Tensor> Normal(const std::shared_ptr<Tensor> &tensor, float mean, float stddev,
                               std::shared_ptr<Generator> generator) {
    CHECK_GE(stddev, 0.0f);
    CHECK(tensor->Dtype() == DataType::kFLOAT32) << "Random normal currently supports float32 tensors";
    const int64_t num_elements = tensor->NumElements();

    auto device = tensor->GetDevice();
    auto resolved_generator = generator ? generator : GetDefaultGenerator(device);
    CHECK(resolved_generator->GetDevice().type() == device.type())
        << "Generator backend must match tensor device backend: generator=" << resolved_generator->GetDevice()
        << " tensor=" << device;
    if (num_elements == 0) {
        return tensor;
    }

#ifdef USE_CUDA
    if (device.IsCUDA()) {
        CHECK_LE(num_elements, static_cast<int64_t>(std::numeric_limits<uint64_t>::max() / 2))
            << "Random normal tensor is too large";
        const auto [seed, offset] = detail::GeneratorAccessor::ReserveCUDARandomOffset(
            resolved_generator, static_cast<uint64_t>(num_elements) * 2);
        core::DeviceGuard guard(device);
        kernels::cuda::RandomNormalFloat32(tensor->DataPtr(), num_elements, mean, stddev, seed, offset, device);
        return tensor;
    }
#endif

    CHECK(device.IsCPU()) << "Random normal backend is not available for device " << device;
    detail::GeneratorAccessor::FillCPUNormal(resolved_generator, static_cast<float *>(tensor->DataPtr()),
                                             tensor->NumElements(), mean, stddev);
    return tensor;
}

std::pair<int64_t, int64_t> CalculateFanInAndFanOut(const std::shared_ptr<Tensor> &tensor) {
    if (tensor->Dims().size() < 2) {
        LOG(FATAL) << "Fan in and fan out can not be computed for tensor with less than 2 dimensions";
    }
    const auto num_input_fmaps = tensor->Dims()[1];
    const auto num_output_fmaps = tensor->Dims()[0];
    const int64_t receptive_field_size
        = std::accumulate(tensor->Dims().begin() + 2, tensor->Dims().end(), int64_t{1}, CheckedMultiplyFanFactors);
    const int64_t fan_in = CheckedMultiplyFanFactors(num_input_fmaps, receptive_field_size);
    const int64_t fan_out = CheckedMultiplyFanFactors(num_output_fmaps, receptive_field_size);
    return {fan_in, fan_out};
}

namespace {
int64_t CalculateCorrectFan(const std::shared_ptr<Tensor> &tensor, KaimingMode mode) {
    const auto [fan_in, fan_out] = CalculateFanInAndFanOut(tensor);
    return mode == KaimingMode::kFanIn ? fan_in : fan_out;
}

// TODO(dcj): Support templated param later.
float CalculateGain(NonLinearityType nonlinearity, std::optional<float> param = std::nullopt) {
    static std::unordered_set<NonLinearityType> kLinearFns = {
        NonLinearityType::kLinear,           NonLinearityType::kConv1D,           NonLinearityType::kConv2D,
        NonLinearityType::kConv3D,           NonLinearityType::kConvTransposed1d, NonLinearityType::kConvTransposed2d,
        NonLinearityType::kConvTransposed3d,
    };
    if (kLinearFns.contains(nonlinearity) || nonlinearity == NonLinearityType::kSigmoid) {
        return 1.0f;
    } else if (nonlinearity == NonLinearityType::kTanh) {
        return 5.0f / 3;
    } else if (nonlinearity == NonLinearityType::kReLU) {
        return sqrt(2.0f);
    } else if (nonlinearity == NonLinearityType::kLeakyReLU) {
        const float negative_slope = param ? *param : 0.01f;
        return sqrt(2.0f / (1 + negative_slope * negative_slope));
    } else if (nonlinearity == NonLinearityType::kSELU) {
        return 3.0f / 4; // Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
    } else {
        LOG(FATAL) << "Unsupported non-linearity type: " << static_cast<int>(nonlinearity);
    }
    return -1.0f;
}
} // namespace

std::shared_ptr<Tensor> KaimingUniform(const std::shared_ptr<Tensor> &tensor, float a, KaimingMode mode,
                                       NonLinearityType nonlinearity, std::shared_ptr<Generator> generator) {
    for (const auto dim : tensor->Dims()) {
        if (dim == 0) {
            LOG(WARNING) << "Initializing zero-element tensors is a no-op";
            return tensor;
        }
    }
    const auto fan = CalculateCorrectFan(tensor, mode);
    const auto gain = CalculateGain(nonlinearity, a);
    const float stddev = gain / sqrt(fan);
    const float bound = sqrt(3.0f) * stddev; // Calculate uniform bounds from standard deviation
    return tensor->Uniform(-bound, bound, generator);
}

std::shared_ptr<Tensor> Uniform(const std::shared_ptr<Tensor> &tensor, float a, float b,
                                std::shared_ptr<Generator> generator) {
    detail::CheckUniformBounds(a, b);
    CHECK(tensor->Dtype() == DataType::kFLOAT32) << "Random uniform currently supports float32 tensors";
    const int64_t num_elements = tensor->NumElements();

    auto device = tensor->GetDevice();
    auto resolved_generator = generator ? generator : GetDefaultGenerator(device);
    CHECK(resolved_generator->GetDevice().type() == device.type())
        << "Generator backend must match tensor device backend: generator=" << resolved_generator->GetDevice()
        << " tensor=" << device;
    if (num_elements == 0) {
        return tensor;
    }

#ifdef USE_CUDA
    if (device.IsCUDA()) {
        const auto [seed, offset] = detail::GeneratorAccessor::ReserveCUDARandomOffset(
            resolved_generator, static_cast<uint64_t>(num_elements));
        core::DeviceGuard guard(device);
        kernels::cuda::RandomUniformFloat32(tensor->DataPtr(), num_elements, a, b, seed, offset, device);
        return tensor;
    }
#endif

    CHECK(device.IsCPU()) << "Random uniform backend is not available for device " << device;
    detail::GeneratorAccessor::FillCPUUniform(resolved_generator, static_cast<float *>(tensor->DataPtr()),
                                              tensor->NumElements(), a, b);

    return tensor;
}

std::shared_ptr<Tensor> Ones(const std::shared_ptr<Tensor> &tensor) {
    // TODO(dcj): Support other data types later.
    CHECK_EQ(static_cast<int>(tensor->Dtype()), static_cast<int>(DataType::kFLOAT32));
    const int64_t num_elements = tensor->NumElements();
    std::vector<float> buffer(num_elements, 1.0f);

    auto device = tensor->GetDevice();
    core::DeviceGuard guard(device);

    auto impl = core::GetDeviceGuardImpl(device.type());

    impl->MemcpyAsync(tensor->DataPtr(), buffer.data(), num_elements * sizeof(float),
                      device.type() == Device::DeviceType::kCPU ? core::MemcpyKind::kD2D : core::MemcpyKind::kH2D,
                      impl->GetStream(device));

    return tensor;
}

std::shared_ptr<Tensor> Zeros(const std::shared_ptr<Tensor> &tensor) {
    // TODO(dcj): Support other data types later.
    CHECK_EQ(static_cast<int>(tensor->Dtype()), static_cast<int>(DataType::kFLOAT32));
    const int64_t num_elements = tensor->NumElements();
    std::vector<float> buffer(num_elements, 0.0f);

    auto device = tensor->GetDevice();
    core::DeviceGuard guard(device);

    auto impl = core::GetDeviceGuardImpl(device.type());

    impl->MemcpyAsync(tensor->DataPtr(), buffer.data(), num_elements * sizeof(float),
                      device.type() == Device::DeviceType::kCPU ? core::MemcpyKind::kD2D : core::MemcpyKind::kH2D,
                      impl->GetStream(device));

    return tensor;
}

#define ARANGE_CASE(DATA_TYPE, TYPE)                                                                                   \
    case DATA_TYPE: {                                                                                                  \
        std::vector<TYPE> buffer(num_elements);                                                                        \
        std::iota(buffer.begin(), buffer.end(), static_cast<TYPE>(start));                                             \
        impl->MemcpyAsync(tensor->DataPtr(), buffer.data(), num_elements * sizeof(TYPE), kind, stream);                \
        break;                                                                                                         \
    }

std::shared_ptr<Tensor> Arange(int64_t start, int64_t end, DataType dtype, Device device) {
    const int64_t num_elements = end - start;
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{num_elements}, dtype, device);

    core::DeviceGuard guard(device);
    auto *impl = core::GetDeviceGuardImpl(device.type());

    const core::MemcpyKind kind = device.IsCPU() ? core::MemcpyKind::kD2D : core::MemcpyKind::kH2D;
    core::Stream *stream = impl->GetStream(device);

    switch (dtype) {
        ARANGE_CASE(DataType::kUINT8, uint8_t)
        ARANGE_CASE(DataType::kINT8, int8_t)
        ARANGE_CASE(DataType::kUINT16, uint16_t)
        ARANGE_CASE(DataType::kINT16, int16_t)
        ARANGE_CASE(DataType::kUINT32, uint32_t)
        ARANGE_CASE(DataType::kINT32, int32_t)
        ARANGE_CASE(DataType::kUINT64, uint64_t)
        ARANGE_CASE(DataType::kINT64, int64_t)
        ARANGE_CASE(DataType::kBFLOAT16, BF16)
        ARANGE_CASE(DataType::kFLOAT16, FP16)
        ARANGE_CASE(DataType::kFLOAT32, float)
        ARANGE_CASE(DataType::kFLOAT64, double)

    default:
        LOG(FATAL) << "Unsupported data type: " << static_cast<int>(dtype);
        break;
    }

    return tensor;
}

#undef ARANGE_CASE
} // namespace infini_train::nn::init
