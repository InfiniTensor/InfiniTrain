#include "infini_train/include/nn/init.h"

#include <limits>
#include <memory>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/datatype.h"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::init {
namespace {

struct DistributionBounds {
    double lowest;
    double max;
};

DistributionBounds GetDistributionBounds(DataType dtype) {
    switch (dtype) {
    case DataType::kFLOAT16: {
        const double max = static_cast<float>(FP16(static_cast<uint16_t>(0x7bff), FP16::from_bits()));
        return {-max, max};
    }
    case DataType::kBFLOAT16: {
        const double max = static_cast<float>(BF16(static_cast<uint16_t>(0x7f7f), BF16::from_bits()));
        return {-max, max};
    }
    case DataType::kFLOAT32:
        return {-std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    case DataType::kFLOAT64:
        return {-std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    default:
        LOG(FATAL) << "Unsupported distribution dtype: " << kDataTypeToDesc.at(dtype);
        return {};
    }
}

void CheckDistributionTensor(const Tensor &tensor) {
    CHECK(IsFloatingPointDType(tensor.Dtype()))
        << "Uniform and Normal initialization support floating-point tensors only";
}

void CheckUniformParameters(const Tensor &tensor, double from, double to) {
    const auto bounds = GetDistributionBounds(tensor.Dtype());
    CHECK_GE(from, bounds.lowest) << "uniform expects from to be within the range of "
                                  << kDataTypeToDesc.at(tensor.Dtype());
    CHECK_LE(from, bounds.max) << "uniform expects from to be within the range of "
                               << kDataTypeToDesc.at(tensor.Dtype());
    CHECK_GE(to, bounds.lowest) << "uniform expects to to be within the range of "
                                << kDataTypeToDesc.at(tensor.Dtype());
    CHECK_LE(to, bounds.max) << "uniform expects to to be within the range of " << kDataTypeToDesc.at(tensor.Dtype());
    CHECK_LE(from, to) << "uniform expects a [from, to) range, but found from=" << from << " > to=" << to;
    CHECK_LE(to - from, bounds.max) << "uniform expects to - from to fit in " << kDataTypeToDesc.at(tensor.Dtype());
}

void CheckNormalParameters(double std) { CHECK_GE(std, 0.0) << "normal expects std >= 0.0, but found std=" << std; }

} // namespace

std::shared_ptr<Tensor> Normal(const std::shared_ptr<Tensor> &tensor, float mean, float std,
                               std::optional<Generator> generator) {
    CheckDistributionTensor(*tensor);
    CheckNormalParameters(std);
    auto device = tensor->GetDevice();
    core::DeviceGuard guard(device);
    Dispatcher::Instance().Call<void>({device.type(), "Normal"}, tensor, static_cast<double>(mean),
                                      static_cast<double>(std), generator);
    return tensor;
}

std::pair<int64_t, int64_t> CalculateFanInAndFanOut(const std::shared_ptr<Tensor> &tensor) {
    if (tensor->Dims().size() < 2) {
        LOG(FATAL) << "Fan in and fan out can not be computed for tensor with less than 2 dimensions";
    }
    const auto num_input_fmaps = tensor->Dims()[1];
    const auto num_output_fmaps = tensor->Dims()[0];
    int64_t receptive_field_size = 1;
    if (tensor->Dims().size() > 2) {
        receptive_field_size
            *= std::accumulate(tensor->Dims().begin() + 2, tensor->Dims().end(), 1, std::multiplies<int64_t>());
    }
    const auto fan_in = num_input_fmaps * receptive_field_size;
    const auto fan_out = num_output_fmaps * receptive_field_size;
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
                                       NonLinearityType nonlinearity, std::optional<Generator> generator) {
    for (const auto dim : tensor->Dims()) {
        if (dim == 0) {
            LOG(WARNING) << "Initializing zero-element tensors is a no-op";
            return tensor;
        }
    }
    const auto fan = CalculateCorrectFan(tensor, mode);
    const auto gain = CalculateGain(nonlinearity, a);
    const float std = gain / sqrt(fan);
    const float bound = sqrt(3.0f) * std; // Calculate uniform bounds from standard deviation
    return tensor->Uniform(-bound, bound, generator);
}

std::shared_ptr<Tensor> Uniform(const std::shared_ptr<Tensor> &tensor, float a, float b,
                                std::optional<Generator> generator) {
    CheckDistributionTensor(*tensor);
    CheckUniformParameters(*tensor, a, b);
    auto device = tensor->GetDevice();
    core::DeviceGuard guard(device);
    Dispatcher::Instance().Call<void>({device.type(), "Uniform"}, tensor, static_cast<double>(a),
                                      static_cast<double>(b), generator);
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
