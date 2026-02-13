#include "infini_train/include/nn/init.h"

#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <unordered_set>

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif
#ifdef USE_OMP
#include <omp.h>
#endif

#include "glog/logging.h"

#include "infini_train/include/core/device_guard.h"
#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn::init {
namespace {
static std::random_device rd;
static std::mt19937 gen(rd());
} // namespace

std::shared_ptr<Tensor> Normal(const std::shared_ptr<Tensor> &tensor, float mean, float std,
                               std::optional<std::mt19937> generator) {
    const int64_t num_elements = tensor->NumElements();
    std::vector<float> buffer(num_elements);

#ifdef USE_OMP
#pragma omp parallel
    {
        std::mt19937 local_gen(std::random_device{}() + omp_get_thread_num());
        std::normal_distribution<float> local_dis(mean, std);
#pragma omp for
        for (int i = 0; i < buffer.size(); ++i) {
            buffer[i] = generator ? local_dis(generator.value()) : local_dis(local_gen);
        }
    }
#else
    std::normal_distribution<float> dis(mean, std);
    std::generate(buffer.begin(), buffer.end(), [&]() { return generator ? dis(generator.value()) : dis(gen); });
#endif

    auto device = tensor->GetDevice();
    core::DeviceGuard guard(device);
    auto impl = core::GetDeviceGuardImpl(device.type());

    impl->Memcpy(tensor->DataPtr(), buffer.data(), num_elements * sizeof(float),
                 device.type() == Device::DeviceType::kCPU ? core::MemcpyKind::kD2D : core::MemcpyKind::kH2D);
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
                                       NonLinearityType nonlinearity, std::optional<std::mt19937> generator) {
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
                                std::optional<std::mt19937> generator) {
    const int64_t num_elements = tensor->NumElements();
    std::vector<float> buffer(num_elements);

#ifdef USE_OMP
#pragma omp parallel
    {
        std::mt19937 local_gen(std::random_device{}() + omp_get_thread_num());
        std::uniform_real_distribution<float> local_dis(a, b);
#pragma omp for
        for (int i = 0; i < buffer.size(); ++i) {
            buffer[i] = generator ? local_dis(generator.value()) : local_dis(local_gen);
        }
    }
#else
    std::uniform_real_distribution<float> dis(a, b);
    std::generate(buffer.begin(), buffer.end(), [&]() { return generator ? dis(generator.value()) : dis(gen); });
#endif

    auto device = tensor->GetDevice();

    core::DeviceGuard guard(device);
    auto impl = core::GetDeviceGuardImpl(device.type());

    impl->Memcpy(tensor->DataPtr(), buffer.data(), num_elements * sizeof(float),
                 device.type() == Device::DeviceType::kCPU ? core::MemcpyKind::kD2D : core::MemcpyKind::kH2D);

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

    impl->Memcpy(tensor->DataPtr(), buffer.data(), num_elements * sizeof(float),
                 device.type() == Device::DeviceType::kCPU ? core::MemcpyKind::kD2D : core::MemcpyKind::kH2D);

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

    impl->Memcpy(tensor->DataPtr(), buffer.data(), num_elements * sizeof(float),
                 device.type() == Device::DeviceType::kCPU ? core::MemcpyKind::kD2D : core::MemcpyKind::kH2D);

    return tensor;
}

#define ARANGE_CASE(DATA_TYPE, TYPE)                                                                                   \
    case DATA_TYPE: {                                                                                                  \
        std::vector<TYPE> buffer(num_elements);                                                                        \
        std::iota(buffer.begin(), buffer.end(), static_cast<TYPE>(start));                                             \
        impl->Memcpy(tensor->DataPtr(), buffer.data(), num_elements * sizeof(TYPE), kind);                             \
        break;                                                                                                         \
    }

std::shared_ptr<Tensor> Arange(int64_t start, int64_t end, DataType dtype, Device device) {
    const int64_t num_elements = end - start;
    auto tensor = std::make_shared<Tensor>(std::vector<int64_t>{num_elements}, dtype, device);

    core::DeviceGuard guard(device);
    auto *impl = core::GetDeviceGuardImpl(device.type());

    const core::MemcpyKind kind = device.IsCPU() ? core::MemcpyKind::kD2D : core::MemcpyKind::kH2D;

    switch (dtype) {
        ARANGE_CASE(DataType::kUINT8, uint8_t)
        ARANGE_CASE(DataType::kINT8, int8_t)
        ARANGE_CASE(DataType::kUINT16, uint16_t)
        ARANGE_CASE(DataType::kINT16, int16_t)
        ARANGE_CASE(DataType::kUINT32, uint32_t)
        ARANGE_CASE(DataType::kINT32, int32_t)
        ARANGE_CASE(DataType::kUINT64, uint64_t)
        ARANGE_CASE(DataType::kINT64, int64_t)

#ifdef USE_CUDA
        ARANGE_CASE(DataType::kBFLOAT16, nv_bfloat16)
        ARANGE_CASE(DataType::kFLOAT16, half)
#endif

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
