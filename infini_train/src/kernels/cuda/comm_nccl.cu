#ifdef USE_NCCL

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cuda_runtime.h"
#include "glog/logging.h"
#include "nccl.h"

#include "infini_train/include/common/cuda/common_cuda.cuh"
#include "infini_train/include/device.h"
#include "infini_train/include/dispatcher.h"
#include "infini_train/include/nn/functional.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

namespace {
const std::unordered_map<DataType, ncclDataType_t> kNcclDtypeMap = {
    {DataType::kUINT8, ncclUint8},       {DataType::kINT8, ncclInt8},     {DataType::kUINT32, ncclUint32},
    {DataType::kINT32, ncclInt32},       {DataType::kUINT64, ncclUint64}, {DataType::kINT64, ncclInt64},
    {DataType::kBFLOAT16, ncclBfloat16}, {DataType::kFLOAT16, ncclHalf},  {DataType::kFLOAT32, ncclFloat32},
    {DataType::kFLOAT64, ncclFloat64},
};
} // namespace

std::vector<std::shared_ptr<Tensor>> NcclBroadcast(const std::vector<std::shared_ptr<Tensor>> &input_tensors,
                                                   const std::vector<const Device *> &devices) {
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::vector<cudaStream_t> streams;
    std::vector<ncclComm_t> comms;

    for (size_t i = 0; i < devices.size(); ++i) {
        for (const auto &input_tensor : input_tensors) {
            outputs.push_back(std::make_shared<Tensor>(input_tensor->Dims(), input_tensor->Dtype(), devices[i]));
        }
        streams.push_back(dynamic_cast<const CudaDevice *>(devices[i])->Stream());
        comms.push_back(dynamic_cast<const CudaDevice *>(devices[i])->NcclComm());
    }

    int root = -1;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (devices[i] == input_tensors[0]->GetDevice()) {
            root = i;
            break;
        }
    }
    CHECK_NE(root, -1) << "Root not found in input devices";

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < devices.size(); ++i) {
        for (size_t j = 0; j < input_tensors.size(); ++j) {
            const auto &input_tensor = input_tensors[j];
            const auto dtype = input_tensor->Dtype();
            auto nccl_dtype = kNcclDtypeMap.at(dtype);
            auto count = input_tensor->NumElements();
            void *send_buffer = (devices[i] == input_tensor->GetDevice() ? input_tensor->DataPtr() : nullptr);
            NCCL_CHECK(ncclBroadcast(send_buffer, outputs[i * input_tensors.size() + j]->DataPtr(), count, nccl_dtype,
                                     0, comms[i], streams[i]));
        }
    }
    NCCL_CHECK(ncclGroupEnd());

    return outputs;
}

std::vector<std::shared_ptr<Tensor>>
NcclReduceAddCoalesced(const std::vector<std::vector<std::shared_ptr<Tensor>>> &grads, const Device *destination) {
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::vector<cudaStream_t> streams;
    std::vector<ncclComm_t> comms;

    for (size_t i = 0; i < grads[0].size(); ++i) {
        outputs.push_back(std::make_shared<Tensor>(grads[0][i]->Dims(), grads[0][i]->Dtype(), destination));
        outputs[i]->Fill<float>(0.0f);
    }
    for (size_t i = 0; i < grads.size(); ++i) {
        streams.push_back(dynamic_cast<const CudaDevice *>(grads[i][0]->GetDevice())->Stream());
        comms.push_back(dynamic_cast<const CudaDevice *>(grads[i][0]->GetDevice())->NcclComm());
    }

    int root = -1;
    for (size_t i = 0; i < grads.size(); ++i) {
        if (grads[i][0]->GetDevice() == destination) {
            root = i;
            break;
        }
    }
    CHECK_NE(root, -1) << "Destination device not found in grads group";

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < grads.size(); ++i) {
        for (size_t j = 0; j < grads[i].size(); ++j) {
            const auto &grad = grads[i][j];
            const auto dtype = grad->Dtype();
            auto nccl_dtype = kNcclDtypeMap.at(dtype);
            auto count = grad->NumElements();
            void *send_buffer = grad->DataPtr();
            NCCL_CHECK(
                ncclReduce(send_buffer, outputs[j]->DataPtr(), count, nccl_dtype, ncclSum, 0, comms[i], streams[i]));
        }
    }
    NCCL_CHECK(ncclGroupEnd());

    return outputs;
}

std::vector<std::shared_ptr<Tensor>> NcclScatter(const std::shared_ptr<Tensor> &tensor,
                                                 std::vector<const Device *> devices, int64_t dim) {
    std::vector<std::shared_ptr<Tensor>> outputs;
    std::vector<std::shared_ptr<Tensor>> split_tensors = tensor->Split(tensor->Dims()[dim] / devices.size(), dim);
    std::vector<cudaStream_t> streams;
    std::vector<ncclComm_t> comms;
    int src_rank = -1;
    for (size_t i = 0; i < devices.size(); ++i) {
        if (tensor->GetDevice() == devices[i]) {
            src_rank = i;
        }
        outputs.push_back(std::make_shared<Tensor>(split_tensors[i]->Dims(), split_tensors[i]->Dtype(), devices[i]));
        streams.push_back(dynamic_cast<const CudaDevice *>(devices[i])->Stream());
        comms.push_back(dynamic_cast<const CudaDevice *>(devices[i])->NcclComm());
    }

    CHECK_NE(src_rank, -1) << "Source device not found in input devices";

    NCCL_CHECK(ncclGroupStart());
    const auto dtype = tensor->Dtype();
    auto nccl_dtype = kNcclDtypeMap.at(dtype);

    for (size_t i = 0; i < devices.size(); ++i) {
        const auto dtype = tensor->Dtype();
        auto nccl_dtype = kNcclDtypeMap.at(dtype);
        NCCL_CHECK(ncclSend(split_tensors[i]->DataPtr(), split_tensors[i]->NumElements(), nccl_dtype, i,
                            comms[src_rank], streams[src_rank]));
        NCCL_CHECK(
            ncclRecv(outputs[i]->DataPtr(), outputs[i]->NumElements(), nccl_dtype, src_rank, comms[i], streams[i]));
    }
    NCCL_CHECK(ncclGroupEnd());
    return outputs;
}

std::shared_ptr<Tensor> NcclGather(const std::vector<std::shared_ptr<Tensor>> &tensors, const Device *destination,
                                   int64_t dim) {
    std::vector<std::shared_ptr<Tensor>> outouts;
    int64_t num_devices = tensors.size();
    auto dtype = tensors[0]->Dtype();
    auto nccl_dtype = kNcclDtypeMap.at(dtype);

    int64_t total_dim = 0;

    std::vector<cudaStream_t> streams;
    std::vector<ncclComm_t> comms;

    int dest_rank = -1;
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto device = tensors[i]->GetDevice();
        if (device == destination) {
            dest_rank = i;
        }
        streams.push_back(dynamic_cast<const CudaDevice *>(device)->Stream());
        comms.push_back(dynamic_cast<const CudaDevice *>(device)->NcclComm());

        total_dim += tensors[i]->Dims()[dim];
    }

    std::vector<int64_t> out_dims = tensors[0]->Dims();
    out_dims[dim] = total_dim;
    auto output = std::make_shared<Tensor>(out_dims, dtype, destination);

    CHECK_NE(dest_rank, -1) << "Destination device not found in input tensors's devices";

    NCCL_CHECK(ncclGroupStart());
    int64_t offset = 0;

    for (size_t i = 0; i < num_devices; ++i) {
        auto &tensor = tensors[i];
        size_t num_elements = tensor->NumElements();
        void *send_ptr = tensor->DataPtr();

        auto recv_ptr = static_cast<int8_t *>(output->DataPtr()) + offset;

        NCCL_CHECK(ncclSend(send_ptr, num_elements, nccl_dtype, dest_rank, comms[i], streams[i]));
        NCCL_CHECK(ncclRecv(recv_ptr, num_elements, nccl_dtype, i, comms[dest_rank], streams[dest_rank]));

        offset += tensor->SizeInBytes();
    }

    NCCL_CHECK(ncclGroupEnd());
    return output;
}

void NcclAllReduce(const std::vector<std::vector<std::shared_ptr<Tensor>>> &tensors) {
    // tensors: [num_devices][num_tensors_per_device]

    std::vector<cudaStream_t> streams;
    std::vector<ncclComm_t> comms;

    for (size_t i = 0; i < tensors.size(); ++i) {
        auto device_ptr = dynamic_cast<const CudaDevice *>(tensors[i][0]->GetDevice());
        streams.push_back(device_ptr->Stream());
        comms.push_back(device_ptr->NcclComm());
    }

    NCCL_CHECK(ncclGroupStart());

    for (size_t i = 0; i < tensors.size(); ++i) {
        for (size_t j = 0; j < tensors[i].size(); ++j) {
            const auto &tensor = tensors[i][j];
            auto dtype = tensor->Dtype();
            auto nccl_dtype = kNcclDtypeMap.at(dtype);
            auto count = tensor->NumElements();
            void *buffer = tensor->DataPtr();

            NCCL_CHECK(ncclAllReduce(buffer, buffer, count, nccl_dtype, ncclSum, comms[i], streams[i]));
        }
    }

    NCCL_CHECK(ncclGroupEnd());
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_COMM_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, Comm##kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_COMM_KERNEL(NcclBroadcast)
REGISTER_CUDA_COMM_KERNEL(NcclScatter)
REGISTER_CUDA_COMM_KERNEL(NcclGather)
REGISTER_CUDA_COMM_KERNEL(NcclReduceAddCoalesced)
REGISTER_CUDA_COMM_KERNEL(NcclAllReduce)

#undef REGISTER_CUDA_COMM_KERNEL

#endif // USE_NCCL
