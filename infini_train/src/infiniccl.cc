#include "infini_train/include/infiniccl.h"

#include "glog/logging.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Platform detection macros
#if defined(USE_CUDA)
    #include <cuda_runtime.h>
    #define GPU_ERROR_T          cudaError_t
    #define GPU_SUCCESS          cudaSuccess
    #define GPU_GET_ERROR_STR    cudaGetErrorString
    #define GPU_MALLOC(ptr, sz)  cudaMalloc((void**)&(ptr), (sz))
    #define GPU_FREE(ptr)        cudaFree(ptr)
    #define GPU_MEMCPY_D2H(d,s,n) cudaMemcpy(d, s, n, cudaMemcpyDeviceToHost)
    #define GPU_MEMCPY_H2D(d,s,n) cudaMemcpy(d, s, n, cudaMemcpyHostToDevice)
    #define GPU_SYNC()           cudaDeviceSynchronize()
    #define GPU_GET_DEVICE(id)   cudaGetDevice(id)
    #define GPU_SET_DEVICE(id)   cudaSetDevice(id)
    #define GPU_PLATFORM "NVIDIA"    
#elif defined(USE_MACA)
    #include <mcr/mc_runtime_api.h>
    #define GPU_ERROR_T          mcError_t
    #define GPU_SUCCESS          mcSuccess
    #define GPU_GET_ERROR_STR    mcGetErrorString
    #define GPU_MALLOC(ptr, sz)  mcMalloc((void**)&(ptr), (sz))
    #define GPU_FREE(ptr)        mcFree(ptr)
    #define GPU_MEMCPY_D2H(d,s,n) mcMemcpy(d, s, n, mcMemcpyDeviceToHost)
    #define GPU_MEMCPY_H2D(d,s,n) mcMemcpy(d, s, n, mcMemcpyHostToDevice)
    #define GPU_SYNC()           mcDeviceSynchronize()
    #define GPU_GET_DEVICE(id)   mcGetDevice(id)
    #define GPU_SET_DEVICE(id)   mcSetDevice(id)
    #define GPU_PLATFORM "MetaX"    
#endif

// GPU error checking
#define GPU_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(GPU_ERROR_T code, const char *file, int line) {
    if (code != GPU_SUCCESS) {
        fprintf(stderr, "GPU Error: %s at %s:%d\n", GPU_GET_ERROR_STR(code), file, line);
    }
}

// Internal communicator structure
typedef struct {
    MPI_Comm mpi_comm;
    int rank;
    int size;
    int device_id;
    void* staging_buffer;
    size_t staging_buffer_size;
} infiniComm_internal;

// Convert InfiniCCL datatype to MPI datatype
static MPI_Datatype infini_to_mpi_datatype(infiniDataType_t datatype) {
    switch(datatype) {
        case infiniFloat32:
            return MPI_FLOAT;
        case infiniFloat64:
            return MPI_DOUBLE;
        case infiniInt32:
            return MPI_INT;
        case infiniInt64:
            return MPI_LONG_LONG;
        case infiniUint64:
            return MPI_UNSIGNED_LONG_LONG;
        default:
            return MPI_BYTE;
    }
}

static MPI_Op infini_to_mpi_op(infiniRedOp_t op) {
    switch(op) {
        case infiniSum:
            return MPI_SUM;
        case infiniProd:
            return MPI_PROD;
        case infiniMax:
            return MPI_MAX;
        case infiniMin:
            return MPI_MIN;
        default:
            return MPI_OP_NULL;
    }
}

// Get datatype size
static size_t infini_get_datatype_size(infiniDataType_t datatype) {
    switch(datatype) {
        case infiniFloat32:
            return 4;
        case infiniFloat64:
            return 8;
        case infiniInt32:
            return 4;
        case infiniInt64:
        case infiniUint64:
            return 8;
        default:
            return 0;
    }
}

// Error string conversion
const char* infiniGetErrorString(infiniResult_t result) {
    switch(result) {
        case infiniSuccess: return "Success";
        case infiniUnhandledCudaError: return "Unhandled GPU error";
        case infiniSystemError: return "System error";
        case infiniInternalError: return "Internal error";
        case infiniInvalidArgument: return "Invalid argument";
        case infiniInvalidUsage: return "Invalid usage";
        case infiniRemoteError: return "Remote error";
        case infiniInProgress: return "Operation in progress";
        default: return "Unknown error";
    }
}

// Core API implementations
infiniResult_t infiniInit(int* argc, char*** argv) {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        if (MPI_Init(argc, argv) != MPI_SUCCESS) {
            return infiniSystemError;
        }
    }
    return infiniSuccess;
}

infiniResult_t infiniFinalize(void) {
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
        if (MPI_Finalize() != MPI_SUCCESS) {
            return infiniSystemError;
        }
    }
    return infiniSuccess;
}

infiniResult_t infiniGetRank(int* rank) {
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    return infiniSuccess;
}

infiniResult_t infiniGetSize(int* size) {
    MPI_Comm_size(MPI_COMM_WORLD, size);
    return infiniSuccess;
}

infiniResult_t infiniCommInitAll(infiniComm_t* comm, int ndev, const int* devlist) {
    infiniComm_internal* comm_internal = static_cast<infiniComm_internal*>(malloc(sizeof(infiniComm_internal)));
    if (!comm_internal) return infiniInternalError;
    
    // Initialize MPI communicator (duplicate COMM_WORLD)
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_internal->mpi_comm);
    MPI_Comm_rank(comm_internal->mpi_comm, &comm_internal->rank);
    MPI_Comm_size(comm_internal->mpi_comm, &comm_internal->size);
    
    // Set GPU device
    int local_rank = 0;
    char* local_rank_str = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (local_rank_str) {
        local_rank = atoi(local_rank_str);
    }
    
    // Use device from devlist or local_rank
    if (devlist && local_rank < ndev) {
        comm_internal->device_id = devlist[local_rank];
    } else {
        comm_internal->device_id = local_rank;
    }
    
    #if defined(USE_CUDA) || defined(USE_MACA)
    GPU_SET_DEVICE(comm_internal->device_id);
    #endif
    
    comm_internal->staging_buffer = NULL;
    comm_internal->staging_buffer_size = 0;
    
    *comm = comm_internal;
    return infiniSuccess;
}

infiniResult_t infiniCommDestroy(infiniComm_t comm) {
    infiniComm_internal* comm_internal = (infiniComm_internal*)comm;
    if (!comm_internal) return infiniSuccess;
    
    MPI_Comm_free(&comm_internal->mpi_comm);
    
    if (comm_internal->staging_buffer) {
        free(comm_internal->staging_buffer);
    }
    
    free(comm_internal);
    return infiniSuccess;
}

// Main AllReduce implementation
// 在 infiniccl.cpp 中修改 infiniAllReduce 函数
infiniResult_t infiniAllReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    infiniDataType_t datatype,
    infiniRedOp_t op,
    infiniComm_t comm,
    void* stream)
{
    if (comm == nullptr) return infiniInvalidArgument;
    if (sendbuff == nullptr || recvbuff == nullptr) return infiniInvalidArgument;
    
    infiniComm_internal* comm_internal = (infiniComm_internal*)comm;
    
    size_t data_size = infini_get_datatype_size(datatype);
    if (data_size == 0) return infiniInvalidArgument;
    
    size_t total_bytes = count * data_size;
    
    MPI_Op mpi_op = infini_to_mpi_op(op);
    MPI_Datatype mpi_datatype = infini_to_mpi_datatype(datatype);
    
    if (mpi_datatype == MPI_DATATYPE_NULL) return infiniInvalidArgument;
    
    void* host_sendbuf = malloc(total_bytes);
    void* host_recvbuf = malloc(total_bytes);
    if (!host_sendbuf || !host_recvbuf) {
        free(host_sendbuf);
        free(host_recvbuf);
        return infiniSystemError;
    }
    
    GPU_MEMCPY_D2H(host_sendbuf, sendbuff, total_bytes);
    GPU_SYNC();
    
    int ret = MPI_Allreduce(host_sendbuf, host_recvbuf, count, 
                           mpi_datatype, mpi_op, comm_internal->mpi_comm);
    if (ret != MPI_SUCCESS) {
        free(host_sendbuf);
        free(host_recvbuf);
        return infiniSystemError;
    }
    
    GPU_MEMCPY_H2D(recvbuff, host_recvbuf, total_bytes);
    GPU_SYNC();
    
    free(host_sendbuf);
    free(host_recvbuf);
    
    return infiniSuccess;
}
