#ifndef INFINICCL_H
#define INFINICCL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// InfiniCCL: NCCL-Compatible API using OpenMPI (Simplified)
// ============================================================================

// --- Return Codes ---
typedef enum {
    infiniSuccess                 = 0,
    infiniUnhandledCudaError      = 1,
    infiniSystemError             = 2,
    infiniInternalError           = 3,
    infiniInvalidArgument         = 4,
    infiniInvalidUsage            = 5,
    infiniRemoteError             = 6,
    infiniInProgress              = 7,
    infiniNumResults              = 8
} infiniResult_t;

// --- Data Types (Simplified) ---
typedef enum {
    infiniFloat32    = 0,
    infiniFloat      = 0,
    infiniFloat64    = 1,
    infiniDouble     = 1,
    infiniInt32      = 2,
    infiniInt        = 2,
    infiniInt64      = 3,
    infiniUint64     = 4
} infiniDataType_t;

// --- Reduction Operations (Simplified) ---
typedef enum {
    infiniSum        = 0,
    infiniProd       = 1,
    infiniMax        = 2,
    infiniMin        = 3
} infiniRedOp_t;

// --- Opaque Handles ---
typedef void* infiniComm_t;

// ============================================================================
// Core API Functions (Minimal)
// ============================================================================

// Initialization
infiniResult_t infiniInit(int* argc, char*** argv);
infiniResult_t infiniFinalize(void);

// Rank/Size Query
infiniResult_t infiniGetRank(int* rank);
infiniResult_t infiniGetSize(int* size);

// Communicator Management
infiniResult_t infiniCommInitAll(infiniComm_t* comm, int ndev, const int* devlist);
infiniResult_t infiniCommDestroy(infiniComm_t comm);

// --- Error Handling ---
const char* infiniGetErrorString(infiniResult_t result);

// ============================================================================
// Memory Management Functions
// ============================================================================

infiniResult_t infiniAllocDeviceMemory(void** ptr, size_t size);
infiniResult_t infiniFreeDeviceMemory(void* ptr);
infiniResult_t infiniMemcpyHostToDevice(void* dst, const void* src, size_t size);
infiniResult_t infiniMemcpyDeviceToHost(void* dst, const void* src, size_t size);
infiniResult_t infiniDeviceSynchronize(void);

// ============================================================================
// Collective Operations (Only AllReduce for now)
// ============================================================================

extern "C" infiniResult_t infiniAllReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    infiniDataType_t datatype,
    infiniRedOp_t op,
    infiniComm_t comm,
    void* stream
);

#ifdef __cplusplus
}
#endif

#endif // INFINICCL_H
