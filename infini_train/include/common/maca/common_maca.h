#pragma once

#include <mcr/mc_runtime.h>
#include <mcr/mc_runtime_api.h>
#include <mcblas/mcblas.h>

#ifdef USE_MCCL
#include <mccl.h>
#endif

#include "glog/logging.h"

namespace infini_train::common::maca {

// Common MACA Macros
#define MACA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        mcError_t status = call;                                                                                     \
        if (status != mcSuccess) {                                                                                   \
            LOG(FATAL) << "MACA Error: " << mcGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

#define MCBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        mcblasStatus_t status = call;                                                                                  \
        if (status != MCBLAS_STATUS_SUCCESS) {                                                                         \
            LOG(FATAL) << "MCBLAS Error: " << mcblasGetStatusString(status) << " at " << __FILE__ << ":" << __LINE__;  \
        }                                                                                                              \
    } while (0)

// 尽量把“校验失败”输出得足够可定位：device/ctx/stream/src/dst/bytes/kind
inline void LogMcMemcpyArgs(const char *tag, const void *dst_ptr, const void *src_ptr, size_t nbytes, mcMemcpyKind kind,
                            mcStream_t stream) {
    int cur_dev = -1;
    mcError_t e_dev = mcGetDevice(&cur_dev);

    mcCtx_t ctx = nullptr;
    mcError_t e_ctx = mcCtxGetCurrent(&ctx);

    LOG(ERROR) << tag << " mcGetDevice=" << (int)e_dev << " cur_dev=" << cur_dev << " mcCtxGetCurrent=" << (int)e_ctx
               << " ctx=" << (void *)ctx << " dst_ptr=" << dst_ptr << " src_ptr=" << src_ptr << " bytes=" << nbytes
               << " kind=" << (int)kind << " stream=" << (void *)stream;
}

// 返回 mcError_t，保持“像 runtime API 一样”的风格；
// 上层仍然可以用 MACA_CHECK(...) 直接终止，也可以自己处理错误码。
inline mcError_t CheckMcMemcpyAsyncArgs(const void *dst_ptr, const void *src_ptr, size_t nbytes, mcMemcpyKind kind,
                                        mcStream_t stream, int expected_dst_device /*可传 -1 表示不校验设备一致性*/) {
    // 1) 基本空指针与 size
    if (nbytes == 0) {
        // 0 bytes 复制通常允许，但很多场景下是上游 bug。你可改成 INFO/WARNING 或直接返回 success。
        LOG(WARNING) << "[MACA memcpy CHECK] nbytes==0 (noop).";
        return mcSuccess;
    }
    if (dst_ptr == nullptr || src_ptr == nullptr) {
        LogMcMemcpyArgs("[MACA memcpy CHECK FAIL] nullptr ptr.", dst_ptr, src_ptr, nbytes, kind, stream);
        return mcErrorInvalidValue; // 若你们枚举名不同，改这里即可
    }

    // 2) kind 合法性（按需扩充）
    switch (kind) {
    case mcMemcpyHostToDevice:
    case mcMemcpyDeviceToHost:
    case mcMemcpyDeviceToDevice:
    case mcMemcpyHostToHost:
        break;
    default:
        LogMcMemcpyArgs("[MACA memcpy CHECK FAIL] invalid memcpy kind.", dst_ptr, src_ptr, nbytes, kind, stream);
        return mcErrorInvalidValue;
    }

    // 3) stream 句柄基本校验：用 query 触发 runtime 对 stream 的合法性检查
    //    若你们没有 mcStreamQuery，可改为 mcGetLastError + 其他等价接口
    if (stream == nullptr) {
        // 允许默认流的话，这里可以不当错；但你代码里明确传了 device->Stream()，理论上不应为空
        LogMcMemcpyArgs("[MACA memcpy CHECK FAIL] stream is nullptr.", dst_ptr, src_ptr, nbytes, kind, stream);
        return mcErrorInvalidResourceHandle;
    } else {
        mcError_t e_q = mcStreamQuery(stream);
        if (e_q != mcSuccess && e_q != mcErrorNotReady) {
            // NotReady 说明 stream 正在跑，句柄仍然合法
            LogMcMemcpyArgs("[MACA memcpy CHECK FAIL] mcStreamQuery failed.", dst_ptr, src_ptr, nbytes, kind, stream);
            LOG(ERROR) << "  mcStreamQuery err=" << (int)e_q << " (" << mcGetErrorString(e_q) << ")";
            return e_q;
        }
    }

    // 4) 指针属性校验：Host/Device 类型、所在 device，尽量在 kind 维度上做一致性检查
    //    （如果你们的 MACA runtime 不支持该接口，可用 #ifdef 包起来或在这里返回 success）
    mcPointerAttribute_t src_attr{};
    mcPointerAttribute_t dst_attr{};

    mcError_t e_src_attr = mcPointerGetAttributes(&src_attr, src_ptr);
    mcError_t e_dst_attr = mcPointerGetAttributes(&dst_attr, dst_ptr);

    // 有的 runtime 对“普通 host 指针”可能返回 invalid value，这取决于实现；
    // 这里策略：HostToDevice / DeviceToHost 时允许 host 指针查属性失败，但 device 指针必须能查到。
    auto is_attr_ok = [](mcError_t e) { return e == mcSuccess; };

    if (kind == mcMemcpyHostToDevice) {
        // dst 应是 device
        if (!is_attr_ok(e_dst_attr)) {
            LogMcMemcpyArgs("[MACA memcpy CHECK FAIL] dst_ptr attr query failed (expected device ptr).", dst_ptr,
                            src_ptr, nbytes, kind, stream);
            LOG(ERROR) << "  mcPointerGetAttributes(dst) err=" << (int)e_dst_attr << " ("
                       << mcGetErrorString(e_dst_attr) << ")";
            return e_dst_attr;
        }
        // 可选：检查 dst 在期望 device 上
        if (expected_dst_device >= 0) {
            // 你们结构体字段可能叫 device / deviceOrdinal；按实际改
            if ((int)dst_attr.device != expected_dst_device) {
                LogMcMemcpyArgs("[MACA memcpy CHECK FAIL] dst_ptr on unexpected device.", dst_ptr, src_ptr, nbytes,
                                kind, stream);
                LOG(ERROR) << "  dst_attr.device=" << (int)dst_attr.device << " expected=" << expected_dst_device;
                return mcErrorInvalidDevice;
            }
        }
        // src 是 host：若能查到属性也行；查不到则不强制报错
    } else if (kind == mcMemcpyDeviceToHost) {
        if (!is_attr_ok(e_src_attr)) {
            LogMcMemcpyArgs("[MACA memcpy CHECK FAIL] src_ptr attr query failed (expected device ptr).", dst_ptr,
                            src_ptr, nbytes, kind, stream);
            LOG(ERROR) << "  mcPointerGetAttributes(src) err=" << (int)e_src_attr << " ("
                       << mcGetErrorString(e_src_attr) << ")";
            return e_src_attr;
        }
    } else if (kind == mcMemcpyDeviceToDevice) {
        if (!is_attr_ok(e_src_attr) || !is_attr_ok(e_dst_attr)) {
            LogMcMemcpyArgs("[MACA memcpy CHECK FAIL] D2D requires both ptr attrs.", dst_ptr, src_ptr, nbytes, kind,
                            stream);
            LOG(ERROR) << "  mcPointerGetAttributes(src) err=" << (int)e_src_attr << " ("
                       << mcGetErrorString(e_src_attr) << ")";
            LOG(ERROR) << "  mcPointerGetAttributes(dst) err=" << (int)e_dst_attr << " ("
                       << mcGetErrorString(e_dst_attr) << ")";
            return mcErrorInvalidValue;
        }
        // 可选：同 device 校验、或 peer 可达性校验（若 MACA 有类似 deviceCanAccessPeer 接口）
    } else {
        // H2H：不做额外限制
    }

    // 5) 当前 device / ctx 是否可用：至少保证 mcGetDevice / mcCtxGetCurrent 正常
    //    （你之前已经在日志里打印过，这里只做 hard check）
    int cur_dev = -1;
    mcError_t e_dev = mcGetDevice(&cur_dev);
    mcCtx_t ctx = nullptr;
    mcError_t e_ctx = mcCtxGetCurrent(&ctx);
    if (e_dev != mcSuccess || e_ctx != mcSuccess || ctx == nullptr) {
        LogMcMemcpyArgs("[MACA memcpy CHECK FAIL] runtime device/ctx invalid.", dst_ptr, src_ptr, nbytes, kind, stream);
        LOG(ERROR) << "  mcGetDevice err=" << (int)e_dev << " (" << mcGetErrorString(e_dev) << ")";
        LOG(ERROR) << "  mcCtxGetCurrent err=" << (int)e_ctx << " (" << mcGetErrorString(e_ctx) << ")"
                   << " ctx=" << (void *)ctx;
        return (e_ctx != mcSuccess) ? e_ctx : e_dev;
    }

    return mcSuccess;
}


#ifdef USE_MCCL
#define MCCL_CHECK(expr)                                                                                               \
    do {                                                                                                               \
        mcclResult_t _status = (expr);                                                                                 \
        if (_status != mcclSuccess) {                                                                                  \
            LOG(FATAL) << "MCCL error: " << mcclGetErrorString(_status) << " at " << __FILE__ << ":" << __LINE__       \
                       << " (" << #expr << ")";                                                                        \
        }                                                                                                              \
    } while (0)
#endif

} // namespace infini_train::common::maca
