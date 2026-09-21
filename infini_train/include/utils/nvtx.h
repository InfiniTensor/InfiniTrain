#pragma once

// NVTX support for nsys timeline analysis.
//
// NVTX records host-side timestamps only and issues no CUDA call. This is what
// distinguishes it from PROFILE_MODE: the profiler wraps every dispatched
// kernel in SynchronizeStream + EventRecord/EventSynchronize, which serializes
// the pipeline and both inflates per-operator device time and hides every
// asynchronous-overlap and host-drain effect. NVTX leaves the execution shape
// intact, so ranges can be read directly against the CUDA kernel track.
//
// Everything below compiles away to nothing unless -DNVTX_MODE=1 is passed,
// which keeps CPU-only builds free of any CUDA include-path dependency. Callers
// should therefore use INFINI_TRAIN_NVTX_PUSH/POP unconditionally.

#ifdef NVTX_MODE

// The nvtx3 C API is header-only; nvtxRangePushA/Pop resolve through a runtime
// dlopen of the injection library, so no nvToolsExt library needs linking
// (libnvToolsExt.so was removed in CUDA 13).
#include <nvtx3/nvToolsExt.h>

namespace infini_train {
namespace utils {

// RAII wrapper around the NVTX push/pop stack, for ranges whose extent matches
// a C++ scope. Preferred over the macros below: it keeps the push/pop balanced
// even if the enclosed code throws.
class NvtxRange {
public:
    explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }

    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;
};

} // namespace utils
} // namespace infini_train

#define INFINI_TRAIN_NVTX_PUSH(name) ::nvtxRangePushA(name)
#define INFINI_TRAIN_NVTX_POP() ::nvtxRangePop()

#else

#define INFINI_TRAIN_NVTX_PUSH(name) ((void)0)
#define INFINI_TRAIN_NVTX_POP() ((void)0)

#endif
