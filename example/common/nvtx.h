#pragma once

#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>

class NvtxRange {
public:
    explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
    ~NvtxRange() { nvtxRangePop(); }
};

#define NVTX_RANGE(name) NvtxRange nvtx_range_##__LINE__(name)

#else

class NvtxRange {
public:
    explicit NvtxRange(const char *) {}
};

#define NVTX_RANGE(name) NvtxRange nvtx_range_##__LINE__(name)

#endif
