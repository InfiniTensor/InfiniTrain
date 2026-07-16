#pragma once

#include <cstdint>

// FlashAttention keeps this type in its parameter ABI even when dropout is
// compiled out. InfiniTrain only supports dropout=0 in the native adapter, so
// no generator state is acquired or consumed.
// FIXME(zbl): This minimal stand-in is coupled to the pinned FlashAttention
// headers. Revalidate it when upgrading FlashAttention or including real ATen,
// since layout/API changes may break compilation or conflict with ATen's type.
namespace at {

struct PhiloxCudaState {
    uint64_t seed = 0;
    uint64_t offset = 0;
};

} // namespace at
