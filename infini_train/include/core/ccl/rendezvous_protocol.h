#pragma once

#include <stdint.h>

namespace infini_train::core::rendezvous {
inline constexpr uint32_t kMagic = 0x49545244;
inline constexpr uint32_t kMaxKeySize = 4096;
inline constexpr uint32_t kMaxValueSize = 1024 * 1024;

enum class Operation : uint32_t {
  kPut = 1,
  kGet = 2,
};

struct Header {
  uint32_t magic;
  uint32_t operation;
  uint32_t key_size;
  uint32_t value_size;
};

struct Response {
  uint32_t status;
  uint32_t value_size;
};
}  // namespace infini_train::core::rendezvous
