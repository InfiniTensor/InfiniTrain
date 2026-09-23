# ==========================================================================
# Tensor tests
# ==========================================================================
#
# File layout:
#   tests/tensor/*.cc            shared tests (compiled for all configured devices)
#   tests/tensor/cpu_only/*.cc   CPU-only tests (test_tensor_cpu_only target)

# Shared parameterized tests - produces one target per configured test device.
file(GLOB TENSOR_SHARED_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc)
infini_train_add_test_suite(test_tensor SOURCES ${TENSOR_SHARED_SOURCES})

# CPU-only tests — independent target.
file(GLOB TENSOR_CPU_ONLY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cpu_only/test_*.cc)
if(TENSOR_CPU_ONLY_SOURCES)
  infini_train_add_test(test_tensor_cpu_only
    SOURCES ${TENSOR_CPU_ONLY_SOURCES}
    LABELS cpu
  )
endif()
