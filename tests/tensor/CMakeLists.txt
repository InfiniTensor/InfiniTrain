# ==========================================================================
# Tensor tests
# ==========================================================================
#
# File layout:
#   tests/tensor/*.cc            shared tests (compiled for both CPU and CUDA)
#   tests/tensor/cpu_only/*.cc   CPU-only tests (test_tensor_cpu_only target)
#   tests/tensor/cuda_only/*.cc  CUDA-only tests (test_tensor_cuda_only, USE_CUDA=ON only)

# Shared parameterized tests — produces test_tensor_cpu / test_tensor_cuda.
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

# CUDA-only tests — only built when USE_CUDA is enabled.
if(USE_CUDA)
  file(GLOB TENSOR_CUDA_ONLY_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cuda_only/test_*.cc)
  if(TENSOR_CUDA_ONLY_SOURCES)
    infini_train_add_test(test_tensor_cuda_only
      SOURCES ${TENSOR_CUDA_ONLY_SOURCES}
      LABELS cuda
    )
  endif()
endif()
