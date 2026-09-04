# Tests CMakeLists.txt
# This file manages the test infrastructure for InfiniTrain

# Include shared test macros (must be before any test subdirectory)
include(${PROJECT_SOURCE_DIR}/cmake/test_macros.cmake)

# Common test utilities
add_subdirectory(common)

# Distributed tests
add_subdirectory(distributed)
# Module tests
add_subdirectory(module)
# Backend extension contract tests
add_subdirectory(backend)

# Tensor tests
add_subdirectory(tensor)

# Optimizer tests
add_subdirectory(optimizer)

# Autograd operator tests
add_subdirectory(autograd)

# LoRA tests
add_subdirectory(lora)

# Hook tests
add_subdirectory(hook)

# Dtype dispatch tests
add_subdirectory(dtype)

# Transformer architecture tests
add_subdirectory(transformer)

# Checkpoint tests
add_subdirectory(checkpoint)

# Suite declaration above is device-independent. Instantiate the built-in
# devices after every test subdirectory has registered its sources.
_infini_train_instantiate_test_suites(
  SUITE_REGISTRY INFINI_TRAIN_CPU_TEST_SUITES
  TARGET_SUFFIX cpu
  GTEST_PREFIX CPU
  DEVICE_TYPE kCPU
  DEVICE_INDEX 0
  LABELS cpu
)

if(USE_CUDA)
  _infini_train_instantiate_test_suites(
    SUITE_REGISTRY INFINI_TRAIN_CUDA_TEST_SUITES
    TARGET_SUFFIX cuda
    GTEST_PREFIX CUDA
    DEVICE_TYPE kCUDA
    DEVICE_INDEX 0
    LABELS cuda accelerator hardware
    ACCELERATOR
  )
endif()
