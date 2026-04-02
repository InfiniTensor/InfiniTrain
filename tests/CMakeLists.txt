# Tests CMakeLists.txt
# This file manages the test infrastructure for InfiniTrain

# Include shared test macros (must be before any test subdirectory)
include(${CMAKE_CURRENT_SOURCE_DIR}/common/test_macros.cmake)

# Common test utilities
add_subdirectory(common)

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

# Slow label tests
add_subdirectory(slow)
