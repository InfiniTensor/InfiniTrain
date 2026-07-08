# ==========================================================================
# Generator (random number generator) tests
# ==========================================================================

file(GLOB GENERATOR_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_generator_*.cc)

# CPU tests: run all tests that do NOT have "CUDA" in their test suite name
# or test parameterization prefix.
infini_train_add_test(test_generator
  SOURCES ${GENERATOR_SOURCES}
  LABELS cpu
  TEST_FILTER "-*CUDA*"
)

# CUDA tests: run tests that DO have "CUDA" in their test suite name
# or test parameterization prefix.
if(USE_CUDA)
  infini_train_add_test(test_generator_cuda
    SOURCES ${GENERATOR_SOURCES}
    LABELS cuda
    TEST_FILTER "*CUDA*"
  )
endif()
